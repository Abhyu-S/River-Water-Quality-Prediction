# -*- coding: utf-8 -*-
"""
Production-Quality Hybrid Spatial XGBoost Pipeline
- Implements strictly safe Spatial Clustering (No Leakage).
- Uses Custom Scikit-Learn Transformer for Zonal Feature Engineering.
- 5-Fold Cross-Validation with RandomizedSearchCV.
- Evaluates on a strict Hold-Out Test Set.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings

# Sklearn Tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
INPUT_FILE = "Final_Dataset_For_ML.csv"
OUTPUT_DIR = "result_production_hybrid"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Initializing Production Hybrid Pipeline...")
print("-" * 50)

# ---------------------------------------------
# 1. LOAD DATA & BASIC PHYSICS
# ---------------------------------------------
# We can do row-independent physics calculations before splitting
# because they don't depend on other rows (no leakage).

print("[1/5] Loading and Calculating Physics Indices...")
df = pd.read_csv(INPUT_FILE)
epsilon = 1e-6

# --- A. Physics Indices ---
df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + epsilon)
df['MNDWI'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'] + epsilon)
df['NDTI'] = (df['B4'] - df['B3']) / (df['B4'] + df['B3'] + epsilon)
df['TURB_1'] = df['B4'] / (df['B3'] + epsilon)
df['TURB_2'] = df['B4'] / (df['B2'] + epsilon)
df['CDOM_Proxy'] = df['B2'] / (df['B3'] + epsilon)
df['NDCI'] = (df['B5'] - df['B4']) / (df['B5'] + df['B4'] + epsilon)
df['Total_Bright'] = df[['B2', 'B3', 'B4', 'B8']].sum(axis=1)
df['Rat_B4_B11'] = df['B4'] / (df['B11'] + epsilon)

# --- B. Temporal Features ---
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

def get_season(m):
    if m in [6,7,8,9]: return 1 # Monsoon
    if m in [10,11]: return 2   # Post-Monsoon
    if m in [12,1,2]: return 3  # Winter
    return 4                    # Pre-Monsoon
df['Season'] = df['Month'].apply(get_season)

# --- C. Initial Cleanup ---
# We KEEP 'lat' and 'lon' for now because the Transformer needs them later.
# We Drop metadata that is truly useless.
meta_cols = ['system:index', '.geo', 'id', 'Location']
df = df.drop(columns=[c for c in meta_cols if c in df.columns])

# Separate Target
X = df.drop(columns=['pH'])
y = df['pH']

print(f"    > Base Data Shape: {X.shape}")

# ---------------------------------------------
# 2. STRICT SPLIT (Hold-out Set)
# ---------------------------------------------
print("\n[2/5] Splitting Data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    > Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ---------------------------------------------
# 3. CUSTOM SPATIAL TRANSFORMER
# ---------------------------------------------
# This class ensures K-Means is fit ONLY on training data
# and then applied consistent logic to test data.

class ZonalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.interaction_cols = ['NDTI', 'NDCI', 'TURB_1', 'Total_Bright']
        
    def fit(self, X, y=None):
        # Fit KMeans only on Lat/Lon
        coords = X[['lat', 'lon']]
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(coords)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        coords = X_copy[['lat', 'lon']]
        
        # 1. Predict Zone using the fitted KMeans
        hydro_zones = self.kmeans.predict(coords)
        X_copy['Hydro_Zone'] = hydro_zones
        
        # 2. Create Interactions (Zone * Physics)
        # Note: Multiplying by cluster ID (0,1,2) is a bit unusual mathematically,
        # but I am preserving my original logic here.
        for col in self.interaction_cols:
            if col in X_copy.columns:
                X_copy[f'Zone_x_{col}'] = X_copy['Hydro_Zone'] * X_copy[col]
        
        # 3. Drop Lat/Lon now that we are done with them
        X_copy = X_copy.drop(columns=['lat', 'lon'])
        
        # 4. One-Hot Encode 'Season' and 'Hydro_Zone'
        # We use pd.get_dummies, but we must ensure columns match training.
        # However, inside a pipeline, simple get_dummies can break if a category is missing.
        # For robustness in this script, we'll stick to simple numeric encoding for Tree models
        # or just let XGBoost handle the numeric 'Hydro_Zone' column directly.
        X_copy = pd.get_dummies(X_copy, columns=['Season', 'Hydro_Zone'], drop_first=True)
        
        return X_copy

# ---------------------------------------------
# 4. BUILD PIPELINE
# ---------------------------------------------
print("\n[3/5] Building Pipeline...")

# Pipeline: 
# 1. Generate Zones (Fit on Train, Apply to Test)
# 2. Scale (Safe scaling)
# 3. XGBoost
pipeline = Pipeline([
    ('zonal', ZonalFeatureEngineer(n_clusters=4)),
    ('scaler', StandardScaler()), # Helpful for the interaction terms
    ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42))
])

# ---------------------------------------------
# 5. RANDOMIZED SEARCH (5-Fold CV)
# ---------------------------------------------
print("\n[4/5] Starting RandomizedSearchCV...")

param_dist = {
    'xgb__n_estimators': [500, 1000],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth': [4, 6, 8],
    'xgb__subsample': [0.7, 0.8],
    'xgb__colsample_bytree': [0.7, 0.8],
    # 'zonal__n_clusters': [3, 4, 5] # We could even tune the number of zones!
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,          # 20 iterations
    cv=5,               # 5-Fold Cross Validation
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# This will trigger the ZonalFeatureEngineer.fit() 5 times (once per fold)
# ensuring no leakage occurs during validation.
search.fit(X_train, y_train)

print(f"    > Best CV R² Score: {search.best_score_:.4f}")
print(f"    > Best Params: {search.best_params_}")

# ---------------------------------------------
# 6. FINAL EVALUATION
# ---------------------------------------------
print("\n[5/5] Final Test Evaluation...")

# Retrains best model on full X_train automatically
best_model = search.best_estimator_

# The 'predict' calls 'transform' internally on X_test,
# using the Clusters learned from X_train.
preds_test = best_model.predict(X_test)

r2 = r2_score(y_test, preds_test)
rmse = np.sqrt(mean_squared_error(y_test, preds_test))

print("-" * 50)
print(f"FINAL TEST RESULTS (Hybrid Production)")
print("-" * 50)
print(f"  > R² Score: {r2:.4f}")
print(f"  > RMSE:     {rmse:.4f}")
print("=" * 50)

# Save
joblib.dump(best_model, f"{OUTPUT_DIR}/best_hybrid_pipeline.pkl")
print(f"\nSaved robust pipeline to {OUTPUT_DIR}")