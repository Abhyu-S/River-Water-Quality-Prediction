### **Project Report: Spatiotemporal Generalization in Satellite-Derived River pH Modeling**

**Objective**  
To determine if a universal "Global" machine learning model can accurately predict river pH from Sentinel-2 satellite data across the entire Ganga basin, or if distinct hydro-optical regimes require a "Hybrid Zonal" approach.  
**Executive Summary**  
Our investigation demonstrates that a single, universal inversion algorithm fails to generalize across the full length of the river ($R^2 \\approx \-0.19$ on unseen locations). This confirms the presence of significant **Spatial Non-Stationarity**—meaning the relationship between water color and pH changes physically as the river flows from upstream to downstream.  
Consequently, we developed a **Hybrid Zonal Model** that uses unsupervised clustering to identify these distinct optical regimes. This approach successfully raised predictive accuracy to **$R^2 \= 0.59$**, proving that while a "One-Size-Fits-All" model is impossible, a "Regime-Specific" approach is highly effective.

---

**1\. Methodology & Forensic Analysis**

We conducted a three-stage ablation study to isolate the source of predictive power and test for spatial overfitting.  
**Experiment A: The Global Baseline (Physics-Only)**

* **Hypothesis:** A single physical relationship (e.g., "Greener water \= Higher pH") applies everywhere.  
* **Method:** Trained an XGBoost model using *only* spectral indices (NDWI, NDTI, etc.) on 80% of the data.  
* **Result:** **$R^2 \\approx 0.39$**.  
* **Interpretation:** The model struggled. The low score indicates that the optical properties of the water are too complex or inconsistent for a simple global equation.

**Experiment B: The Spatial Stress Test (Generalization)**

* **Hypothesis:** If the model has learned true physics, it should work on a river segment it has never seen before.  
* **Method:** We performed "Leave-One-Group-Out" Cross-Validation. We divided the river into 4 distinct spatial segments, training on 3 and testing on the 4th.  
* **Result:** **$R^2 \\approx \-0.19$ (Negative Score)**.  
* **Conclusion:** The model failed catastrophically on unseen locations. This proves the existence of **Concept Drift**: the specific correlation between satellite color and pH in Varanasi does not hold true for Patna. This scientifically invalidates the use of a purely global model for this river.

**Experiment C: The Space-Time Reality Check**

* **Hypothesis:** Is the model just memorizing the calendar?  
* **Method:** We trained a dummy model using *only* Location and Time (Lat, Lon, Month), deleting all satellite data.  
* **Result:** **$R^2 \\approx 0.40$**.  
* **Interpretation:** 40% of the variance in pH is driven purely by seasonal flow and static industrial zones. Any valid satellite model must beat this "Memorization Baseline."

---

**2\. The Solution: Hybrid Zonal Modeling**

Since Experiment B proved that water physics varies by location, we rejected the Global Model in favor of a **Hybrid Zonal Architecture**.

* **Approach:** We employed Unsupervised K-Means Clustering on spatial coordinates to define dynamic **"Optical Water Types" (OWTs)** or Hydro-Zones.  
* **Mechanism:** Instead of forcing one equation on the whole river, the model first identifies the "Zone" (Upstream/Downstream/Urban), then applies a specialized calibration for that regime.  
* **Final Result:** **$R^2 \= 0.59$**.  
* **Impact:**  
  * The Hybrid Model outperformed the "Memorization Baseline" (0.40) by **\~0.19**.  
  * This confirms that the satellite data *is* contributing significant value, but only when constrained within the correct local optical context.

---

**3\. Conclusion**

The investigation confirms that the Ganga river system is too optically heterogeneous for a single "Universal" remote sensing model. The failure of the spatial generalization test ($R^2 \= \-0.19$) was a critical finding that justified the shift to Zonal Modeling.  
By acknowledging this spatial non-stationarity and implementing a Hybrid Zonal framework, we achieved a robust **$R^2 \= 0.59$**. This represents a statistically significant improvement over both the global baseline and pure spatiotemporal memorization, successfully demonstrating the viability of "Region-Specific" satellite monitoring.
