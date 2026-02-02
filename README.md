### Models on the Web

Most high R2 scores (0.80–0.90) you see online are typically from lakes or reservoirs, not rivers. Rivers are significantly harder to model due to flow dynamics, sediment mixing, and "mixed pixel" effects (land contamination in the pixel).

1. (Similar Performance): A study using Support Vector Machines (SVM) on river data achieved an R2 of 0.647 for pH prediction. our 0.59 is statistically comparable to this, meaning our model performs near the standard for non-deep learning approaches on flowing water.
2. (The "Lake" Bias): A study on reservoirs (static water) using Sentinel-2 and Landsat-8 achieved an R2 > 0.80 for pH. Our problem (Rivers) is harder than their problem (Lakes).
3. (Model Superiority): Research comparing multiple models for inland water quality (Tian et al.) found that XGBoost outperformed Random Forest, SVM, and ANN in retrieving water parameters.
---
### Why this approach is better?

1. pH is not "optically active." Unlike algae (green) or sediment (brown), pH doesn't change the color of water directly. Satellite models can only guess pH by finding hidden correlations with other things that do have color (like Chlorophyll).
2. Our Edge: Many projects claim high accuracy for pH (0.9+) by overfitting to a specific lake. This approach acknowledges the indirect core nature of the problem.
3. A river in the mountains has different optical properties than the same river in the plains. A standard model treats them as the same. Unlike standard 'global' models that fail when river geography changes, my analysis specifically tackles spatial non-stationarity, recognizing that the relationship between satellite bands and pH shifts as the river flows downstream.
4. eep Learning (CNNs) requires massive datasets which don't exist for river ground-truth pH. So XGBoost was the preferable choice here.
