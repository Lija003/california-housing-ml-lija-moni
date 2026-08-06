# California Housing Price Prediction — Project Summary

**Author:** Lija Moni
**Dataset:** California Housing Prices (1990 census, 20,640 block groups, 10 features)

## Approach

I followed the standard end-to-end regression workflow: explore, clean, engineer features, train and evaluate.

- **EDA** showed `median_income` and location (longitude/latitude) as the strongest drivers of house value, with a clear coastal price premium visible on a geographic scatter plot. The target variable (`median_house_value`) is capped at $500,001, which limits how well any model can predict the priciest neighborhoods.
- **Cleaning:** The 207 missing `total_bedrooms` values (~1% of rows) were filled with the column median rather than dropped, since the gaps didn't look systematic and the dataset is small enough that losing rows wasn't worth it.
- **Encoding & feature engineering:** `ocean_proximity` was one-hot encoded. Three ratio features were added — `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` — since raw counts alone are hard to compare across differently sized block groups.
- **Modeling:** Two models were trained on an 80/20 train/test split — Linear Regression (scaled features) as a baseline, and a Random Forest Regressor (200 trees) to capture non-linear effects like the coastal price premium.

## Best Model & Performance

**Random Forest Regressor** was the best-performing model:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $50,889 | $72,669 | 0.597 |
| **Random Forest** | **$32,139** | **$50,078** | **0.809** |

The Random Forest explains about 81% of the variance in house prices and is off by roughly $32,000 on average — a large improvement over the linear baseline, driven by its ability to capture non-linear location effects that a straight-line model can't.

## Limitation & Next Step

The most significant limitation is the **capped target variable**: house values above $500,001 were clipped in the original 1990 census data, so the model systematically underpredicts the most expensive neighborhoods — no amount of tuning can recover information that was truncated at the source. With more time, I would either exclude or separately model the capped rows and try gradient boosting (XGBoost/LightGBM) with proper hyperparameter tuning and cross-validation to squeeze out further performance.
