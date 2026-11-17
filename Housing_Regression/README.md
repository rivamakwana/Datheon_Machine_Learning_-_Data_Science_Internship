# California Housing: Predicting Median House Value

Regression project comparing **OLS**, **Ridge (L2)** and **Lasso (L1)** for predicting district-level median house value in California, with feature engineering and multicollinearity diagnosis (VIF) along the way.

Everything lives in one Jupyter notebook: `housing_regression.ipynb`.

## Dataset

- **Source file:** `housing.csv` (California Housing)
- **Size:** 20,640 districts, 10 columns
- **Target:** `median_house_value`
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and the categorical `ocean_proximity`
- **Data quirks**
  - `total_bedrooms` has 207 missing values (1%).
  - The target is censored at $500,001 (965 districts sit at the cap).
  - `ocean_proximity = ISLAND` has only 5 rows.

## Workflow

1. **Load and inspect** the data.
2. **Handle missing values** with median imputation. For EDA the whole dataset is imputed; for modelling the imputer sits inside the pipeline and is fitted on training folds only, so there is no leakage.
3. **Explore** with histograms and correlation heatmaps. `median_income` is by far the strongest predictor (r = 0.69).
4. **Engineer features**
   - Ratios: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
   - Centered interactions and non-linear terms: `income_x_age`, `income_x_rooms_per_hh`, `income_sq`, `lat_x_lon`
5. **Diagnose multicollinearity** with a hand-rolled Variance Inflation Factor. The raw count columns (`total_rooms`, `total_bedrooms`, `households`, VIF 18 to 34) are redundant with the new ratios and are dropped. The remaining collinearity (mainly latitude/longitude, VIF about 10) is kept and left to regularization.
6. **Preprocess** with a `ColumnTransformer`: median impute and standardise numeric features, one-hot encode `ocean_proximity` (`drop="first"`).
7. **Fit models** on an 80/20 split (16,512 train / 4,128 test, `random_state=42`)
   - OLS baseline
   - Ridge and Lasso tuned with `GridSearchCV` (5-fold CV, scored by RMSE)
8. **Evaluate** with RMSE, MAE and R², then plot predicted vs actual, residual distributions, coefficient comparisons and regularization paths.

## Results

Test-set performance:

| Model      | Test RMSE | Test MAE | Test R² | Best alpha |
|------------|----------:|---------:|--------:|-----------:|
| OLS        |    71,291 |   51,186 |  0.6122 |        n/a |
| Ridge (L2) |    71,300 |   51,193 |  0.6120 |        1.0 |
| Lasso (L1) |    71,302 |   51,195 |  0.6120 |       10.0 |

## Key Findings

- **All three models perform essentially the same** (R² about 0.61, RMSE about $71k). With about 16.5k training rows and 16 features, OLS has little variance for regularization to reduce.
- **Multicollinearity was solved mostly by feature design:** dropping redundant raw counts and centering variables before building interactions.
- **Regularization still helps where collinearity is real:** the `ocean_proximity_ISLAND` coefficient (5 samples) shrinks from about 145k (OLS) to 116k (Ridge) and 104k (Lasso). At the CV-optimal alpha Lasso keeps all 16 features; the coefficient-path plot shows it zeroing weak features as alpha grows.
- **Residuals** are centred near zero but have a heavy right tail and a diagonal band, caused by the censored target and non-linear geography that a linear model cannot capture.
- **The ceiling is model form, not overfitting.** Tree-based models (e.g. gradient boosting) or distance-to-city features would likely help far more than tuning alpha.

## Getting Started

### Requirements

- numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run

1. Put `housing.csv` in the same folder as the notebook (or edit the path in the data-loading cell).
2. Launch Jupyter and run all cells:

```bash
jupyter notebook housing_regression.ipynb
```

## Project Structure

```
.
├── 2_housing_regression.ipynb   # Full analysis, models and plots
├── housing.csv                  # Dataset
└── README.md
```
