# Telco Customer Churn Prediction: Random Forest + XGBoost Ensemble

A binary classification project that predicts which telecom customers will churn. The goal was a **test ROC-AUC above 0.80**, which every model reached.

Everything lives in one Jupyter notebook: `Telco_Churn_Prediction.ipynb`.

## Dataset

- **Source file:** `Telco-Customer-Churn.csv`
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (Yes/No), encoded as 1/0. About **26.5%** of customers churn, so the classes are imbalanced.
- **Features:** demographics, account info (tenure, contract, payment method, charges) and subscribed services. 15 categorical and 4 numeric features after dropping `customerID`.

## Workflow

1. **Clean the data**
   - `TotalCharges` is stored as text; 11 blank values are converted to NaN. All 11 rows have `tenure = 0` (new customers not yet billed), so they are filled with 0.
   - Drop `customerID` (no predictive value).
2. **Explore** the class imbalance and churn rate by contract type.
3. **Preprocess** with a stratified 80/20 split (5,634 train / 1,409 test) and `OneHotEncoder` for categorical columns; numeric columns pass through unchanged.
4. **Handle imbalance** two ways and compare them
   - **SMOTE**, placed *inside* an `imblearn` Pipeline so it only touches the training portion of each CV fold (no leakage).
   - XGBoost **`scale_pos_weight`** (about 2.77), with no resampling.
5. **Train three pipelines** with stratified 5-fold cross-validation scored by ROC-AUC
   - Random Forest + SMOTE
   - XGBoost + SMOTE
   - XGBoost + `scale_pos_weight`
6. **Build a soft-voting ensemble** that averages the predicted probabilities of Random Forest + XGBoost (both SMOTE).
7. **Evaluate** on the hold-out test set with ROC curves, a classification report, a confusion matrix and the top 10 feature importances.

## Results

| Model                          | CV ROC-AUC (5-fold) | Test ROC-AUC |
|--------------------------------|--------------------:|-------------:|
| XGBoost + SMOTE                |     0.8468 ± 0.0115 |   **0.8459** |
| XGBoost + scale_pos_weight     |     0.8476 ± 0.0106 |       0.8457 |
| Ensemble (RF + XGB, soft vote) |                   – |       0.8427 |
| Random Forest + SMOTE          |     0.8408 ± 0.0091 |       0.8358 |

Ensemble at a 0.5 threshold on the test set: accuracy 0.79, churn precision 0.61, churn recall 0.59, churn F1 0.60.

## Key Findings

- **The target is met by every model**, with XGBoost + SMOTE the best single model (0.8459 test AUC).
- **The ensemble did not beat XGBoost alone** (0.8427 vs 0.8459), since the weaker Random Forest pulls its average down slightly.
- **SMOTE and class weighting perform almost identically** for XGBoost (0.8459 vs 0.8457), so the simpler class-weight option is a good alternative.
- **Top churn drivers**, consistent across both models: month-to-month contracts, no online security, no tech support and electronic-check payment. Tenure and total/monthly charges also rank high in the Random Forest.
- The model catches about 59% of churners at the default threshold, so lowering the threshold could trade precision for more recall.

## Getting Started

### Requirements

- numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, jupyter

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

### Run

1. Put `Telco-Customer-Churn.csv` in the same folder as the notebook (or edit the path in the "Load the dataset" cell).
2. Launch Jupyter and run all cells:

```bash
jupyter notebook Telco_Churn_Prediction.ipynb
```

Cross-validation runs sequentially (`n_jobs=1`) because of SMOTE inside the pipeline, so expect a few minutes.

## Project Structure

```
.
├── Telco_Churn_Prediction.ipynb   # Full analysis, models and plots
├── Telco-Customer-Churn.csv         # Dataset
└── README.md
