# Iris Classification: Logistic Regression vs Decision Tree

A supervised machine-learning project that classifies iris flowers into three species (*setosa*, *versicolor*, *virginica*) from four measurements, and benchmarks two classic models against each other.

Everything lives in one Jupyter notebook: `Iris_Classification.ipynb`.

## Dataset

- **Source file:** `1794206110138_iris.csv` (the classic Iris dataset)
- **Size:** 150 rows, 5 columns, no missing values
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm)
- **Target:** `species`, balanced at 50 samples per class

## Workflow

1. **Load and inspect** the data (shape, dtypes, missing values, class balance, summary stats).
2. **Exploratory analysis** with a seaborn pairplot and a feature-correlation heatmap.
3. **Prepare the data**
   - Encode species labels with `LabelEncoder` (setosa=0, versicolor=1, virginica=2).
   - Stratified **80/20 train/test split** (120 train / 30 test, `random_state=42`).
   - Standardise features with `StandardScaler`, **fit on the training set only** to avoid data leakage.
4. **Train two models**
   - `LogisticRegression(max_iter=1000)`
   - `DecisionTreeClassifier(max_depth=3)`
5. **Evaluate** with accuracy, precision, recall and F1-score (weighted averages), plus per-class classification reports.
6. **Visualise** model comparison bars, confusion-matrix heatmaps, 2D decision boundaries and the fitted decision tree.

## Results

Test-set performance (30 samples):

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression |   0.9333 |    0.9333 | 0.9333 |   0.9333 |
| Decision Tree       |   0.9667 |    0.9697 | 0.9667 |   0.9666 |

## Key Findings

- **Setosa** is perfectly separable; both models classify it with no errors.
- All mistakes are between **versicolor** and **virginica**, whose feature ranges overlap.
- The Decision Tree edged out Logistic Regression on this split, but with only 30 test samples a single flower shifts accuracy by 3.3%. Cross-validation would give a more reliable comparison.
- The decision-boundary plots retrain both models on **petal length and petal width** only, since boundaries can only be drawn in 2D. They are for illustration and are separate from the four-feature models evaluated above.

## Getting Started

### Requirements

- numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run

1. Place `1794206110138_iris.csv` in the same folder as the notebook (or edit the path in the "Load the dataset" cell).
2. Launch Jupyter and run all cells:

```bash
jupyter notebook 1_Iris_Classification.ipynb
```

## Project Structure

```
.
├── 1_Iris_Classification.ipynb   # Full analysis, models and plots
├── 1794206110138_iris.csv        # Dataset
└── README.md
```
