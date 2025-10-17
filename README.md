# Football Analytics: Goal Scoring Classification with Machine Learning

> This project was developed as part of the **Data Mining** course assignment in the **Master’s program**. It demonstrates a complete machine learning pipeline with model comparison and hyperparameter tuning.

This project implements an end-to-end machine learning pipeline to classify professional football players...

---

## Project Overview
- **Problem Type:** Multi-class classification  
- **Goal:** Predict a player’s goal-scoring tier  
- **Target Labels:** Low (0–2), Medium (3–7), High (8+)  
- **Dataset Size:** 6,824 players, 46 features  
- **Area:** Football performance analytics  
- **Stack:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, imbalanced-learn (SMOTE)

---

## Dataset
The dataset contains professional player statistics across top European leagues.

**Example features**
- `age`, `height`, `position`
- Match metrics: `minutes`, `assists`, `shots total`, `dribbles completed`, `pressures`, `tackles`
- Team/league info: `league`, `squad`, `Season`
- Engineered: `played_CL` (binary), `goal_category` (target)

**Notes**
- One missing value in `height` → mean imputation  
- `goal_category` derived from `goals`: Low ≤2, Medium 3–7, High >7  
- `played_CL` created from `CLBestScorer` non-null indicator  
- Identifiers removed from features: `player`, `squad`, `Season`, `league`  
- Direct leakage removed from features: `goals`

---

## Data Preprocessing
- Train/test split: **70/30**, stratified by target  
- Numeric pipeline: `SimpleImputer(strategy="mean")` → `StandardScaler`  
- Categorical pipeline: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`  
- Combined via `ColumnTransformer`  
- Class imbalance addressed **only on training set** using **SMOTE**

**Class balance**
| Class  | Before SMOTE | After SMOTE |
|--------|---------------|-------------|
| Low    | 3720          | 3720        |
| Medium | 761           | 3720        |
| High   | 295           | 3720        |

---

## Models
Three classifiers from different families:
- **Logistic Regression** (multinomial baseline)
- **Random Forest** (ensemble trees)
- **Support Vector Machine** (linear baseline; RBF after tuning)

**Evaluation metrics**: Accuracy, Precision, Recall, F1 (weighted), Confusion Matrix.

---

## Results (Before Tuning)
| Model                | Accuracy | Weighted F1 |
|----------------------|----------|-------------|
| Logistic Regression  | 0.82     | 0.83        |
| Random Forest        | **0.85** | **0.85**    |
| SVM (Linear)         | 0.81     | 0.83        |

**Best baseline:** Random Forest.

---

## Hyperparameter Tuning
Performed with **GridSearchCV** (StratifiedKFold, 5-fold) optimizing **weighted F1**.

**Logistic Regression**
- Best: `C=100`, `penalty=l2`, `solver=lbfgs`, `max_iter=2000`
- Test: Acc ≈ 0.814, F1 ≈ 0.829 (minor change vs baseline)

**Random Forest**
- Grid: `n_estimators` {100, 200, 500}, `max_depth` {None, 10, 20, 30}, `max_features` {sqrt, log2}
- Best: `n_estimators=200`, `max_depth=30`, `max_features=sqrt`
- Test: Acc ≈ 0.847, F1 ≈ 0.852 (stable, strong)

**SVM**
- Grid: linear and RBF kernels; `C` ∈ {0.1, 1, 10}; `gamma` ∈ {scale, auto} for RBF
- Best: **RBF**, `C=10`, `gamma=scale`
- Test: Acc ≈ 0.837, F1 ≈ 0.842 (clear improvement over linear)

**Final choice:** Random Forest (robust across metrics on imbalanced problem), with tuned SVM (RBF) close behind.

---

## Reproducibility
- Self-contained Jupyter notebook: `notebooks/football_goal_scoring_ml.ipynb`  
- Reads the dataset via **relative path**: `data/Output.csv`  
- Random seeds set where applicable (`random_state=42`)  
- All steps (EDA → preprocessing → SMOTE → training → tuning → evaluation) run end-to-end without errors

---

## Project Structure

project-directory
│
├── data
│ └── Output.csv
│
├── notebooks
│ └── football_goal_scoring_ml.ipynb
│
├── README.md
└── requirements.txt


---

## Future Work

Potential improvements:

- Add advanced models such as XGBoost and LightGBM

- Use cross-validation instead of train-test split only

- Perform SHAP-based model explainability

- Deploy as a web app using Streamlit

- Improve feature engineering and domain insight

## Author

Merve Gülfer Baytekin
