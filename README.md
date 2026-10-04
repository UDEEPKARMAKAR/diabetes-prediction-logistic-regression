# Diabetes Prediction using Logistic Regression

**Minor Project** | Udeep Karmakar

Binary classification of diabetes (0 = No diabetes, 1 = Diabetes) on the Pima Indians Diabetes Database using scikit-learn.

## Dataset
- Source: [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 rows, 8 input features, 1 target column (`Outcome`)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Workflow
Load CSV → inspect data → replace invalid zeros with NaN → stratified 80/20 train-test split → median imputation → StandardScaler → balanced Logistic Regression → evaluate → interpret coefficients

All preprocessing and the model are chained in one scikit-learn `Pipeline`, so the test data never leaks into training.

## Results (test set, 154 rows)
| Metric | Result |
|---|---|
| Accuracy | 0.734 |
| Precision | 0.603 |
| Recall | 0.704 |
| F1-score | 0.650 |
| ROC-AUC | 0.813 |

**Confusion matrix:** TN = 75, FP = 25, FN = 16, TP = 38

## Key Findings
- Glucose (1.18) and BMI (0.71) have the strongest positive coefficients
- Recall matters in screening, since false negatives are missed cases
- Class weights are balanced because only about 35% of rows are diabetic

## Limitation
This small historical dataset represents a specific population. The model is for learning classification concepts, not for medical diagnosis.

## Files
- `Udeep_Karmakar_Logistic_Regression_Project.ipynb`: executed notebook with outputs and graphs
- `Udeep_Karmakar_Logistic_Regression_Report.pdf`: one-page project report
- `diabetes.csv`: dataset

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn
