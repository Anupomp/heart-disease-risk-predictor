# Evaluation Report: Heart Disease Risk Prediction

## Models Compared
- Logistic Regression
- Random Forest
- Gradient Boosting

## Key Metrics (Test Set)

| Model             | Accuracy | Precision | Recall | ROC-AUC |
|------------------|----------|-----------|--------|---------|
| Logistic Regression | 84.6%    | 0.83      | 0.86   | 0.89    |
| Random Forest     | 87.4%    | 0.86      | 0.88   | 0.91    |
| Gradient Boosting | 88.6%    | 0.89      | 0.87   | 0.93    |

## Observations
- Gradient Boosting had the best overall performance.
- Chest pain type and cholesterol were top predictors.
- ROC-AUC shows all models handled class imbalance well.

## Screenshots:
- Confusion Matrix: `screenshots/confusion_matrix_rf.png`
- ROC Curve: `screenshots/roc_curve.png`

