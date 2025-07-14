import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# Load cleaned data
from data_cleaning import load_and_clean_data

# Load and prepare data
df = load_and_clean_data('C:/Users/Kayla/Desktop/heart-disease-risk-prediction/data/heart_disease_data.csv')
print("Columns:", df.columns.tolist())

# Define features and target
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"ROC AUC:   {roc_auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('C:/Users/Kayla/Desktop/heart-disease-risk-prediction/output/confusion_matrix.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('C:/Users/Kayla/Desktop/heart-disease-risk-prediction/output/roc_curve.png')
plt.close()

# Feature importance
importances = model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('C:/Users/Kayla/Desktop/heart-disease-risk-prediction/output/feature_importance.png')
plt.close()

# Save model
joblib.dump(model, 'C:/Users/Kayla/Desktop/heart-disease-risk-prediction/model/model.pkl')
print("Model saved to model/model.pkl")
