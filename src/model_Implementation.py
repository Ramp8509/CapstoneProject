# Step 4: Model Implementation for Fraud Detection

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)

from imblearn.over_sampling import SMOTE

# Load processed dataset saved from Step 3
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "creditcard_processed.csv"
df = pd.read_csv(data_path)

# Separate features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance with SMOTE (on training data only)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Function to train and evaluate models
def evaluate_model(name, model):
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    print(f"\n{name} Results")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc
    }

# Train and evaluate models
results = []

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
results.append(evaluate_model("Logistic Regression", log_reg))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
results.append(evaluate_model("Decision Tree", dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results.append(evaluate_model("Random Forest", rf))

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
results.append(evaluate_model("XGBoost", xgb))

# Support Vector Machine (SVM) on smaller subset for efficiency
svm = SVC(probability=True, random_state=42)
X_train_smote

# SVM training on a smaller subset (for efficiency)
sample_size = min(30000, len(X_train_smote))
X_train_svm = X_train_smote.sample(sample_size, random_state=42)
y_train_svm = y_train_smote.loc[X_train_svm.index]

svm = SVC(probability=True, random_state=42)
svm.fit(X_train_svm, y_train_svm)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Support Vector Machine",
    "Accuracy": accuracy_score(y_test, y_pred_svm),
    "Precision": precision_score(y_test, y_pred_svm, zero_division=0),
    "Recall": recall_score(y_test, y_pred_svm, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_svm, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob_svm)
})

print("\nSVM Results")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Create a DataFrame to compare all models
results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)

# Print the comparison table
print("\nModel Comparison:")
print(results_df)

# Optional: Plot the Recall for each model
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Recall")
plt.xticks(rotation=20)
plt.title("Model Comparison by Recall")
plt.tight_layout()
plt.show()

# Save models and results
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(log_reg, models_dir / "logistic_regression.pkl")
joblib.dump(dt, models_dir / "decision_tree.pkl")
joblib.dump(rf, models_dir / "random_forest.pkl")
joblib.dump(xgb, models_dir / "xgboost.pkl")
joblib.dump(svm, models_dir / "svm.pkl")

results_df.to_csv(models_dir / "model_comparison.csv", index=False)

print("\nModels and comparison results saved successfully.")