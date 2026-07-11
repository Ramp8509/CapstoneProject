from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import RandomForestClassifier

import shap

# Load dataset
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "creditcard.csv"
if not data_path.exists():
    data_path = Path.cwd() / "data" / "creditcard.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Could not find dataset at: {data_path}")
df = pd.read_csv(data_path)
df.head()

#Dataset Information
df.info()

#Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

#Check Duplicate Records
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

#Remove duplicates
df = df.drop_duplicates()

#Check Class Imbalance
df["Class"].value_counts()
sns.countplot(x="Class", data=df)
plt.title("Distribution of Legitimate vs Fraud Transactions")
plt.show()
#The dataset is highly imbalanced, with fraudulent transactions representing less than 1% of all observations. 
#This imbalance will be addressed later using SMOTE during model training.

#Outlier Detection
plt.figure(figsize=(8,4))
sns.boxplot(x=df["Amount"])
plt.title("Transaction Amount Distribution")
plt.show()
#The Amount feature contains several extreme values. However, these observations may represent genuine high-value transactions or fraud 
# and should not be removed without careful consideration.

#Scale Numerical Features 
scaler = StandardScaler()
df["Scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
df["Scaled_Time"] = scaler.fit_transform(df[["Time"]])

#Standardization ensures numerical variables have similar scales, improving the performance of algorithms such as Logistic Regression and Support Vector Machines.
df.drop(["Amount","Time"], axis=1, inplace=True)

#Exploratory Data Analysis (EDA)
plt.figure(figsize=(8,5))
sns.histplot(df["Scaled_Amount"], bins=50)
plt.title("Distribution of Transaction Amount")
plt.show()
#Most transactions involve relatively small amounts, while a small number of high-value transactions create a right-skewed distribution.

#Correlation Heatmap

plt.figure(figsize=(16,12))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlaion Matrix")
plt.show()
#Most PCA-transformed variables exhibit weak correlations, indicating successful dimensionality reduction and low multicollinearity.

#Fraud vs Amount
sns.boxplot(x="Class", y="Scaled_Amount", data=df)
plt.show()
#Observation: Fraudulent transactions occur across a wide range of transaction amounts, indicating that amount alone is insufficient for distinguishing fraud.

#Feature Engineering

df["Amount_Category"] = pd.qcut(
    df["Scaled_Amount"],
    q=4,
    labels=["Low","Medium","High","Very High"]
)

df["Amount_Category"] = df["Amount_Category"].astype("category").cat.codes
#Binning transaction amounts into quartiles may help capture non-linear relationships between transaction value and fraud risk.

#Feature Importance -Random Forest Importance


X = df.drop("Class", axis=1)

y = df["Class"]

rf = RandomForestClassifier(random_state=42)

rf.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()
#The Random Forest model identifies the most influential variables contributing to fraud prediction. These features will guide feature selection and improve model interpretability.

#Features with larger absolute SHAP values have a greater influence on identifying fraudulent transactions.

#Feature Selection - SelectKBest (Filter Method)

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=15
)

X_new = selector.fit_transform(X, y)

selected_features = X.columns[
    selector.get_support()
]

selected_features

#Mutual Information evaluates the dependency between each feature and the target variable. 
#The top 15 features are retained to reduce noise and computational complexity while preserving predictive power.


#Dimensionality Reduction

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    alpha=0.5,
    cmap="coolwarm"
)

plt.title("PCA Projection")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()

#PCA was applied to reduce the feature space to two principal components for visualization. While the original dataset already contains 
#PCA-transformed variables, this additional projection provides an intuitive view of the overall data structure and potential class separation.

# Save processed dataset for model implementation

processed_data_path = project_root / "data" / "creditcard_processed.csv"

df.to_csv(processed_data_path, index=False)

print(f"Processed dataset saved to: {processed_data_path}")