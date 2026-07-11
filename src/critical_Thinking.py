import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

models_path = project_root / "models"

best_model = joblib.load(models_path / "random_forest.pkl") #Load the Best Model

import pandas as pd

df = pd.read_csv(project_root / "data" / "creditcard_processed.csv")

X = df.drop("Class", axis=1)

import pandas as pd

df = pd.read_csv(project_root / "data" / "creditcard_processed.csv") #Load the Processed Dataset

X = df.drop("Class", axis=1)

X_sample = X.sample(1000, random_state=42) #Sample the Data

explainer = shap.TreeExplainer(best_model) #Generate SHAP Values

shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample) #SHAP Summary Plot

#The SHAP summary plot highlights the features that contribute most to the model's fraud predictions.
#Features with larger SHAP values have a greater influence on the model's decision, improving transparency and helping stakeholders understand why transactions are classified as fraudulent.