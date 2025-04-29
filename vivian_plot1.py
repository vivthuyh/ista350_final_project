import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Initialize Kaggle API
os.environ['KAGGLE_USERNAME'] = 'vivianmyhuynh'
os.environ['KAGGLE_KEY'] = "5a62aabdedf5fabae671c687e5c481d9"

import kaggle as kg

kg.api.authenticate()
# Dataset you want to download
dataset = 'https://www.kaggle.com/datasets/rishabhm76/pima-diabetes-database'
# Download the dataset
kg.api.dataset_download_files(dataset='rishabhm76/pima-diabetes-database', path='final_project.zip', unzip=True)
df = pd.read_csv('final_project.zip/Pima Indians Diabetes Database.csv', encoding='ISO-8859-1')
print(df.head())

# 1. Scatter Plot: Triceps Skinfold Thickness vs. BMI (after filtering out zeros)
df_filtered = df[df["Triceps skin fold thickness (mm)"] != 0]

# Define x and y
x = df_filtered["Triceps skin fold thickness (mm)"]
y = df_filtered["Body mass index (weight in kg/(height in m)^2)"]

# Add a constant to x for the intercept
X = sm.add_constant(x)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Predict values for regression line
predictions = model.predict(X)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(
    x,
    y,
    s=30,
    color="#246fa3",
    edgecolor="white",       # Add white border
    linewidth=0.5,           # Thin border
    alpha=0.6,               # Slight transparency
    label="Data points"
)
plt.plot(x, predictions, color="red", label="OLS Regression Line")
plt.title("Triceps Skinfold Thickness vs. BMI", fontsize=14)
plt.xlabel("Triceps Skinfold Thickness (mm)", fontsize=12)
plt.ylabel("BMI (kg/m²)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("scatter_thickness_bmi_upgraded.png")
plt.show()