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

# 3. Boxplot: Blood Glucose Distribution by Diabetes Status
plt.figure(figsize=(8,6))
sns.boxplot(
    data=df,
    x="Class variable (0 or 1)",
    y="Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
    palette=["#abebc6","#eb984e"]
)
plt.title("Blood Glucose Distribution by Diabetes Status", fontsize=14)
plt.xlabel("Diabetes Status (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Plasma Glucose Concentration (mg/dL)", fontsize=12)
plt.tight_layout()
plt.savefig("boxplot_glucose_diabetes.png")
plt.show()