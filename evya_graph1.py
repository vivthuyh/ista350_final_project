import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


# Initialize Kaggle API
os.environ['KAGGLE_USERNAME'] = 'Evyasivarajan'
os.environ['KAGGLE_KEY'] = "506c9760672886c20e61c242a1bb3c08"

import kaggle as kg

kg.api.authenticate()

# Dataset you want to download (replace with your dataset's path on Kaggle)
dataset = 'https://www.kaggle.com/datasets/rishabhm76/pima-diabetes-database'  # Example dataset

# Download the dataset
kg.api.dataset_download_files(dataset='rishabhm76/pima-diabetes-database', path='final_project.zip', unzip=True)

df = pd.read_csv('final_project.zip/Pima Indians Diabetes Database.csv', encoding ='ISO-8859-1')
print(df.head())

#Figure 1: Scatter Plot (General BMI vs Glucose in Pima County)
df.rename(columns={
    'Number of times pregnant': 'Pregnancies',
    'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': 'Glucose',
    'Diastolic blood pressure (mm Hg)': 'BloodPressure',
    'Triceps skin fold thickness (mm)': 'SkinThickness',
    '2-Hour serum insulin (mu U/ml)': 'Insulin',
    'Body mass index (weight in kg/(height in m)^2)': 'BMI',
    'Diabetes pedigree function': 'DiabetesPedigree',
    'Age (years)': 'Age',
    'Class variable (0 or 1)': 'Outcome'
}, inplace=True)
plt.figure(figsize=(8, 6))
sns.regplot(
    x='BMI',
    y='Glucose',
    data=df,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.xlim(15, 50)   
plt.ylim(50, 200)  
plt.xlabel('Body Mass Index (kg/m²)')
plt.ylabel('Glucose (mg/dL)')
plt.title('Body Mass Index vs. Glucose')
plt.grid(True)
plt.tight_layout()
plt.show()
