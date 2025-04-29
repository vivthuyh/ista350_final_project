'''
ISTA 350 Final Project
Box plot fig 2
'''

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
os.environ['KAGGLE_USERNAME'] = 'caarolyyn'
os.environ['KAGGLE_KEY'] = '66445087575cd15ed79a86921a6f4dcb'
dataset = 'https://www.kaggle.com/datasets/rishabhm76/pima-diabetes-database'  # Example dataset
# Download the dataset
kaggle.api.dataset_download_files(dataset='rishabhm76/pima-diabetes-database', path='final_project.zip', unzip=True)
df = pd.read_csv('final_project.zip/Pima Indians Diabetes Database.csv', encoding ='ISO-8859-1')

df['Diabetes Outcome'] = df['Class variable (0 or 1)'].map({0: 'No', 1: 'Yes'})
df.describe()
df

# data cleaning
weird_data = ['Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
              'Diastolic blood pressure (mm Hg)',
              'Triceps skin fold thickness (mm)',
              '2-Hour serum insulin (mu U/ml)',
              'Body mass index (weight in kg/(height in m)^2)']

# just going to drop these observations
for data in weird_data:
    df[data] = df[data].replace(0, np.nan)
df = df.dropna(subset = weird_data)

# look at final dataset
df.describe()
# dropped half of the data...
# documenting anyways, I think its fine for the context of this class

# categorize BMI
def categorize(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi <= bmi < 24.9:
        return 'Normal'
    else:
        return 'Overweight'
df.loc[:, 'BMI Category'] = df['Body mass index (weight in kg/(height in m)^2)'].apply(categorize)
df

# boxplot: distribution of pregnancies by diabetes outcome 
df['Pregnancy Category'] = ['Pregnancies <= 6' if x <= 6 else 'Pregnancies > 7' for x in df['Number of times pregnant']]

plt.figure(figsize = (8,6))
sns.boxplot(data = df,
            x = 'Pregnancy Category',
            y = 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 
           palette = 'hls')
plt.title('Plasma Glucose Concentration by Pregnancy Category', fontsize = 18)
plt.xlabel('')
plt.ylabel('Plasma Glucose Concentration (mg/dL)', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()