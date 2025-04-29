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

# Create blood pressure categories
def categorize_bp(bp):
    if bp < 70:
        return "Low"
    elif bp <= 90:
        return "Normal"
    else:
        return "High"

df["Blood Pressure Category"] = df["Diastolic blood pressure (mm Hg)"].apply(categorize_bp)

# 2. Pie Charts: Diabetes status distribution within each Blood Pressure Category
# Prepare data
bp_diabetes = df.groupby(["Blood Pressure Category", "Class variable (0 or 1)"]).size().unstack(fill_value=0)

# Set up the figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))

# Blood Pressure Categories with ranges
categories = ["Low (<70)", "Normal (70-90)", "High (>90)"]
original_categories = ["Low", "Normal", "High"]
axes = [ax1, ax2, ax3]

# Plot each pie chart
for i in range(3):
    category = original_categories[i]
    if category in bp_diabetes.index:
        values = bp_diabetes.loc[category]
        wedges, texts, autotexts = axes[i].pie(
            values,
            startangle=90,
            shadow=True,  # Adds shadow for 3D look
             colors=["#9FE2BF","#F08080"],
            wedgeprops={'edgecolor': 'black'},
            autopct='%1.1f%%',  # add percentage inside slices
            pctdistance=0.7  # move percentage text slightly inward
        )
        axes[i].set_title(f"{categories[i]} Blood Pressure", fontsize=14)

fig.legend(
    wedges,
    ["No Diabetes", "Diabetes"],
    title="Status",
    loc="lower center",
    ncol=2,
    fontsize=12
)

plt.suptitle("Diabetes Status Distribution by Blood Pressure Category", fontsize=16)
plt.tight_layout()
plt.savefig("piecharts_bmi_bp.png")
plt.show()