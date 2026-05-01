import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ----------------------------
# STEP 1: GENERATE DATASET
# ----------------------------
np.random.seed(42)

df = pd.DataFrame({
    'Age': np.random.randint(20, 80, 300),
    'Gender': np.random.choice(['Male', 'Female'], 300),
    'Disease': np.random.choice(['Diabetes', 'Heart Disease', 'Hypertension'], 300),
    'Region': np.random.choice(['Urban', 'Rural'], 300),
    'Length_of_Stay': np.random.randint(1, 15, 300),
    'Treatment_Cost': np.random.randint(5000, 50000, 300),
    'Readmission': np.random.choice([0, 1], 300)
})

print("Sample Data:\n", df.head())

# ----------------------------
# STEP 2: DATA CLEANING
# ----------------------------
df.dropna(inplace=True)

# Encode categorical
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Region'] = df['Region'].map({'Urban': 0, 'Rural': 1})
df = pd.get_dummies(df, columns=['Disease'], drop_first=True)

# ----------------------------
# STEP 3: EDA
# ----------------------------

# Disease Distribution
sns.countplot(x='Disease_Heart Disease', data=df)
plt.title("Disease Distribution")
plt.show()

# Readmission
sns.countplot(x='Readmission', data=df)
plt.title("Readmission Analysis")
plt.show()

# Cost Analysis
sns.boxplot(x='Gender', y='Treatment_Cost', data=df)
plt.title("Cost by Gender")
plt.show()

# Correlation
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# ----------------------------
# STEP 4: PREDICTION
# ----------------------------
X = df[['Age', 'Length_of_Stay', 'Treatment_Cost']]
y = df['Readmission']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Prediction Accuracy:", accuracy)

# ----------------------------
# STEP 5: INSIGHTS
# ----------------------------
print("\nInsights:")
print("- Higher length of stay may increase readmission risk")
print("- Treatment cost varies across patients")
print("- Data helps identify high-risk patients early")