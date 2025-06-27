# Importing Libraries:
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import sklearn

# Display all columns in the dataset:
pd.set_option('display.max_columns', None)

# Reading Dataset (fix the path):
dataset = pd.read_csv("C:\\Users\\Pavan\\OneDrive\\Documents\\Desktop\\Stroke-Prediction-Project\\Stroke-Prediction-Project\\Stroke-Prediction-Project\\Stroke_data.csv")  # If the CSV is in the same directory as Model.py

# Or use the full path:
# dataset = pd.read_csv(r"c:\Users\Pavan\OneDrive\Documents\Desktop\Stroke-Prediction-Project\Stroke-Prediction-Project\Stroke-Prediction-Project\Stroke_data.csv")

# Dropping unnecessary feature:
#1=col,0=roe
dataset = dataset.drop('id', axis=1)

# Filling NaN Values in BMI feature using median:
dataset['bmi'] = dataset['bmi'].fillna(dataset['bmi'].median())

# Dropping rows with 'Other' gender:
Other_gender = dataset[dataset['gender'] == 'Other'].index[0]
dataset = dataset.drop(Other_gender, axis=0)

# Renaming values in work type and smoking status for simplicity:
dataset.replace({'Self-employed': 'Self_employed'}, inplace=True)
dataset.replace({'never smoked': 'never_smoked', 'formerly smoked': 'formerly_smoked'}, inplace=True)

# Splitting into Dependent & Independent Features:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Label Encoding for binary categorical features:
X['ever_married'] = np.where(X['ever_married'] == 'Yes', 1, 0)  # Married = 1, Not Married = 0
X['Residence_type'] = np.where(X['Residence_type'] == 'Rural', 1, 0)  # Rural = 1, Urban = 0

# One-Hot Encoding for categorical features:
X = pd.get_dummies(X, drop_first=True)

# Rearranging the columns for better understanding:
X = X[['gender_Male', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'Residence_type', 'avg_glucose_level', 'bmi',
       'work_type_Never_worked', 'work_type_Private', 'work_type_Self_employed', 'work_type_children',
       'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes']]

# Over Sampling to handle class imbalance:
oversampler = RandomOverSampler(sampling_strategy=0.4)
x_oversampler, y_oversampler = oversampler.fit_resample(X, y)

# Train-Test Split:
X_train, X_test, y_train, y_test = train_test_split(x_oversampler, y_oversampler, test_size=0.2, random_state=0)

# RandomForestClassifier:
RandomForest = RandomForestClassifier(random_state=42)
RandomForest = RandomForest.fit(X_train, y_train)

# Creating a pickle file for the classifier:
filename = 'Stroke.pkl'
try:
    with open(filename, 'wb') as file:
        pickle.dump(RandomForest, file)
    print(f"Model saved successfully as {filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

print(sklearn.__version__)