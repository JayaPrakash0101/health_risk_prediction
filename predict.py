import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('Medical_insurance.csv')

df['sex'] = df['sex'].astype(int)
df['smoker'] = df['smoker'].astype(int)

X = df.drop(columns = ['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
df = pd.merge(x_test, y_test, how = 'inner', left_index = True, right_index = True)

xgb = XGBRegressor()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

min_claim = y_test.min()
max_claim = y_test.max()

def calculate_risk_score(predicted_claim, age, smoker, bmi):
    min_usd = df_test['Predicted Charges (USD)'].min()
    max_usd = df_test['Predicted Charges (USD)'].max()

    base_risk_score = ((predicted_claim - min_usd) / (max_usd - min_usd)) * 100

    smoker_penalty = 1.3 if smoker == 1 else 1.0
    high_bmi_penalty = 1.2 if bmi > 30 else 1.0
    high_age_penalty = 1.2 if age > 50 else 1.0

    final_risk_score = base_risk_score * smoker_penalty * high_bmi_penalty * high_age_penalty

    return round(min(max(final_risk_score, 0), 100), 2)

def convert_usd_to_inr(predicted_usd, age, smoker, bmi):
    min_premium_inr = 2000
    max_premium_inr = 10000

    min_usd = df_test['Predicted Charges (USD)'].min()
    max_usd = df_test['Predicted Charges (USD)'].max()
    scaled_premium = min_premium_inr + ((predicted_usd - min_usd) / (max_usd - min_usd)) * (max_premium_inr - min_premium_inr)
    
    smoker_penalty = 1.3 if smoker == 1 else 1.0
    high_bmi_penalty = 1.2 if bmi > 30 else 1.0
    high_age_penalty = 1.2 if age > 50 else 1.0
    final_premium = scaled_premium * smoker_penalty * high_bmi_penalty * high_age_penalty
    
    return round(min(max(final_premium, min_premium_inr), max_premium_inr), 2)

#Copy test data
df_test = x_test.copy()
df_test['Predicted Charges (USD)'] = y_pred

#Conversoin of USD to INR
df_test['Predicted Premium (INR)'] = df_test.apply(lambda row: convert_usd_to_inr(row['Predicted Charges (USD)'], row['age'], row['smoker'], row['bmi']), axis=1)
df_test['Predicted Risk Score'] = df_test.apply(lambda row: calculate_risk_score(row['Predicted Charges (USD)'], row['age'], row['smoker'], row['bmi']), axis=1)

print("\nPredictions on test data:\n")

print(df_test.head())

print("\nR2 score for charges prediction: {:.3f}".format(r2_score(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error for charges: ${:.2f}'.format(mae))

#Pickle the model (only use this if you want to save the model)
# pickle.dump(xgb, open("ml_model.pkl", "wb"))  
# model = pickle.load(open("ml_model.pkl", 'rb'))
# print("\nModel saved successfully!")

# print(df_test['Predicted Charges (USD)'].min())
# print(df_test['Predicted Charges (USD)'].max())