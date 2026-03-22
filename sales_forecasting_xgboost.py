# 1.-----Import Libraries-------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#2.------Loading dataset---------

df = pd.read_csv("sales.csv", encoding="latin1")

#3.------Data Cleaning-----------

df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')
df = df.groupby('Order Date')['Sales'].sum().reset_index()

#4.------Feature Engineering-----

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['DayName'] = df['Order Date'].dt.day_name()
df['Quarter'] = df['Order Date'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)


dates = df['Order Date']

# -------- Add Lag Features --------

df['Lag_1'] = df['Sales'].shift(1) # Previous day sales #imp
df['Lag_7'] = df['Sales'].shift(7) # Last week same day sales #imp
df['Lag_14'] = df['Sales'].shift(14) #imp
df['Lag_30'] = df['Sales'].shift(30) #imp
df['Rolling_Mean_7'] = df['Sales'].rolling(7).mean() #imp

df = df.dropna() # Remove Nan

# Remove unnecessary columns for model
df_model = df[['Year','Month','Day','DayOfWeek','Quarter',
               'IsWeekend',
               'Lag_1','Lag_7',
               'Rolling_Mean_7',
               'Lag_14','Lag_30',
               'Sales',]]

#5-[Define input features(x) and target variable(y)]
x = df_model[['Year','Month','Day','DayOfWeek','Quarter',
              'IsWeekend',
              'Lag_1','Lag_7',
              'Rolling_Mean_7',
              'Lag_14','Lag_30']]

y = df_model['Sales']

#7. -------- Train-Test Split --------

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# XGBoost model selected after comparing with Linear Regression and Random Forest (best MAE)

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Model train
model.fit(x_train, y_train)

#Model prediction
y_pred = model.predict(x_test)

#Error Calculation
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("XGBoost MAE:", mae)

print("Predicted:", y_pred[:5])
print("Actual:", y_test.values[:5])

#  ------ Visualization -------

plt.figure(figsize=(12,6))

# correct dates for test set
test_dates = dates.iloc[-len(y_test):]

plt.plot(test_dates, y_test.values, label='Actual')
plt.plot(test_dates, y_pred, label='Predicted')
plt.title(f"Actual vs Predicted Sales (MAE: {mae:.2f})")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()