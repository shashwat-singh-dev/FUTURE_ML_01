# 📊 Sales Forecasting using XGBoost

## 🚀 Project Overview
This project focuses on **time series sales forecasting** using machine learning.  
The goal was to predict future sales based on historical patterns using feature engineering and advanced models.

---

## 🧠 Approach

### 1. Data Preprocessing
- Converted `Order Date` to datetime format  
- Sorted data chronologically  
- Aggregated daily sales  

---

### 2. Feature Engineering
Created meaningful time-series features:

- 📅 Date Features:
  - Year, Month, Day
  - DayOfWeek
  - Quarter
  - IsWeekend

- 🔁 Lag Features:
  - Lag_1 (previous day)
  - Lag_7 (last week same day)
  - Lag_14
  - Lag_30

- 📈 Rolling Features:
  - Rolling Mean (7 days)

---

### 3. Models Tried
| Model            | Result (MAE) |
|------------------|-------------|
| Linear Regression | >700        |
| Random Forest     | >700        |
| XGBoost           | ✅ **~686** |

👉 XGBoost performed best due to its ability to capture complex patterns.

---

## 🏆 Final Model
- **Model:** XGBoost Regressor  
- **MAE Score:** ~686  
- Optimized using:
  - n_estimators = 500  
  - max_depth = 6  
  - learning_rate = 0.05  

---

## 📊 Visualization
Actual vs Predicted sales comparison:

*(Add your plot image here)*

---

## 📁 Files Included
- `sales_forecasting_xgboost.py` → Main model code  
- `sales.csv` → Dataset  
- `plot.png` → Visualization (optional)

---

## ⚙️ How to Run

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
python sales_forecasting_xgboost.py
```
## 👨‍💻 Author
**Shashwat Singh**  
B.Tech (3rd Year) | Data Analyst | Machine Learning Enthusiast  

---

## 🔗 Connect with Me
- LinkedIn: *((https://www.linkedin.com/in/shashwat-singh-aa83022a1/))*  
