# Automated-Trading-Strategy-Optimization-Using-Multi-Modal-Reinforcement-Learning-1
# Learning Objectives
This project will help you:

Understand and implement Linear Regression, Logistic Regression, and K-Nearest Neighbors (KNN)

Fetch and process real-world stock data using the yfinance library

Apply feature engineering and machine learning for stock price prediction and movement classification

# Setup
pip install yfinance scikit-learn pandas matplotlib seaborn


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# Part 1: Downloading Stock Data

Task 1.1: Fetch 1-Year Daily Data
ticker = "AAPL"  # You can also try "TCS.NS", "RELIANCE.NS", "TSLA", etc.
data = yf.download(ticker, period="1y")
data.head()


# Part 2: Linear Regression – Predicting Next Day’s Close Price
Task 2.1: Create Target Variable
data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)
Task 2.2: Train Linear Regression Model
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data["Next_Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
Task 2.3: Plot Actual vs Predicted
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Linear Regression - Next Day Close Price')
plt.show()
 # Part 3: Logistic Regression – Predicting Price Movement
Task 3.1: Define Binary Target
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)
Task 3.2: Train Logistic Regression Model
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#Part 4: K-Nearest Neighbors (KNN) Classification
Task 4.1: Evaluate KNN for Various K Values
python
Copy
Edit
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_knn)
    print(f"K={k}, Accuracy={acc}")
#Bonus Tasks
Add technical indicators (e.g., moving averages, RSI) and observe accuracy change

Plot confusion matrix heatmap using seaborn.heatmap

Apply PCA before running KNN to reduce dimensions

#Compare all three models in a markdown summary
 1. Submission Checklist
 2. Code for Linear Regression with performance plot
 3.  Code for Logistic Regression with confusion matrix
 4.  Code for KNN with varying k values
 5.   Answered bonus questions (optional)

