
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Download Indian stock market data
ticker = "RELIANCE.NS"

print("Downloading stock market data...")

data = yf.download(
    ticker,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise ValueError("No stock data found. Check ticker or internet connection.")

# Handle Yahoo Finance column format
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. Create machine learning features
df = data[["Close", "Volume"]].copy()

df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Return"].rolling(10).std()
df["Volume_Change"] = df["Volume"].pct_change()

# Target: next trading day's closing price
df["Target"] = df["Close"].shift(-1)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

features = [
    "Close",
    "Volume",
    "Return",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility",
    "Volume_Change"
]

X = df[features]
y = df["Target"]

# 3. Split data chronologically (no random shuffle)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 4. Train Machine Learning model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# 5. Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n----- MODEL PERFORMANCE -----")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 6. Predict next trading day's closing price
latest_features = X.iloc[[-1]]

next_price = model.predict(latest_features)[0]
last_price = df["Close"].iloc[-1]

change = ((next_price - last_price) / last_price) * 100

print("\n----- STOCK PREDICTION -----")
print("Stock:", ticker)
print("Latest available close:", round(last_price, 2))
print("Predicted next close:", round(next_price, 2))
print("Predicted change:", round(change, 2), "%")

if change > 0:
    print("Model direction: UP")
elif change < 0:
    print("Model direction: DOWN")
else:
    print("Model direction: FLAT")

# 7. Plot actual vs predicted prices
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Price"
)

plt.plot(
    y_test.index,
    predictions,
    label="Predicted Price"
)

plt.title("Indian Stock Market Prediction - " + ticker)
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("stock_prediction.png")
plt.show()
