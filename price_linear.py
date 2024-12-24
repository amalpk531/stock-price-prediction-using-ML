# Stock Price Prediction(closing stock price) using linear regression model
# Install required packages: pip install pandas numpy matplotlib scikit-learn yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Set up the ticker symbol and the end date
ticker = 'PTC.NS'  # Change this to your desired stock ticker
end_date = datetime.today().strftime('%Y-%m-%d')

# Download historical stock data
data = yf.download(ticker, start='2010-01-01', end=end_date)
print(data.tail)

# Ensure the data contains a 'Close' column
if 'Close' not in data.columns:
    raise ValueError("The downloaded data does not contain a 'Close' column.")

# Prepare the data
data = data[['Close']]
data['Prev Close'] = data['Close'].shift(1)
data = data.dropna()

# Define features (X) and labels (y)
X = data[['Prev Close']]  # Previous day's close price
y = data['Close']         # Current day's close price

# Train the model using the entire dataset
model = LinearRegression()
model.fit(X, y)

# Predict the next day's close price using the most recent close price
last_known_price = data['Close'].iloc[-1].item()  # Get the last known close price as a scalar
next_day_pred = model.predict(np.array([[last_known_price]]).reshape(1, -1))

# Display results
print(f"The most recent known close price: {last_known_price:.2f}")
print(f"Predicted next day's close price: {next_day_pred[0][0]:.2f}")

# Plot historical data and highlight the prediction
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
plt.axhline(y=next_day_pred[0][0], color='red', linestyle='--', label='Predicted Next Day Close')
plt.title(f'{ticker} Stock Price and Next Day Prediction')
plt.legend()
plt.show()
