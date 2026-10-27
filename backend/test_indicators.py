import yfinance as yf
from stock_analysis import calculate_indicators

# Download approximately one year of historical stock data
stock = yf.Ticker("RELIANCE.NS")
data = stock.history(period="1y")

if data.empty:
    print("Unable to retrieve stock data. Check your internet connection.")
else:
    # Calculate technical indicators
    result = calculate_indicators(data)

    # Display the latest indicator values
    latest = result.iloc[-1]

    print("\n--- Stock Technical Analysis ---")
    print(f"Latest Closing Price: {latest['Close']:.2f}")
    print(f"SMA (20 days): {latest['SMA_20']:.2f}")
    print(f"SMA (50 days): {latest['SMA_50']:.2f}")
    print(f"EMA (20 days): {latest['EMA_20']:.2f}")
    print(f"Daily Return: {latest['Daily_Return']:.2f}%")
    print(f"RSI (14 days): {latest['RSI_14']:.2f}")