import pandas as pd


def calculate_indicators(data):
    """Calculate technical indicators from historical stock data."""

    df = data.copy()

    # Ensure historical observations are in chronological order.
    df = df.sort_index()

# Remove duplicate dates if present.
    df = df.loc[~df.index.duplicated(keep="last")]

    # Ensure closing prices are numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # 1. Simple Moving Averages
    # Average closing price over the last 20 and 50 trading days
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # 2. Exponential Moving Average
    # Gives greater weight to more recent prices
    df["EMA_20"] = df["Close"].ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    # 3. Daily Percentage Return
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # 4. Relative Strength Index (RSI)
    change = df["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()

    # Handle periods with no losses safely
    relative_strength = average_gain.div(
        average_loss.where(average_loss != 0)
    )

    df["RSI_14"] = 100 - (100 / (1 + relative_strength))

    # RSI is 100 when gains occur without losses
    df.loc[
        (average_loss == 0) & (average_gain > 0),
        "RSI_14"
    ] = 100

    # RSI is 50 when both gains and losses are zero
    df.loc[
        (average_loss == 0) & (average_gain == 0),
        "RSI_14"
    ] = 50

    return df