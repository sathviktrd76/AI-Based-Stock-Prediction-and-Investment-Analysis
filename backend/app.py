from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import math

from ml_model import train_and_evaluate
from stock_analysis import calculate_indicators

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "message": "AI Stock Prediction API is running!"
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/api/stock/<symbol>")
def get_stock_data(symbol):
    """Return historical stock data for the selected period."""

    period = request.args.get("period", "1mo")
    allowed_periods = ["1mo", "3mo", "6mo", "1y"]

    if period not in allowed_periods:
        return jsonify({
            "error": "Invalid period. Use 1mo, 3mo, 6mo, or 1y."
        }), 400

    try:
        data = yf.Ticker(symbol).history(
            period=period,
            auto_adjust=True
        )

        if data.empty:
            return jsonify({
                "error": f"No stock data found for {symbol}."
            }), 404

        data = data.reset_index()

        # Remove rows with missing OHLC prices.
        price_columns = ["Open", "High", "Low", "Close"]
        data = data.dropna(subset=price_columns)

        result = []

        for _, row in data.iterrows():
            # Skip rows containing non-finite price values.
            prices = [
                float(row[column])
                for column in price_columns
            ]

            if not all(math.isfinite(price) for price in prices):
                continue

            # Handle missing or invalid volume safely.
            volume_value = row["Volume"]

            if (
                volume_value is None
                or not math.isfinite(float(volume_value))
                or float(volume_value) < 0
            ):
                volume = 0
            else:
                volume = int(volume_value)

            result.append({
                "date": str(row["Date"].date()),
                "open": prices[0],
                "high": prices[1],
                "low": prices[2],
                "close": prices[3],
                "volume": volume
            })

        if not result:
            return jsonify({
                "error": f"No valid stock prices found for {symbol}."
            }), 404

        return jsonify({
            "symbol": symbol.upper(),
            "period": period,
            "data": result
        })

    except Exception as error:
        app.logger.exception("Unable to retrieve stock data")

        return jsonify({
            "error": f"Unable to retrieve stock data: {str(error)}"
        }), 500


@app.route("/api/indicators/<symbol>")
def get_stock_indicators(symbol):
    """Return the latest technical indicators for a stock."""

    try:
        data = yf.Ticker(symbol).history(
            period="1y",
            auto_adjust=True
        )

        if data.empty:
            return jsonify({
                "error": f"No stock data found for {symbol}."
            }), 404

        indicators = calculate_indicators(data)

        if indicators.empty:
            return jsonify({
                "error": f"Unable to calculate indicators for {symbol}."
            }), 422

        latest = indicators.iloc[-1]

        def safe_number(value):
            if value is None:
                return None

            number = float(value)

            if not math.isfinite(number):
                return None

            return round(number, 2)

        return jsonify({
            "symbol": symbol.upper(),
            "date": str(indicators.index[-1].date()),
            "close": safe_number(latest["Close"]),
            "sma_20": safe_number(latest["SMA_20"]),
            "sma_50": safe_number(latest["SMA_50"]),
            "ema_20": safe_number(latest["EMA_20"]),
            "rsi_14": safe_number(latest["RSI_14"]),
            "daily_return": safe_number(latest["Daily_Return"])
        })

    except Exception as error:
        app.logger.exception("Unable to calculate stock indicators")

        return jsonify({
            "error": f"Unable to calculate indicators: {str(error)}"
        }), 500


@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    """Train and evaluate the models and predict the latest trend."""

    try:
        # Fetch five years of daily historical data for ML training.
        data = yf.Ticker(symbol).history(
            period="5y",
            interval="1d",
            auto_adjust=True
        )

        if data.empty:
            return jsonify({
                "error": f"No stock data found for {symbol}."
            }), 404

        result = train_and_evaluate(data)

        return jsonify({
            "symbol": symbol.upper(),

            # Primary model prediction
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model": result["model"],

            # Random Forest feature importance
            "feature_importance": result["feature_importance"],

            # Evaluation metrics for both models
            "evaluation": result["evaluation"],

            # Training and testing information
            "training_samples": result["training_samples"],
            "testing_samples": result["testing_samples"],

            # Compare Random Forest and KNN predictions
            "different_test_predictions": result[
                "different_test_predictions"
            ],
            "total_test_predictions": result[
                "total_test_predictions"
            ],

            # Date associated with the reported prediction
            "last_data_date": result["last_data_date"],

            "note": (
                "This is an educational model output, "
                "not a guarantee of future stock performance."
            )
        })

    except Exception as error:
        app.logger.exception("Unable to generate stock prediction")

        return jsonify({
            "error": f"Unable to generate prediction: {str(error)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)