import time
import yfinance as yf
from ml_model import train_and_evaluate

# Download one year of historical stock data
stock = yf.Ticker("RELIANCE.NS")
data = stock.history(period="1y")

if data.empty:
    print("Unable to retrieve stock data. Check your internet connection.")
else:
    try:
        start_time = time.time()

        result = train_and_evaluate(data)

        print(f"\nModel training and prediction took {time.time() - start_time:.2f} seconds.") 

        print("\n--- AI Stock Prediction Results ---")
        print(f"Stock: RELIANCE.NS")
        print(f"Predicted Trend: {result['prediction']}")
        print(f"Model Confidence: {result['confidence']:.2%}")
        print(f"Primary Model: {result['model']}")

        print("\n--- Model Evaluation ---")

        for model_name, metrics in result["evaluation"].items():
            print(f"\n{model_name}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision_macro']:.4f}")
            print(f"Recall:    {metrics['recall_macro']:.4f}")
            print(f"F1 Score:  {metrics['f1_macro']:.4f}")

        print(f"\nTraining samples: {result['training_samples']}")
        print(f"Testing samples: {result['testing_samples']}")

    except Exception as error:
        print(f"\nModel test failed: {error}")