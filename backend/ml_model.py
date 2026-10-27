import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Features used by both machine learning models
FEATURE_COLUMNS = [
    "Daily_Return",
    "SMA_20",
    "EMA_20",
    "RSI_14",
    "Volume",
    "Price_SMA_Ratio",
    "Price_EMA_Ratio",
    "Momentum_5D",
    "Volume_Change",
    "Volatility_20D",
]


def prepare_features(data):
    """
    Calculate technical indicators, additional features,
    and future-return target labels.

    UP:       Five-observation future return >= +2%.
    DOWN:     Five-observation future return <= -2%.
    NEUTRAL:  Five-observation future return between -2% and +2%.

    All input features use current or historical data only.
    Recent rows without known future returns are retained
    for the latest prediction.
    """

    df = data.copy()

    # Ensure observations are in chronological order.
    df = df.sort_index()

    # Remove duplicate dates if present.
    df = df.loc[~df.index.duplicated(keep="last")]

    # Ensure numeric prices and volume.
    for column in ["Close", "Volume"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Replace invalid closing prices with missing values.
    df.loc[df["Close"] <= 0, "Close"] = float("nan")

    # --------------------------------------------------
    # 1. Existing technical indicators
    # --------------------------------------------------

    # Daily percentage return.
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Simple moving average.
    df["SMA_20"] = df["Close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    # Exponential moving average.
    df["EMA_20"] = df["Close"].ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    # RSI calculation.
    change = df["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()

    relative_strength = average_gain.div(
        average_loss.where(average_loss != 0)
    )

    df["RSI_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # RSI = 100 when gains occur without losses.
    df.loc[
        (average_loss == 0) & (average_gain > 0),
        "RSI_14",
    ] = 100

    # RSI = 50 when both gains and losses are zero.
    df.loc[
        (average_loss == 0) & (average_gain == 0),
        "RSI_14",
    ] = 50

    # --------------------------------------------------
    # 2. Additional predictive features
    # --------------------------------------------------

    # Distance of price from its 20-day moving average.
    df["Price_SMA_Ratio"] = (
        df["Close"] / df["SMA_20"] - 1
    ) * 100

    # Distance of price from its 20-day exponential average.
    df["Price_EMA_Ratio"] = (
        df["Close"] / df["EMA_20"] - 1
    ) * 100

    # Five-observation historical momentum.
    df["Momentum_5D"] = (
        df["Close"].pct_change(periods=5) * 100
    )

    # Percentage change in volume.
    previous_volume = df["Volume"].shift(1)

    df["Volume_Change"] = (
        (df["Volume"] - previous_volume)
        / previous_volume.where(previous_volume > 0)
    ) * 100

    # Rolling standard deviation of daily returns.
    df["Volatility_20D"] = (
        df["Daily_Return"].rolling(window=20).std()
    )

    # --------------------------------------------------
    # 3. Create future-return target labels
    # --------------------------------------------------

    df["Future_Return_5D"] = (
        df["Close"].shift(-5) / df["Close"] - 1
    ) * 100

    # Unknown future returns remain missing.
    df["Target"] = pd.Series(
        index=df.index,
        dtype="object",
    )

    df.loc[
        df["Future_Return_5D"] >= 2,
        "Target",
    ] = "UP"

    df.loc[
        df["Future_Return_5D"] <= -2,
        "Target",
    ] = "DOWN"

    df.loc[
        df["Future_Return_5D"].between(
            -2,
            2,
            inclusive="neither",
        ),
        "Target",
    ] = "NEUTRAL"

    # Replace infinite values so they cannot enter either model.
    df = df.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    return df, FEATURE_COLUMNS


def calculate_metrics(actual, predicted):
    """Calculate classification evaluation metrics."""

    return {
        "accuracy": round(
            accuracy_score(actual, predicted),
            4,
        ),
        "precision_macro": round(
            precision_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "recall_macro": round(
            recall_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "f1_macro": round(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    }


def train_and_evaluate(data):
    """
    Train and evaluate Random Forest and KNN chronologically.
    Use the latest available features for the current prediction.
    """

    df, feature_columns = prepare_features(data)

    # Only rows with known targets and complete features
    # may be used for training or testing.
    labeled_df = df.dropna(
        subset=feature_columns + ["Future_Return_5D", "Target"]
    ).copy()

    if len(labeled_df) < 100:
        raise ValueError(
            "Not enough historical data. "
            "Please use a longer stock history."
        )

    X = labeled_df[feature_columns]
    y = labeled_df["Target"]

    # Chronological split: reserve the latest 20% for testing.
    split_index = int(len(labeled_df) * 0.8)

    # Gap prevents overlap between five-observation target windows.
    train_end = split_index - 5

    if train_end <= 0 or split_index >= len(labeled_df):
        raise ValueError(
            "Not enough data for a chronological train/test split."
        )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------
    # Baseline: Always predict the most common training class
    # --------------------------------------------------

    majority_class = y_train.mode().iloc[0]

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    print("\n========== MODEL DIAGNOSTICS ==========")

    print("\nTraining class distribution:")
    print(
        y_train.value_counts()
        .reindex(["DOWN", "NEUTRAL", "UP"], fill_value=0)
    )

    print("\nTesting class distribution:")
    print(
        y_test.value_counts()
        .reindex(["DOWN", "NEUTRAL", "UP"], fill_value=0)
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features:", feature_columns)

    print("\nMost common training class:", majority_class)
    print("Baseline evaluation:", baseline_metrics)

    print("=======================================\n")

    # --------------------------------------------------
    # Model 1: Random Forest
    # --------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
    )

    random_forest.fit(X_train, y_train)

    rf_test_predictions = random_forest.predict(X_test)

    # --------------------------------------------------
    # Model 2: K-Nearest Neighbors
    # --------------------------------------------------

    knn = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=7),
        ),
    ])

    knn.fit(X_train, y_train)

    knn_test_predictions = knn.predict(X_test)

    # --------------------------------------------------
    # Prediction diagnostics
    # --------------------------------------------------

    print("\nRandom Forest predicted class distribution:")
    print(
        pd.Series(rf_test_predictions)
        .value_counts()
        .reindex(["DOWN", "NEUTRAL", "UP"], fill_value=0)
    )

    print("\nKNN predicted class distribution:")
    print(
        pd.Series(knn_test_predictions)
        .value_counts()
        .reindex(["DOWN", "NEUTRAL", "UP"], fill_value=0)
    )

    # --------------------------------------------------
    # Confusion matrix: Random Forest
    # --------------------------------------------------

    labels = ["DOWN", "NEUTRAL", "UP"]

    rf_matrix = confusion_matrix(
        y_test,
        rf_test_predictions,
        labels=labels,
    )

    print("\n========== RANDOM FOREST CONFUSION MATRIX ==========")
    print("Class order:", labels)
    print("Rows = actual classes; columns = predicted classes")
    print(rf_matrix)
    print("=====================================================\n")

    # --------------------------------------------------
    # Confusion matrix: KNN
    # --------------------------------------------------

    knn_matrix = confusion_matrix(
        y_test,
        knn_test_predictions,
        labels=labels,
    )

    print("\n========== KNN CONFUSION MATRIX ==========")
    print("Class order:", labels)
    print("Rows = actual classes; columns = predicted classes")
    print(knn_matrix)
    print("==========================================\n")

    # --------------------------------------------------
    # Evaluate all three approaches
    # --------------------------------------------------

    evaluation = {
        "Baseline": baseline_metrics,
        "Random Forest": calculate_metrics(
            y_test,
            rf_test_predictions,
        ),
        "KNN": calculate_metrics(
            y_test,
            knn_test_predictions,
        ),
    }

    print("\n========== MODEL COMPARISON ==========")

    for model_name, metrics in evaluation.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")

    print("======================================\n")

    different_predictions = int(
        (rf_test_predictions != knn_test_predictions).sum()
    )

    # --------------------------------------------------
    # Train the final Random Forest using all labeled rows
    # --------------------------------------------------

    final_random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
    )

    final_random_forest.fit(X, y)

    # Find the most recent row with complete input features.
    latest_features = df.dropna(
        subset=feature_columns
    )

    if latest_features.empty:
        raise ValueError(
            "Unable to calculate indicators for the latest stock data."
        )

    latest_row = latest_features.iloc[[-1]]
    latest_X = latest_row[feature_columns]

    # Generate the latest prediction.
    prediction = final_random_forest.predict(latest_X)[0]

    # Calculate feature importance.
    feature_importances = dict(
        zip(
            feature_columns,
            final_random_forest.feature_importances_.tolist(),
        )
    )

    feature_importances = dict(
        sorted(
            feature_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Probability assigned to the predicted class.
    probabilities = final_random_forest.predict_proba(latest_X)[0]

    predicted_class_index = list(
        final_random_forest.classes_
    ).index(prediction)

    confidence = float(probabilities[predicted_class_index])

    # Identify the date of the latest feature row.
    latest_index = latest_features.index[-1]

    if isinstance(latest_index, pd.Timestamp):
        last_data_date = latest_index.strftime("%Y-%m-%d")
    else:
        last_data_date = str(latest_index)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "model": "Random Forest",
        "feature_importance": feature_importances,
        "evaluation": evaluation,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "different_test_predictions": different_predictions,
        "total_test_predictions": int(len(X_test)),
        "last_data_date": last_data_date,
    }