import joblib
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# LOAD MODEL
# ============================================================

big_move_saved = joblib.load(
    "stock_model.joblib"
)

BIG_MOVE_MODEL = big_move_saved["model"]
BIG_MOVE_FEATURES = big_move_saved["features"]


direction_saved = joblib.load(
    "direction_model.joblib"
)

DIRECTION_MODEL = direction_saved["model"]
DIRECTION_FEATURES = direction_saved["features"]


# ============================================================
# CONFIGURATION
# ============================================================

ticker_data = pd.read_csv("tickers.csv")

TICKERS = (
    ticker_data["symbol"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .unique()
    .tolist()
)

MIN_PRICE = 5
MAX_PRICE = 500
MIN_AVERAGE_VOLUME = 1_000_000
MIN_DAILY_CHANGE = 0
MIN_RELATIVE_VOLUME = 0.5
MAX_RSI = 80

MIN_BIG_MOVE_PROBABILITY = 0.70
MIN_DIRECTION_CONFIDENCE = 0.55

HISTORY_PERIOD = "1y"


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(close_prices, period=14):
    price_changes = close_prices.diff()

    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)

    average_gain = gains.rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_loss = losses.rolling(
        window=period,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    return 100 - (100 / (1 + relative_strength))


def calculate_atr(data, period=14):
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()


def download_market_data():
    market = yf.download(
        "SPY",
        period=HISTORY_PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if market.empty:
        raise RuntimeError("Could not download SPY data.")

    if isinstance(market.columns, pd.MultiIndex):
        market.columns = market.columns.get_level_values(0)

    market = market.copy()

    market["spy_return_5_day"] = (
        market["Close"].pct_change(5) * 100
    )

    market["spy_return_20_day"] = (
        market["Close"].pct_change(20) * 100
    )

    return market[
        [
            "spy_return_5_day",
            "spy_return_20_day",
        ]
    ]


MARKET_DATA = download_market_data()


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stock(symbol):
    try:
        data = yf.download(
            symbol,
            period=HISTORY_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if data.empty or len(data) < 200:
            print(
                f"Skipping {symbol}: "
                "at least 200 trading days are required."
            )
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        if not required_columns.issubset(data.columns):
            print(f"Skipping {symbol}: missing required columns.")
            return None

        data = data.copy()

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        data["daily_change"] = (
            data["Close"].pct_change() * 100
        )

        data["return_5_day_past"] = (
            data["Close"].pct_change(5) * 100
        )

        data["return_20_day_past"] = (
            data["Close"].pct_change(20) * 100
        )

        data["average_volume_5"] = (
            data["Volume"]
            .rolling(window=5)
            .mean()
        )

        data["average_volume_20"] = (
            data["Volume"]
            .rolling(window=20)
            .mean()
        )

        data["relative_volume"] = (
            data["Volume"]
            / data["average_volume_20"]
        )

        data["volume_trend_5_20"] = (
            data["average_volume_5"]
            / data["average_volume_20"]
        )

        data["moving_average_20"] = (
            data["Close"]
            .rolling(window=20)
            .mean()
        )

        data["moving_average_50"] = (
            data["Close"]
            .rolling(window=50)
            .mean()
        )

        data["moving_average_200"] = (
            data["Close"]
            .rolling(window=200)
            .mean()
        )

        data["distance_from_ma_20"] = (
            (
                data["Close"]
                - data["moving_average_20"]
            )
            / data["moving_average_20"]
        ) * 100

        data["distance_from_ma_50"] = (
            (
                data["Close"]
                - data["moving_average_50"]
            )
            / data["moving_average_50"]
        ) * 100

        data["distance_from_ma_200"] = (
            (
                data["Close"]
                - data["moving_average_200"]
            )
            / data["moving_average_200"]
        ) * 100

        data["rsi"] = calculate_rsi(
            data["Close"],
            period=14,
        )

        data["atr_14"] = calculate_atr(
            data,
            period=14,
        )

        data["atr_percent"] = (
            data["atr_14"]
            / data["Close"]
        ) * 100

        data["volatility_20"] = (
            data["Close"]
            .pct_change()
            .rolling(window=20)
            .std()
            * np.sqrt(252)
            * 100
        )

        data = data.join(
            MARKET_DATA,
            how="left",
        )

        data["relative_strength_5_day"] = (
            data["return_5_day_past"]
            - data["spy_return_5_day"]
        )

        data["relative_strength_20_day"] = (
            data["return_20_day_past"]
            - data["spy_return_20_day"]
        )

        latest = data.iloc[-1]

        # ====================================================
        # MODEL INPUT
        # ====================================================

        feature_values = {
            "price": latest["Close"],
            "daily_change": latest["daily_change"],
            "return_5_day_past":
                latest["return_5_day_past"],
            "return_20_day_past":
                latest["return_20_day_past"],
            "relative_volume":
                latest["relative_volume"],
            "volume_trend_5_20":
                latest["volume_trend_5_20"],
            "rsi": latest["rsi"],
            "distance_from_ma_20":
                latest["distance_from_ma_20"],
            "distance_from_ma_50":
                latest["distance_from_ma_50"],
            "distance_from_ma_200":
                latest["distance_from_ma_200"],
            "atr_percent":
                latest["atr_percent"],
            "volatility_20":
                latest["volatility_20"],
            "spy_return_5_day":
                latest["spy_return_5_day"],
            "spy_return_20_day":
                latest["spy_return_20_day"],
            "relative_strength_5_day":
                latest["relative_strength_5_day"],
            "relative_strength_20_day":
                latest["relative_strength_20_day"],
        }

        model_input = pd.DataFrame(
            [feature_values]
        )

        required_model_features = list(
            set(BIG_MOVE_FEATURES)
            | set(DIRECTION_FEATURES)
        )

        missing_features = [
            feature
            for feature in required_model_features
            if feature not in model_input.columns
        ]

        if missing_features:
            print(
                f"Skipping {symbol}: missing features "
                f"{missing_features}"
            )
            return None

        if model_input[
            required_model_features
        ].isna().any().any():
            print(
                f"Skipping {symbol}: "
                "one or more model features are unavailable."
            )
            return None

        # ====================================================
        # MODEL PREDICTIONS
        # ====================================================

        probability_big_move = BIG_MOVE_MODEL.predict_proba(
            model_input[BIG_MOVE_FEATURES]
        )[0][1]

        probability_up_move = UP_MODEL.predict_proba(
            model_input[UP_FEATURES]
        )[0][1]

        probability_down_move = DOWN_MODEL.predict_proba(
            model_input[DOWN_FEATURES]
        )[0][1]

        probability_up_given_big_move = (
            DIRECTION_MODEL.predict_proba(
                model_input[DIRECTION_FEATURES]
            )[0][1]
        )

        probability_down_given_big_move = (
            1.0 - probability_up_given_big_move
        )

        if (
            probability_up_given_big_move
            >= MIN_DIRECTION_CONFIDENCE
        ):
            predicted_direction = "UP"

        elif (
            probability_up_given_big_move
            <= 1.0 - MIN_DIRECTION_CONFIDENCE
        ):
            predicted_direction = "DOWN"

        else:
            predicted_direction = "UNCERTAIN"

        # Approximate joint probabilities.
        probability_big_up_move = (
            probability_big_move
            * probability_up_given_big_move
        )

        probability_big_down_move = (
            probability_big_move
            * probability_down_given_big_move
        )

        # ====================================================
        # CURRENT VALUES AND REASONS
        # ====================================================

        current_price = float(latest["Close"])
        current_volume = int(latest["Volume"])

        average_volume = float(
            latest["average_volume_20"]
        )

        relative_volume = float(
            latest["relative_volume"]
        )

        daily_change = float(
            latest["daily_change"]
        )

        current_rsi = float(
            latest["rsi"]
        )

        moving_average_20 = float(
            latest["moving_average_20"]
        )

        above_20_day_ma = (
            current_price > moving_average_20
        )

        reasons = []

        if probability_big_move >= MIN_BIG_MOVE_PROBABILITY:
            reasons.append(
                "large-move probability "
                f"{probability_big_move * 100:.2f}%"
            )

        if predicted_direction != "UNCERTAIN":
            reasons.append(
                f"predicted direction {predicted_direction}"
            )

        if daily_change >= MIN_DAILY_CHANGE:
            reasons.append("positive daily momentum")

        if relative_volume >= MIN_RELATIVE_VOLUME:
            reasons.append(
                f"relative volume {relative_volume:.2f}"
            )

        if above_20_day_ma:
            reasons.append(
                "above 20-day moving average"
            )

        if current_rsi <= MAX_RSI:
            reasons.append(
                f"RSI {current_rsi:.2f} below {MAX_RSI}"
            )

        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "daily_change": round(daily_change, 2),
            "current_volume": current_volume,
            "average_volume": int(average_volume),
            "relative_volume": round(
                relative_volume,
                2,
            ),
            "rsi": round(current_rsi, 2),
            "moving_average_20": round(
                moving_average_20,
                2,
            ),
            "distance_from_ma_20": round(
                float(latest["distance_from_ma_20"]),
                2,
            ),
            "distance_from_ma_50": round(
                float(latest["distance_from_ma_50"]),
                2,
            ),
            "distance_from_ma_200": round(
                float(latest["distance_from_ma_200"]),
                2,
            ),
            "atr_percent": round(
                float(latest["atr_percent"]),
                2,
            ),
            "volatility_20": round(
                float(latest["volatility_20"]),
                2,
            ),
            "above_20_day_ma": above_20_day_ma,

            # Raw decimal values for filtering
            "big_move_probability":
                probability_big_move,
            "up_probability":
                probability_up_given_big_move,
            "down_probability":
                probability_down_given_big_move,

            # Display percentages
            "probability_big_move_5_days": round(
                probability_big_move * 100,
                2,
            ),
            "probability_up_given_big_move": round(
                probability_up_given_big_move * 100,
                2,
            ),
            "probability_down_given_big_move": round(
                probability_down_given_big_move * 100,
                2,
            ),
            "probability_big_up_move": round(
                probability_big_up_move * 100,
                2,
            ),
            "probability_big_down_move": round(
                probability_big_down_move * 100,
                2,
            ),
            "predicted_direction":
                predicted_direction,
            "meets_big_move_threshold": (
                probability_big_move
                >= MIN_BIG_MOVE_PROBABILITY
            ),
            "reason": ", ".join(reasons),
        }

    except Exception as error:
        print(
            f"Error checking {symbol}: {error}"
        )
        return None


# ============================================================
# FILTERING AND SCORING
# ============================================================

def meets_requirements(stock):
    return (
        MIN_PRICE <= stock["price"] <= MAX_PRICE
        and stock["average_volume"]
            >= MIN_AVERAGE_VOLUME
        and stock["big_move_probability"]
            >= MIN_BIG_MOVE_PROBABILITY
    )


def calculate_score(stock):
    score = 0

    if MIN_PRICE <= stock["price"] <= MAX_PRICE:
        score += 1

    if stock["average_volume"] >= MIN_AVERAGE_VOLUME:
        score += 1

    if stock["daily_change"] >= MIN_DAILY_CHANGE:
        score += 1

    if stock["relative_volume"] >= MIN_RELATIVE_VOLUME:
        score += 1

    if stock["rsi"] <= MAX_RSI:
        score += 1

    if stock["above_20_day_ma"]:
        score += 1

    return score


# ============================================================
# MAIN
# ============================================================

def main():
    analyzed_stocks = []

    for symbol in TICKERS:
        print(f"Checking {symbol}...")

        stock = analyze_stock(symbol)

        if stock:
            stock["score"] = calculate_score(stock)
            stock["qualifies"] = meets_requirements(stock)
            analyzed_stocks.append(stock)

    if not analyzed_stocks:
        print("\nNo stocks could be analyzed.")
        return

    results = pd.DataFrame(analyzed_stocks)

    results = results.sort_values(
        by=[
            "qualifies",
            "probability_big_move_5_days",
            "score",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print("\nRanked scanner results:")
    print(results.to_string(index=False))

    qualifying_results = results[
        results["qualifies"]
    ].copy()

    results.to_csv(
        "results.csv",
        index=False,
    )

    qualifying_results.to_csv(
        "qualifying_results.csv",
        index=False,
    )

    print("\nQUALIFYING LARGE-MOVE CANDIDATES")
    print("=" * 80)

    if qualifying_results.empty:
        print("No stocks met the large-move requirements.")
    else:
        columns = [
            "symbol",
            "price",
            "probability_big_move_5_days",
            "daily_change",
            "relative_volume",
            "atr_percent",
            "volatility_20",
            "score",
        ]

        print(
            qualifying_results[columns]
            .to_string(index=False)
        )

    print("\nFiles saved:")
    print("results.csv")
    print("qualifying_results.csv")


if __name__ == "__main__":
    main()