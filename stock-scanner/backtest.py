import pandas as pd
import yfinance as yf


# Load symbols from tickers.csv
ticker_data = pd.read_csv("tickers.csv")
TICKERS = ticker_data["symbol"].dropna().str.upper().tolist()


# Same settings used by the scanner
MIN_PRICE = 5
MAX_PRICE = 500
MIN_AVERAGE_VOLUME = 1_000_000
MIN_DAILY_CHANGE = 0
MIN_RELATIVE_VOLUME = 0.5
MAX_RSI = 80

# Minimum score required to create a signal
MIN_SIGNAL_SCORE = 6


def remove_overlapping_signals(signals, cooldown_days=5):
    selected_rows = []
    next_allowed_index = -1

    for index, row in signals.iterrows():
        if index >= next_allowed_index:
            selected_rows.append(row)
            next_allowed_index = index + cooldown_days

    if not selected_rows:
        return pd.DataFrame(columns=signals.columns)

    return pd.DataFrame(selected_rows)

def calculate_rsi(close_prices, period=14):
    price_changes = close_prices.diff()

    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)

    average_gain = gains.rolling(window=period).mean()
    average_loss = losses.rolling(window=period).mean()

    relative_strength = (
        average_gain / average_loss.replace(0, float("nan"))
    )

    return 100 - (100 / (1 + relative_strength))


def prepare_stock_data(symbol):
    """
    Download historical data and calculate all scanner measurements
    for every trading day.
    """
    try:
        data = yf.download(
            symbol,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            print(f"No data returned for {symbol}")
            return None

        # yfinance can sometimes return multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.copy()

        data["symbol"] = symbol

        # Daily price change
        data["daily_change"] = data["Close"].pct_change() * 100

        # Twenty-day average volume
        data["average_volume"] = (
            data["Volume"]
            .rolling(window=20)
            .mean()
        )

        # Relative volume
        data["relative_volume"] = (
            data["Volume"] / data["average_volume"]
        )

        # Twenty-day moving average
        data["moving_average_20"] = (
            data["Close"]
            .rolling(window=20)
            .mean()
        )

        data["above_20_day_ma"] = (
            data["Close"] > data["moving_average_20"]
        )

        # RSI
        data["rsi"] = calculate_rsi(data["Close"])

        # Future returns
        data["return_1_day"] = (
            data["Close"].shift(-1) / data["Close"] - 1
        ) * 100

        data["return_5_day"] = (
            data["Close"].shift(-5) / data["Close"] - 1
        ) * 100

        data["return_10_day"] = (
            data["Close"].shift(-10) / data["Close"] - 1
        ) * 100

        return data

    except Exception as error:
        print(f"Error processing {symbol}: {error}")
        return None


def calculate_historical_score(data):
    """
    Calculate the same six-point score used by the scanner.
    """
    data = data.copy()

    data["score"] = 0

    price_condition = (
        (data["Close"] >= MIN_PRICE)
        & (data["Close"] <= MAX_PRICE)
    )

    volume_condition = (
        data["average_volume"] >= MIN_AVERAGE_VOLUME
    )

    daily_change_condition = (
        data["daily_change"] >= MIN_DAILY_CHANGE
    )

    relative_volume_condition = (
        data["relative_volume"] >= MIN_RELATIVE_VOLUME
    )

    rsi_condition = data["rsi"] <= MAX_RSI

    moving_average_condition = data["above_20_day_ma"]

    data["score"] += price_condition.astype(int)
    data["score"] += volume_condition.astype(int)
    data["score"] += daily_change_condition.astype(int)
    data["score"] += relative_volume_condition.astype(int)
    data["score"] += rsi_condition.astype(int)
    data["score"] += moving_average_condition.astype(int)

    return data

def select_non_overlapping_signals(data, minimum_score, holding_days):
    selected = []
    next_entry_position = 0

    for position in range(len(data)):
        if position < next_entry_position:
            continue

        row = data.iloc[position]

        if row["score"] >= minimum_score:
            selected.append(row)
            next_entry_position = position + holding_days

    if not selected:
        return pd.DataFrame(columns=data.columns)

    return pd.DataFrame(selected)

def backtest_stock(symbol):
    data = prepare_stock_data(symbol)

    if data is None:
        return None

    data = calculate_historical_score(data)

    signal_mask = data["score"] >= MIN_SIGNAL_SCORE

    # Preserve original row positions before filtering.
    data = data.reset_index()
    signals = data[signal_mask.to_numpy()].copy()

    if signals.empty:
        return None

    signals = remove_overlapping_signals(
        signals,
        cooldown_days=5,
    )

    signals = signals.rename(
        columns={
            "Date": "date",
            "Close": "entry_price",
            "Volume": "volume",
        }
    )

    columns = [
        "date",
        "symbol",
        "entry_price",
        "daily_change",
        "volume",
        "average_volume",
        "relative_volume",
        "rsi",
        "moving_average_20",
        "above_20_day_ma",
        "score",
        "return_1_day",
        "return_5_day",
        "return_10_day",
    ]

    return signals[columns]

def create_summary(results):
    summary = {
        "total_signals": len(results),
        "average_score": results["score"].mean(),

        "average_1_day_return":
            results["return_1_day"].mean(),

        "average_5_day_return":
            results["return_5_day"].mean(),

        "average_10_day_return":
            results["return_10_day"].mean(),

        "median_1_day_return":
            results["return_1_day"].median(),

        "median_5_day_return":
            results["return_5_day"].median(),

        "median_10_day_return":
            results["return_10_day"].median(),

        "win_rate_1_day":
            (results["return_1_day"] > 0).mean() * 100,

        "win_rate_5_day":
            (results["return_5_day"] > 0).mean() * 100,

        "win_rate_10_day":
            (results["return_10_day"] > 0).mean() * 100,
    }

    return pd.DataFrame([summary])


def create_symbol_summary(results):
    return (
        results.groupby("symbol")
        .agg(
            signals=("symbol", "size"),
            average_score=("score", "mean"),
            average_1_day_return=("return_1_day", "mean"),
            average_5_day_return=("return_5_day", "mean"),
            average_10_day_return=("return_10_day", "mean"),
            win_rate_1_day=(
                "return_1_day",
                lambda values: (values > 0).mean() * 100,
            ),
            win_rate_5_day=(
                "return_5_day",
                lambda values: (values > 0).mean() * 100,
            ),
            win_rate_10_day=(
                "return_10_day",
                lambda values: (values > 0).mean() * 100,
            ),
        )
        .reset_index()
        .sort_values(
            by="average_5_day_return",
            ascending=False,
        )
    )


def main():
    all_results = []

    for symbol in TICKERS:
        print(f"Backtesting {symbol}...")

        stock_results = backtest_stock(symbol)

        if stock_results is not None:
            all_results.append(stock_results)

    if not all_results:
        print("No historical signals were found.")
        return

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    overall_summary = create_summary(results)
    symbol_summary = create_symbol_summary(results)

    print("\nOVERALL BACKTEST SUMMARY")
    print("=" * 80)
    print(overall_summary.to_string(index=False))

    print("\nRESULTS BY SYMBOL")
    print("=" * 80)
    print(symbol_summary.to_string(index=False))

    results.to_csv(
        "backtest_signals.csv",
        index=False,
    )

    overall_summary.to_csv(
        "backtest_summary.csv",
        index=False,
    )

    symbol_summary.to_csv(
        "backtest_by_symbol.csv",
        index=False,
    )

    print("\nFiles created:")
    print("backtest_signals.csv")
    print("backtest_summary.csv")
    print("backtest_by_symbol.csv")


if __name__ == "__main__":
    main()