import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

# Load symbols from tickers.csv
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
MIN_DAILY_CHANGE = 1.0
MIN_RELATIVE_VOLUME = 1.0
MAX_RSI = 70
MIN_SIGNAL_SCORE = 6
HOLDING_DAYS = 5
BACKTEST_PERIOD = "5y"


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(close_prices, period=14):
    """
    Calculate RSI using rolling average gains and losses.
    """
    price_changes = close_prices.diff()

    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)

    average_gain = gains.rolling(window=period).mean()
    average_loss = losses.rolling(window=period).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, float("nan"))
    )

    return 100 - (100 / (1 + relative_strength))


def calculate_historical_score(data):
    """
    Calculate each scanner condition separately and assign
    a six-point historical score.
    """
    data = data.copy()

    data["condition_price"] = (
        (data["Close"] >= MIN_PRICE)
        & (data["Close"] <= MAX_PRICE)
    )

    data["condition_average_volume"] = (
        data["average_volume"] >= MIN_AVERAGE_VOLUME
    )

    data["condition_daily_change"] = (
        data["daily_change"] >= MIN_DAILY_CHANGE
    )

    data["condition_relative_volume"] = (
        data["relative_volume"] >= MIN_RELATIVE_VOLUME
    )

    data["condition_rsi"] = (
        data["rsi"] <= MAX_RSI
    )

    data["condition_above_ma"] = (
        data["above_20_day_ma"]
    )

    condition_columns = [
        "condition_price",
        "condition_average_volume",
        "condition_daily_change",
        "condition_relative_volume",
        "condition_rsi",
        "condition_above_ma",
    ]

    data["score"] = (
        data[condition_columns]
        .astype(int)
        .sum(axis=1)
    )

    return data


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_stock_data(symbol):
    """
    Download historical data and calculate scanner measurements
    and realistic future returns.

    The signal is generated after the current trading day closes.
    Entry occurs at the next trading day's opening price.
    """
    try:
        data = yf.download(
            symbol,
            period=BACKTEST_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            print(f"No data returned for {symbol}")
            return None

        # yfinance can return MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = {"Open", "Close", "Volume"}

        if not required_columns.issubset(data.columns):
            print(f"Missing required columns for {symbol}")
            return None

        data = data.copy()
        data["symbol"] = symbol

        # Current-day measurements
        data["daily_change"] = (
            data["Close"].pct_change() * 100
        )

        data["average_volume"] = (
            data["Volume"]
            .rolling(window=20)
            .mean()
        )

        data["relative_volume"] = (
            data["Volume"] / data["average_volume"]
        )

        data["moving_average_20"] = (
            data["Close"]
            .rolling(window=20)
            .mean()
        )

        data["above_20_day_ma"] = (
            data["Close"] > data["moving_average_20"]
        )

        data["rsi"] = calculate_rsi(
            data["Close"],
            period=14,
        )

        # The signal is known after today's close.
        # The earliest realistic entry is tomorrow's open.
        data["entry_next_open"] = data["Open"].shift(-1)

        # Exit at the closing price 1, 5, or 10 trading days
        # after the signal date.
        data["exit_1_day_close"] = data["Close"].shift(-1)
        data["exit_5_day_close"] = data["Close"].shift(-5)
        data["exit_10_day_close"] = data["Close"].shift(-10)

        # Returns measured from next-day open
        data["return_1_day"] = (
            (
                data["exit_1_day_close"]
                / data["entry_next_open"]
            ) - 1
        ) * 100

        data["return_5_day"] = (
            (
                data["exit_5_day_close"]
                / data["entry_next_open"]
            ) - 1
        ) * 100

        data["return_10_day"] = (
            (
                data["exit_10_day_close"]
                / data["entry_next_open"]
            ) - 1
        ) * 100

        data = calculate_historical_score(data)

        data = data.reset_index()

        # Handle cases where the index column name differs
        if "Date" in data.columns:
            data = data.rename(columns={"Date": "date"})
        elif "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "date"})
        else:
            data = data.rename(
                columns={data.columns[0]: "date"}
            )

        return data

    except Exception as error:
        print(f"Error processing {symbol}: {error}")
        return None


# ============================================================
# SIGNAL SELECTION
# ============================================================

def select_non_overlapping_signals(
    data,
    minimum_score,
    holding_days,
):
    """
    Select qualifying signals while preventing overlapping
    positions in the same stock.
    """
    selected_positions = []
    next_allowed_position = 0

    for position in range(len(data)):
        if position < next_allowed_position:
            continue

        row = data.iloc[position]

        if pd.isna(row["score"]):
            continue

        if row["score"] >= minimum_score:
            selected_positions.append(position)

            # Ignore additional signals during the holding period
            next_allowed_position = position + holding_days

    if not selected_positions:
        return pd.DataFrame(columns=data.columns)

    return data.iloc[selected_positions].copy()


def backtest_stock(symbol):
    """
    Prepare one stock and select its non-overlapping signals.

    Returns:
        signals: qualifying non-overlapping signals
        baseline: every historically valid trading day
    """
    data = prepare_stock_data(symbol)

    if data is None:
        return None, None

    required_indicator_columns = [
        "average_volume",
        "relative_volume",
        "rsi",
        "moving_average_20",
    ]

    # Remove early rows where indicators are unavailable
    valid_data = data.dropna(
        subset=required_indicator_columns
    ).copy()

    baseline = valid_data.dropna(
        subset=[
            "entry_next_open",
            "return_1_day",
            "return_5_day",
            "return_10_day",
        ]
    ).copy()

    signals = select_non_overlapping_signals(
        data=valid_data,
        minimum_score=MIN_SIGNAL_SCORE,
        holding_days=HOLDING_DAYS,
    )

    # Remove signals too close to the end of the dataset
    signals = signals.dropna(
        subset=[
            "entry_next_open",
            "return_1_day",
            "return_5_day",
            "return_10_day",
        ]
    ).copy()

    if signals.empty:
        signals = None

    return signals, baseline


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

def calculate_return_statistics(data, prefix):
    """
    Calculate average returns, median returns, and win rates.
    """
    return {
        f"{prefix}_observations": len(data),

        f"{prefix}_average_1_day_return":
            data["return_1_day"].mean(),

        f"{prefix}_average_5_day_return":
            data["return_5_day"].mean(),

        f"{prefix}_average_10_day_return":
            data["return_10_day"].mean(),

        f"{prefix}_median_1_day_return":
            data["return_1_day"].median(),

        f"{prefix}_median_5_day_return":
            data["return_5_day"].median(),

        f"{prefix}_median_10_day_return":
            data["return_10_day"].median(),

        f"{prefix}_win_rate_1_day":
            (data["return_1_day"] > 0).mean() * 100,

        f"{prefix}_win_rate_5_day":
            (data["return_5_day"] > 0).mean() * 100,

        f"{prefix}_win_rate_10_day":
            (data["return_10_day"] > 0).mean() * 100,
    }


def create_overall_summary(results, baseline):
    """
    Compare scanner signals with all valid historical days.
    """
    signal_statistics = calculate_return_statistics(
        results,
        prefix="signal",
    )

    baseline_statistics = calculate_return_statistics(
        baseline,
        prefix="baseline",
    )

    summary = {
        "minimum_signal_score": MIN_SIGNAL_SCORE,
        "holding_days": HOLDING_DAYS,
        "average_signal_score": results["score"].mean(),
        **signal_statistics,
        **baseline_statistics,
    }

    # Estimated excess return over the baseline
    summary["excess_1_day_return"] = (
        summary["signal_average_1_day_return"]
        - summary["baseline_average_1_day_return"]
    )

    summary["excess_5_day_return"] = (
        summary["signal_average_5_day_return"]
        - summary["baseline_average_5_day_return"]
    )

    summary["excess_10_day_return"] = (
        summary["signal_average_10_day_return"]
        - summary["baseline_average_10_day_return"]
    )

    summary["excess_1_day_win_rate"] = (
        summary["signal_win_rate_1_day"]
        - summary["baseline_win_rate_1_day"]
    )

    summary["excess_5_day_win_rate"] = (
        summary["signal_win_rate_5_day"]
        - summary["baseline_win_rate_5_day"]
    )

    summary["excess_10_day_win_rate"] = (
        summary["signal_win_rate_10_day"]
        - summary["baseline_win_rate_10_day"]
    )

    return pd.DataFrame([summary])


def create_symbol_summary(results, baseline):
    """
    Compare signal performance and baseline performance
    separately for every symbol.
    """
    signal_summary = (
        results.groupby("symbol")
        .agg(
            signals=("symbol", "size"),
            average_score=("score", "mean"),

            signal_average_1_day_return=(
                "return_1_day",
                "mean",
            ),
            signal_average_5_day_return=(
                "return_5_day",
                "mean",
            ),
            signal_average_10_day_return=(
                "return_10_day",
                "mean",
            ),

            signal_win_rate_1_day=(
                "return_1_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
            signal_win_rate_5_day=(
                "return_5_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
            signal_win_rate_10_day=(
                "return_10_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
        )
        .reset_index()
    )

    baseline_summary = (
        baseline.groupby("symbol")
        .agg(
            baseline_days=("symbol", "size"),

            baseline_average_1_day_return=(
                "return_1_day",
                "mean",
            ),
            baseline_average_5_day_return=(
                "return_5_day",
                "mean",
            ),
            baseline_average_10_day_return=(
                "return_10_day",
                "mean",
            ),

            baseline_win_rate_1_day=(
                "return_1_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
            baseline_win_rate_5_day=(
                "return_5_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
            baseline_win_rate_10_day=(
                "return_10_day",
                lambda values:
                    (values > 0).mean() * 100,
            ),
        )
        .reset_index()
    )

    summary = signal_summary.merge(
        baseline_summary,
        on="symbol",
        how="left",
    )

    summary["excess_1_day_return"] = (
        summary["signal_average_1_day_return"]
        - summary["baseline_average_1_day_return"]
    )

    summary["excess_5_day_return"] = (
        summary["signal_average_5_day_return"]
        - summary["baseline_average_5_day_return"]
    )

    summary["excess_10_day_return"] = (
        summary["signal_average_10_day_return"]
        - summary["baseline_average_10_day_return"]
    )

    summary["excess_1_day_win_rate"] = (
        summary["signal_win_rate_1_day"]
        - summary["baseline_win_rate_1_day"]
    )

    summary["excess_5_day_win_rate"] = (
        summary["signal_win_rate_5_day"]
        - summary["baseline_win_rate_5_day"]
    )

    summary["excess_10_day_win_rate"] = (
        summary["signal_win_rate_10_day"]
        - summary["baseline_win_rate_10_day"]
    )

    return summary.sort_values(
        by="excess_5_day_return",
        ascending=False,
    )


def create_condition_summary(baseline):
    """
    Compare historical performance when each condition passes
    against performance when that condition fails.
    """
    condition_names = {
        "condition_price": "Price range",
        "condition_average_volume": "Average volume",
        "condition_daily_change": "Daily change",
        "condition_relative_volume": "Relative volume",
        "condition_rsi": "RSI",
        "condition_above_ma": "Above 20-day MA",
    }

    rows = []

    for column, display_name in condition_names.items():
        passed = baseline[baseline[column] == True]
        failed = baseline[baseline[column] == False]

        if passed.empty or failed.empty:
            continue

        row = {
            "condition": display_name,
            "passed_days": len(passed),
            "failed_days": len(failed),

            "passed_average_1_day_return":
                passed["return_1_day"].mean(),

            "failed_average_1_day_return":
                failed["return_1_day"].mean(),

            "excess_1_day_return":
                passed["return_1_day"].mean()
                - failed["return_1_day"].mean(),

            "passed_average_5_day_return":
                passed["return_5_day"].mean(),

            "failed_average_5_day_return":
                failed["return_5_day"].mean(),

            "excess_5_day_return":
                passed["return_5_day"].mean()
                - failed["return_5_day"].mean(),

            "passed_average_10_day_return":
                passed["return_10_day"].mean(),

            "failed_average_10_day_return":
                failed["return_10_day"].mean(),

            "excess_10_day_return":
                passed["return_10_day"].mean()
                - failed["return_10_day"].mean(),

            "passed_win_rate_1_day":
                (passed["return_1_day"] > 0).mean() * 100,

            "failed_win_rate_1_day":
                (failed["return_1_day"] > 0).mean() * 100,

            "excess_win_rate_1_day":
                (
                    (passed["return_1_day"] > 0).mean()
                    - (failed["return_1_day"] > 0).mean()
                ) * 100,

            "passed_win_rate_5_day":
                (passed["return_5_day"] > 0).mean() * 100,

            "failed_win_rate_5_day":
                (failed["return_5_day"] > 0).mean() * 100,

            "excess_win_rate_5_day":
                (
                    (passed["return_5_day"] > 0).mean()
                    - (failed["return_5_day"] > 0).mean()
                ) * 100,

            "passed_win_rate_10_day":
                (passed["return_10_day"] > 0).mean() * 100,

            "failed_win_rate_10_day":
                (failed["return_10_day"] > 0).mean() * 100,

            "excess_win_rate_10_day":
                (
                    (passed["return_10_day"] > 0).mean()
                    - (failed["return_10_day"] > 0).mean()
                ) * 100,
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    return summary.sort_values(
        by="excess_5_day_return",
        ascending=False,
    )

# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    all_signals = []
    all_baseline_rows = []

    for symbol in TICKERS:
        print(f"Backtesting {symbol}...")

        signals, baseline = backtest_stock(symbol)

        if baseline is not None and not baseline.empty:
            all_baseline_rows.append(baseline)

        if signals is not None and not signals.empty:
            all_signals.append(signals)

    if not all_signals:
        print("No historical signals were found.")
        return

    if not all_baseline_rows:
        print("No valid baseline data was found.")
        return

    results = pd.concat(
        all_signals,
        ignore_index=True,
    )

    baseline = pd.concat(
        all_baseline_rows,
        ignore_index=True,
    )

    overall_summary = create_overall_summary(
        results,
        baseline,
    )

    symbol_summary = create_symbol_summary(
        results,
        baseline,
    )

    condition_summary = create_condition_summary(
        baseline,
    )

    signal_columns = [
        "date",
        "symbol",
        "Close",
        "entry_next_open",
        "exit_1_day_close",
        "exit_5_day_close",
        "exit_10_day_close",
        "daily_change",
        "Volume",
        "average_volume",
        "relative_volume",
        "rsi",
        "moving_average_20",
        "above_20_day_ma",
        "condition_price",
        "condition_average_volume",
        "condition_daily_change",
        "condition_relative_volume",
        "condition_rsi",
        "condition_above_ma",
        "score",
        "return_1_day",
        "return_5_day",
        "return_10_day",
    ]

    available_signal_columns = [
        column
        for column in signal_columns
        if column in results.columns
    ]

    results = results[available_signal_columns]

    print("\nOVERALL BACKTEST SUMMARY")
    print("=" * 100)
    print(overall_summary.to_string(index=False))

    print("\nRESULTS BY SYMBOL")
    print("=" * 100)
    print(symbol_summary.to_string(index=False))

    print("\nRESULTS BY CONDITION")
    print("=" * 100)

    if condition_summary.empty:
        print("No condition analysis could be created.")
    else:
        print(condition_summary.to_string(index=False))

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

    condition_summary.to_csv(
        "backtest_by_condition.csv",
        index=False,
    )

    print("\nFiles created:")
    print("backtest_signals.csv")
    print("backtest_summary.csv")
    print("backtest_by_symbol.csv")
    print("backtest_by_condition.csv")


if __name__ == "__main__":
    main()