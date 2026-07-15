import pandas as pd
import yfinance as yf


ticker_data = pd.read_csv("tickers.csv")
TICKERS = ticker_data["symbol"].dropna().str.upper().tolist()

MIN_PRICE = 5
MAX_PRICE = 500
MIN_AVERAGE_VOLUME = 1_000_000
MIN_DAILY_CHANGE = 0
MIN_RELATIVE_VOLUME = 0.5
MAX_RSI = 80
REQUIRE_ABOVE_20_DAY_MA = False


def calculate_rsi(close_prices, period=14):
    price_changes = close_prices.diff()

    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)

    average_gain = gains.rolling(window=period).mean()
    average_loss = losses.rolling(window=period).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))

    return 100 - (100 / (1 + relative_strength))

def analyze_stock(symbol):
    try:
        data = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if data.empty or len(data) < 2:
            return None

        current_price = float(data["Close"].iloc[-1])
        previous_price = float(data["Close"].iloc[-2])
        current_volume = int(data["Volume"].iloc[-1])
        average_volume = float(data["Volume"].tail(20).mean())
        relative_volume = current_volume / average_volume

        moving_average_20 = float(data["Close"].tail(20).mean())

        rsi_values = calculate_rsi(data["Close"])
        current_rsi = float(rsi_values.iloc[-1])

        above_20_day_ma = current_price > moving_average_20

        daily_change = (
            (current_price - previous_price) / previous_price
        ) * 100

        reasons = []

        if daily_change >= MIN_DAILY_CHANGE:
            reasons.append("positive momentum")

        if relative_volume >= MIN_RELATIVE_VOLUME:
            reasons.append(
                f"relative volume {relative_volume:.2f} above {MIN_RELATIVE_VOLUME}"
            )

        if above_20_day_ma:
            reasons.append("above 20-day moving average")

        if current_rsi <= MAX_RSI:
            reasons.append(f"RSI {current_rsi:.2f} below {MAX_RSI}")

        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "daily_change": round(daily_change, 2),
            "current_volume": current_volume,
            "average_volume": int(average_volume),
            "relative_volume": round(relative_volume, 2),
            "rsi": round(current_rsi, 2),
            "moving_average_20": round(moving_average_20, 2),
            "above_20_day_ma": above_20_day_ma,
            "reason": ", ".join(reasons),
        }


    except Exception as error:
        print(f"Error checking {symbol}: {error}")
        return None


def meets_requirements(stock):
    return (
        MIN_PRICE <= stock["price"] <= MAX_PRICE
        and stock["average_volume"] >= MIN_AVERAGE_VOLUME
        and stock["daily_change"] >= MIN_DAILY_CHANGE
        and stock["relative_volume"] >= MIN_RELATIVE_VOLUME
        and stock["rsi"] <= MAX_RSI
        and (
            not REQUIRE_ABOVE_20_DAY_MA
            or stock["above_20_day_ma"]
        )
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


def main():
    analyzed_stocks = []

    for symbol in TICKERS:
        print(f"Checking {symbol}...")

        stock = analyze_stock(symbol)

        if stock:
            stock["score"] = calculate_score(stock)
            analyzed_stocks.append(stock)

    if not analyzed_stocks:
        print("\nNo stocks could be analyzed.")
        return

    results = pd.DataFrame(analyzed_stocks)

    results = results.sort_values(
        by=["score", "daily_change"],
        ascending=[False, False],
    )

    print("\nRanked scanner results:")
    print(results.to_string(index=False))

    results.to_csv("results.csv", index=False)
    print("\nResults saved to results.csv")


if __name__ == "__main__":
    main()