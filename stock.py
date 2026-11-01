# Install once if needed:
# pip install yfinance pandas matplotlib

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

INDIAN_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
]


def get_day_change(ticker):
    history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
    if history.empty or "Close" not in history.columns:
        return None

    last_close = float(history["Close"].iloc[-1])
    previous_close = float(history["Close"].iloc[-2]) if len(history) > 1 else last_close
    change_percent = ((last_close - previous_close) / previous_close) * 100 if previous_close else 0.0
    return {
        "Ticker": ticker,
        "Last Price": round(last_close, 2),
        "Day Change %": round(change_percent, 2),
        "Trend": "UP" if change_percent >= 0 else "DOWN",
    }


def get_nifty_snapshot():
    nifty = yf.Ticker("^NSEI").history(period="5d", interval="1d", auto_adjust=True)
    if nifty.empty or "Close" not in nifty.columns:
        raise RuntimeError("Nifty data is not available right now. Please check your internet connection.")

    last_nifty = float(nifty["Close"].iloc[-1])
    previous_nifty = float(nifty["Close"].iloc[-2]) if len(nifty) > 1 else last_nifty
    nifty_change = ((last_nifty - previous_nifty) / previous_nifty) * 100 if previous_nifty else 0.0
    return last_nifty, nifty_change


def main():
    print("\nIndia Stock Market Snapshot (NSE)\n")
    nifty_value, nifty_change = get_nifty_snapshot()
    print(f"NIFTY 50: {nifty_value:,.2f}   Change: {nifty_change:+.2f}%")

    market_rows = []
    for ticker in INDIAN_STOCKS:
        data = get_day_change(ticker)
        if data:
            market_rows.append(data)

    df = pd.DataFrame(market_rows).sort_values("Day Change %", ascending=False)
    if df.empty:
        print("\nNo Indian stock data was returned. Please try again later.")
        return

    print("\nTop Indian stocks today:\n")
    print(df.to_string(index=False))

    top_3 = df.head(3)
    plt.figure(figsize=(12, 6))
    for ticker in top_3["Ticker"]:
        history = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
        if not history.empty and "Close" in history.columns:
            plt.plot(history.index, history["Close"], label=ticker)

    plt.title("Recent Price Trend - Top Indian Stocks")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.tight_layout()
    output_path = "stock_market_graph.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGraph saved to: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()