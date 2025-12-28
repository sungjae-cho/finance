"""
Portfolio Tracker with Interactive Visualization
Reads transactions from CSV, fetches stock prices, and plots portfolio value vs invested amount.
Tracks: unrealized gains, realized gains, and dividend income PER PORTFOLIO.
Supports multiple portfolios with checkbox filtering in the UI.
Supports international stocks with automatic currency conversion.

USAGE:
    1. Edit 'transactions.csv' with your own transactions (include 'portfolio' column)
       - For Korean stocks, use Yahoo Finance format: 005930.KS (KOSPI) or 035720.KQ (KOSDAQ)
    2. Set USE_MOCK_DATA = False to fetch real stock prices
    3. Run: python portfolio_tracker.py
    4. Open 'portfolio_chart.html' in your browser
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Import ticker display names from separate file (used as fallback/override)
from parameters import *


def load_company_names_cache() -> dict:
    """Load cached company names from file."""
    if os.path.exists(COMPANY_NAMES_CACHE_FILE):
        try:
            with open(COMPANY_NAMES_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_company_names_cache(cache: dict):
    """Save company names cache to file."""
    try:
        with open(COMPANY_NAMES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save company names cache: {e}")


def fetch_company_name(ticker: str) -> str:
    """Fetch company name from yfinance API."""
    if USE_MOCK_DATA:
        return None

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try different name fields in order of preference
        name = info.get('longName') or info.get('shortName') or info.get('name')
        return name
    except Exception as e:
        print(f"  Warning: Could not fetch company name for {ticker}: {e}")
        return None


def get_company_names(tickers: list) -> dict:
    """
    Get company names for all tickers.
    Priority: 1) Manual override (TICKER_DISPLAY_NAMES), 2) Cache, 3) API fetch
    """
    names = {}
    cache = load_company_names_cache()
    cache_updated = False

    print("Fetching company names...")

    for ticker in tickers:
        # 1. Check manual override first
        if ticker in TICKER_DISPLAY_NAMES:
            names[ticker] = TICKER_DISPLAY_NAMES[ticker]
            print(f"  {ticker}: {names[ticker]} (from config)")
            continue

        # 2. Check cache
        if ticker in cache:
            names[ticker] = cache[ticker]
            print(f"  {ticker}: {names[ticker]} (from cache)")
            continue

        # 3. Fetch from API
        if not USE_MOCK_DATA:
            fetched_name = fetch_company_name(ticker)
            if fetched_name:
                names[ticker] = fetched_name
                cache[ticker] = fetched_name
                cache_updated = True
                print(f"  {ticker}: {names[ticker]} (from API)")
                continue

        # 4. Fallback: clean ticker symbol
        fallback_name = ticker
        for suffix in EXCHANGE_CURRENCY.keys():
            if ticker.endswith(suffix):
                fallback_name = ticker.replace(suffix, '')
                break
        names[ticker] = fallback_name
        print(f"  {ticker}: {names[ticker]} (fallback)")

    # Save updated cache
    if cache_updated:
        save_company_names_cache(cache)

    return names


def get_ticker_display_name(ticker: str, display_names: dict = None) -> str:
    """Get human-readable display name for a ticker."""
    if display_names and ticker in display_names:
        return display_names[ticker]
    if ticker in TICKER_DISPLAY_NAMES:
        return TICKER_DISPLAY_NAMES[ticker]
    # Fallback: remove exchange suffix for cleaner display
    for suffix in EXCHANGE_CURRENCY.keys():
        if ticker.endswith(suffix):
            return ticker.replace(suffix, '')
    return ticker


def get_ticker_currency(ticker: str) -> str:
    """Determine the currency of a ticker based on its exchange suffix."""
    for suffix, currency in EXCHANGE_CURRENCY.items():
        if ticker.upper().endswith(suffix):
            return currency
    return BASE_CURRENCY  # Default to base currency (US stocks)


def get_required_currencies(tickers: list) -> set:
    """Get set of currencies needed for the given tickers."""
    currencies = set()
    for ticker in tickers:
        curr = get_ticker_currency(ticker)
        if curr != BASE_CURRENCY:
            currencies.add(curr)
    return currencies


def generate_mock_exchange_rates(currencies: set, start_date, end_date) -> dict:
    """Generate mock exchange rate data for demonstration."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Approximate exchange rates to USD
    base_rates = {
        'KRW': 1300.0,  # 1 USD = 1300 KRW
        'JPY': 150.0,   # 1 USD = 150 JPY
        'EUR': 0.92,    # 1 USD = 0.92 EUR
        'GBP': 0.79,    # 1 USD = 0.79 GBP
        'HKD': 7.8,     # 1 USD = 7.8 HKD
        'CAD': 1.36,    # 1 USD = 1.36 CAD
        'AUD': 1.53,    # 1 USD = 1.53 AUD
    }

    np.random.seed(123)

    rates = {}
    for curr in currencies:
        base_rate = base_rates.get(curr, 1.0)
        # Add small random fluctuations
        n = len(dates)
        daily_changes = np.random.normal(0, 0.002, n)
        cumulative = np.exp(np.cumsum(daily_changes))
        rates[curr] = pd.Series(base_rate * cumulative, index=dates)

    return rates


def get_exchange_rates(currencies: set, start_date, end_date) -> dict:
    """Fetch exchange rates from yfinance."""
    if USE_MOCK_DATA:
        return generate_mock_exchange_rates(currencies, start_date, end_date)

    if not currencies:
        return {}

    import yfinance as yf

    rates = {}
    for curr in currencies:
        # Yahoo Finance forex symbol format: USDKRW=X means 1 USD = X KRW
        symbol = f'{BASE_CURRENCY}{curr}=X'
        print(f"  Fetching exchange rate: {symbol}")
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)['Close']
            if len(data) > 0:
                # Convert timezone-aware index to timezone-naive
                if data.index.tz is not None:
                    data = data.copy()
                    data.index = data.index.tz_localize(None)
                rates[curr] = data
            else:
                print(f"  Warning: No exchange rate data for {curr}, using fallback")
                rates[curr] = None
        except Exception as e:
            print(f"  Warning: Could not fetch {symbol}: {e}")
            rates[curr] = None

    return rates


def convert_to_base_currency(value: float, currency: str, exchange_rates: dict, date) -> float:
    """Convert a value from foreign currency to base currency."""
    if currency == BASE_CURRENCY:
        return value

    if currency not in exchange_rates or exchange_rates[currency] is None:
        # Fallback rates if data unavailable
        fallback_rates = {'KRW': 1300, 'JPY': 150, 'EUR': 0.92, 'GBP': 0.79, 'HKD': 7.8, 'CAD': 1.36, 'AUD': 1.53}
        rate = fallback_rates.get(currency, 1.0)
        return value / rate

    rates_series = exchange_rates[currency]

    # Normalize the date to be timezone-naive for comparison
    if hasattr(date, 'tz') and date.tz is not None:
        date = date.tz_localize(None)
    date = pd.Timestamp(date)
    if date.tz is not None:
        date = date.tz_localize(None)

    # Find the closest date in the exchange rate data
    try:
        if date in rates_series.index:
            rate = rates_series.loc[date]
        else:
            # Use the nearest previous date
            mask = rates_series.index <= date
            if mask.any():
                rate = rates_series.loc[rates_series.index[mask][-1]]
            else:
                rate = rates_series.iloc[0]

        return value / rate  # Convert from foreign currency to base currency
    except:
        return value  # Return as-is if conversion fails


def convert_from_base_currency(value: float, target_currency: str, exchange_rates: dict, date) -> float:
    """Convert a value from base currency (USD) to target currency (e.g., CAD).
    
    Args:
        value: Amount in base currency (USD)
        target_currency: Target currency code (e.g., 'CAD', 'KRW', 'EUR')
        exchange_rates: Dictionary of exchange rate time series
        date: Date for the exchange rate lookup
        
    Returns:
        Value converted to target currency
    """
    if target_currency == BASE_CURRENCY:
        return value

    # Fallback rates: 1 USD = X target_currency
    fallback_rates = {'KRW': 1300, 'JPY': 150, 'EUR': 0.92, 'GBP': 0.79, 'HKD': 7.8, 'CAD': 1.36, 'AUD': 1.53}
    
    if target_currency not in exchange_rates or exchange_rates[target_currency] is None:
        rate = fallback_rates.get(target_currency, 1.0)
        return value * rate

    rates_series = exchange_rates[target_currency]

    # Normalize the date to be timezone-naive for comparison
    if hasattr(date, 'tz') and date.tz is not None:
        date = date.tz_localize(None)
    date = pd.Timestamp(date)
    if date.tz is not None:
        date = date.tz_localize(None)

    # Find the closest date in the exchange rate data
    try:
        if date in rates_series.index:
            rate = rates_series.loc[date]
        else:
            # Use the nearest previous date
            mask = rates_series.index <= date
            if mask.any():
                rate = rates_series.loc[rates_series.index[mask][-1]]
            else:
                rate = rates_series.iloc[0]

        return value * rate  # Multiply to convert from base currency to target currency
    except:
        return value  # Return as-is if conversion fails


def convert_to_cad(value: float, exchange_rates: dict, date) -> float:
    """Convenience function to convert USD to CAD.
    
    Args:
        value: Amount in USD
        exchange_rates: Dictionary of exchange rate time series
        date: Date for the exchange rate lookup
        
    Returns:
        Value converted to CAD
    """
    return convert_from_base_currency(value, 'CAD', exchange_rates, date)


def load_transactions(csv_path: str) -> pd.DataFrame:
    """Load and parse transactions from CSV file."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # Sort by date, then by action (buy before sell) to ensure proper order
    action_order = {'buy': 0, 'sell': 1}
    df['_action_order'] = df['action'].map(action_order).fillna(2)
    df = df.sort_values(['date', '_action_order']).reset_index(drop=True)
    df = df.drop(columns=['_action_order'])
    return df


def generate_mock_prices(tickers: list, start_date, end_date) -> pd.DataFrame:
    """Generate realistic mock price data for demonstration.

    Simulates SPLIT-ADJUSTED prices - all historical prices are divided by
    all future split ratios. This matches yfinance's default behavior.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Stock splits to apply (same as in generate_mock_splits)
    stock_splits = {
        'NVDA': [('2021-07-20', 4.0), ('2024-06-10', 10.0)],
        'GOOGL': [('2022-07-15', 20.0)],
        'TSLA': [('2022-08-25', 3.0)],
    }

    # Base prices in local currency - these represent CURRENT (post-split) prices
    # For split-adjusted data, we work backwards from current prices
    base_prices = {
        'AAPL': 130, 'MSFT': 270, 'GOOGL': 120, 'NVDA': 13.5,
        'TSLA': 280, 'UBER': 55, 'QCOM': 160, 'U': 110,
        'ASML': 700, 'ABNB': 170, 'ARM': 130,
        # Korean stocks (prices in KRW)
        '005930.KS': 85000,   # Samsung Electronics
        '000660.KS': 140000,  # SK Hynix
        '035720.KS': 450000,  # Kakao (KOSPI)
        '035720.KQ': 450000,  # Kakao (KOSDAQ)
        '035420.KS': 350000,  # Naver
        '012450.KS': 900000,  # Samsung Elec Mech
        '079550.KS': 450000,  # LIG Nex1
        '042660.KS': 140000,  # Daewoo Shipbuilding
        '006800.KS': 25000,   # Mirae Asset
        '064350.KS': 230000,  # Hyundai Rotem
    }
    growth_rates = {
        'AAPL': 0.0004, 'MSFT': 0.0003, 'GOOGL': 0.0003, 'NVDA': 0.0008,
        'TSLA': 0.0003, 'UBER': 0.0002, 'QCOM': 0.0002, 'U': -0.0003,
        'ASML': 0.0003, 'ABNB': 0.0001, 'ARM': 0.0004,
        '005930.KS': 0.0001, '000660.KS': 0.0003, '035720.KS': -0.0002,
        '035720.KQ': -0.0002, '035420.KS': 0.0001, '012450.KS': 0.0001,
        '079550.KS': 0.0001, '042660.KS': 0.0001, '006800.KS': 0.0001,
        '064350.KS': 0.0001,
    }
    volatility = {
        'AAPL': 0.018, 'MSFT': 0.015, 'GOOGL': 0.018, 'NVDA': 0.028,
        'TSLA': 0.035, 'UBER': 0.025, 'QCOM': 0.020, 'U': 0.035,
        'ASML': 0.022, 'ABNB': 0.028, 'ARM': 0.030,
        '005930.KS': 0.018, '000660.KS': 0.025, '035720.KS': 0.030,
        '035720.KQ': 0.030, '035420.KS': 0.022, '012450.KS': 0.020,
        '079550.KS': 0.020, '042660.KS': 0.022, '006800.KS': 0.018,
        '064350.KS': 0.020,
    }

    np.random.seed(42)

    prices = {}
    for ticker in tickers:
        base = base_prices.get(ticker, 100)
        growth = growth_rates.get(ticker, 0.0002)
        vol = volatility.get(ticker, 0.015)
        n = len(dates)
        daily_returns = np.random.normal(growth, vol, n)
        cumulative = np.exp(np.cumsum(daily_returns))
        # Split-adjusted prices: continuous price series with no visible splits
        ticker_prices = base * cumulative

        prices[ticker] = ticker_prices

    return pd.DataFrame(prices, index=dates)


def generate_mock_dividends(tickers: list, start_date, end_date) -> dict:
    """Generate mock dividend data for demonstration."""
    # Quarterly dividend amounts per share (in local currency)
    dividend_rates = {
        'AAPL': 0.25, 'MSFT': 0.75, 'GOOGL': 0.20, 'NVDA': 0.04,
        # Korean stocks (dividends in KRW, typically annual)
        '005930.KS': 361,    # Samsung - ~361 KRW per share annually
        '000660.KS': 1200,   # SK Hynix
        '035420.KS': 500,    # Naver
        '035720.KQ': 0,      # Kakao - no dividend
    }

    # Dividend frequency (Q=quarterly, A=annual)
    dividend_freq = {
        'AAPL': 'Q', 'MSFT': 'Q', 'GOOGL': 'Q', 'NVDA': 'Q',
        '005930.KS': 'A', '000660.KS': 'A', '035420.KS': 'A', '035720.KQ': 'A',
    }

    dividends = {}
    for ticker in tickers:
        rate = dividend_rates.get(ticker, 0)
        freq = dividend_freq.get(ticker, 'Q')

        if rate > 0:
            div_dates = []
            current_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year

            if freq == 'Q':
                # Quarterly dividends (Feb, May, Aug, Nov)
                for year in range(current_year, end_year + 1):
                    for month in [2, 5, 8, 11]:
                        try:
                            div_date = pd.Timestamp(year=year, month=month, day=15)
                            if pd.Timestamp(start_date) <= div_date <= pd.Timestamp(end_date):
                                while div_date.dayofweek >= 5:
                                    div_date = div_date + pd.Timedelta(days=1)
                                div_dates.append(div_date)
                        except:
                            pass
            else:
                # Annual dividends (typically April for Korean stocks)
                for year in range(current_year, end_year + 1):
                    try:
                        div_date = pd.Timestamp(year=year, month=4, day=15)
                        if pd.Timestamp(start_date) <= div_date <= pd.Timestamp(end_date):
                            while div_date.dayofweek >= 5:
                                div_date = div_date + pd.Timedelta(days=1)
                            div_dates.append(div_date)
                    except:
                        pass

            if div_dates:
                dividends[ticker] = pd.Series(rate, index=pd.DatetimeIndex(div_dates))
            else:
                dividends[ticker] = pd.Series(dtype=float)
        else:
            dividends[ticker] = pd.Series(dtype=float)

    return dividends


def get_dividends(tickers: list, start_date, end_date) -> dict:
    """Fetch dividend data for tickers."""
    if USE_MOCK_DATA:
        return generate_mock_dividends(tickers, start_date, end_date)
    else:
        import yfinance as yf
        dividends = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                div = stock.dividends
                if len(div) > 0:
                    # Convert timezone-aware index to timezone-naive for comparison
                    if div.index.tz is not None:
                        div = div.copy()
                        div.index = div.index.tz_localize(None)
                    # Filter to date range
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    mask = (div.index >= start_ts) & (div.index <= end_ts)
                    dividends[ticker] = div[mask]
                else:
                    dividends[ticker] = pd.Series(dtype=float)
            except Exception as e:
                print(f"Warning: Could not fetch dividends for {ticker}: {e}")
                dividends[ticker] = pd.Series(dtype=float)
        return dividends


def generate_mock_splits(tickers: list, start_date, end_date) -> dict:
    """Generate mock stock split data for demonstration."""
    # Actual stock splits (ratio means multiply shares by this)
    mock_splits = {
        'NVDA': [('2021-07-20', 4.0), ('2024-06-10', 10.0)],  # 4:1 then 10:1 split
        'GOOGL': [('2022-07-15', 20.0)],  # 20:1 split
        'TSLA': [('2022-08-25', 3.0)],   # 3:1 split
    }

    splits = {}
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for ticker in tickers:
        if ticker in mock_splits and mock_splits[ticker]:
            split_dates = []
            split_ratios = []
            for date_str, ratio in mock_splits[ticker]:
                split_date = pd.Timestamp(date_str)
                if start_ts <= split_date <= end_ts:
                    split_dates.append(split_date)
                    split_ratios.append(ratio)
            if split_dates:
                splits[ticker] = pd.Series(split_ratios, index=pd.DatetimeIndex(split_dates))
            else:
                splits[ticker] = pd.Series(dtype=float)
        else:
            splits[ticker] = pd.Series(dtype=float)

    return splits


def get_stock_splits(tickers: list, start_date, end_date) -> dict:
    """Fetch stock split data for tickers."""
    if USE_MOCK_DATA:
        return generate_mock_splits(tickers, start_date, end_date)
    else:
        import yfinance as yf
        splits = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                split_data = stock.splits
                if len(split_data) > 0:
                    # Convert timezone-aware index to timezone-naive for comparison
                    if split_data.index.tz is not None:
                        split_data = split_data.copy()
                        split_data.index = split_data.index.tz_localize(None)
                    # Filter to date range
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    mask = (split_data.index >= start_ts) & (split_data.index <= end_ts)
                    splits[ticker] = split_data[mask]
                else:
                    splits[ticker] = pd.Series(dtype=float)
            except Exception as e:
                print(f"Warning: Could not fetch splits for {ticker}: {e}")
                splits[ticker] = pd.Series(dtype=float)
        return splits


def calculate_portfolio_history(transactions: pd.DataFrame) -> tuple:
    """
    Calculate daily portfolio metrics for each portfolio separately.
    Handles multi-currency conversion for international stocks.
    Returns: (history_by_portfolio dict, portfolio_names list)
    """
    start_date = transactions['date'].min()
    end_date = datetime.today()

    tickers = transactions['ticker'].unique().tolist()
    portfolios = transactions['portfolio'].unique().tolist()

    # Identify required currencies and fetch exchange rates
    required_currencies = get_required_currencies(tickers)
    if required_currencies:
        print(f"International stocks detected. Currencies needed: {', '.join(required_currencies)}")

    # Map each ticker to its currency
    ticker_currency = {ticker: get_ticker_currency(ticker) for ticker in tickers}

    if USE_MOCK_DATA:
        print(f"Using mock price data for: {', '.join(tickers)}")
        prices = generate_mock_prices(tickers, start_date, end_date)
    else:
        import yfinance as yf
        print(f"Downloading price data for: {', '.join(tickers)}")
        try:
            # yfinance returns split-adjusted prices by default (auto_adjust=True)
            # Historical prices are already divided by all future split ratios
            # We handle splits by multiplying share counts for holdings bought before split dates
            data = yf.download(tickers, start=start_date, end=end_date, progress=True)
            prices = data['Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
            if prices.empty:
                raise ValueError("No price data returned")
        except Exception as e:
            print(f"Error downloading prices: {e}")
            print("Falling back to mock data...")
            prices = generate_mock_prices(tickers, start_date, end_date)

    prices = prices.ffill()

    # Fetch exchange rates if needed
    exchange_rates = {}
    # Always include CAD and KRW for display currency options, plus any currencies from stocks
    display_currencies = {'CAD', 'KRW'}
    currencies_to_fetch = required_currencies.union(display_currencies)
    if currencies_to_fetch:
        print(f"Fetching exchange rates for: {', '.join(sorted(currencies_to_fetch))}")
        exchange_rates = get_exchange_rates(currencies_to_fetch, start_date, end_date)
        # Forward fill exchange rates
        for curr in exchange_rates:
            if exchange_rates[curr] is not None:
                exchange_rates[curr] = exchange_rates[curr].ffill()

    print("Fetching dividend data...")
    dividends = get_dividends(tickers, start_date, end_date)

    # Debug: print dividend info
    for ticker in tickers:
        currency_info = f" ({ticker_currency[ticker]})" if ticker_currency[ticker] != BASE_CURRENCY else ""
        if ticker in dividends and len(dividends[ticker]) > 0:
            print(f"  {ticker}{currency_info}: {len(dividends[ticker])} dividend payments found")
        else:
            print(f"  {ticker}{currency_info}: No dividends")

    print("Fetching stock split data...")
    splits = get_stock_splits(tickers, start_date, end_date)

    # Debug: print split info
    for ticker in tickers:
        if ticker in splits and len(splits[ticker]) > 0:
            for split_date, ratio in splits[ticker].items():
                print(f"  {ticker}: {ratio}:1 split on {split_date.strftime('%Y-%m-%d')}")

    # Initialize per-portfolio, per-stock tracking
    portfolio_state = {}
    for pf in portfolios:
        portfolio_state[pf] = {}
        for ticker in tickers:
            portfolio_state[pf][ticker] = {
                'holdings': 0,
                'cost_basis': 0,  # Stored in BASE_CURRENCY
                'realized_gain': 0,
                'dividends': 0,
                'total_invested': 0,
                'total_withdrawn': 0
            }

    # Track which dividends have been processed (per portfolio, per ticker)
    processed_dividends = {pf: {ticker: set() for ticker in tickers} for pf in portfolios}

    # Helper function to calculate cumulative split ratio for future splits
    def get_future_split_ratio(ticker: str, from_date: pd.Timestamp) -> float:
        """Calculate cumulative split ratio for all splits occurring AFTER from_date."""
        if ticker not in splits or len(splits[ticker]) == 0:
            return 1.0
        ratio = 1.0
        for split_date in splits[ticker].index:
            split_date_ts = pd.Timestamp(split_date)
            if split_date_ts.date() > from_date.date():
                ratio *= float(splits[ticker][split_date])
        return ratio

    # Build history per portfolio AND per stock
    history_by_portfolio_stock = {pf: {ticker: [] for ticker in tickers} for pf in portfolios}
    transaction_idx = 0

    for date in prices.index:
        date_ts = pd.Timestamp(date)

        # Process transactions for this date
        while transaction_idx < len(transactions):
            tx = transactions.iloc[transaction_idx]
            tx_date = pd.Timestamp(tx['date'])

            if tx_date.date() <= date_ts.date():
                pf = tx['portfolio']
                ticker = tx['ticker']
                shares = float(tx['shares'])  # Ensure scalar
                price_local = float(tx['price'])  # Ensure scalar
                state = portfolio_state[pf][ticker]

                # Convert price to base currency
                curr = ticker_currency[ticker]
                price_base = float(convert_to_base_currency(price_local, curr, exchange_rates, date_ts))

                if tx['action'] == 'buy':
                    # Apply future split ratio to newly bought shares
                    # This accounts for yfinance's split-adjusted prices
                    split_ratio = get_future_split_ratio(ticker, tx_date)
                    adjusted_shares = shares * split_ratio
                    state['holdings'] += adjusted_shares
                    state['cost_basis'] += shares * price_base  # Cost basis uses original shares * price
                    state['total_invested'] += shares * price_base

                elif tx['action'] == 'sell':
                    if state['holdings'] > 0:
                        # Shares in sell transaction should be in post-split terms
                        shares_to_sell = min(shares, state['holdings'])
                        avg_cost = state['cost_basis'] / state['holdings']
                        sale_proceeds = shares_to_sell * price_base
                        cost_of_sold = shares_to_sell * avg_cost
                        state['realized_gain'] += sale_proceeds - cost_of_sold
                        state['total_withdrawn'] += sale_proceeds
                        state['cost_basis'] -= cost_of_sold
                        state['holdings'] -= shares_to_sell
                        print(f"  SELL: {pf}/{ticker} - sold {shares_to_sell:.2f} shares, cost_basis now ${state['cost_basis']:.2f}, holdings now {state['holdings']:.2f}")
                    else:
                        print(f"  WARNING: Sell {shares} shares of {pf}/{ticker} skipped - no holdings!")

                transaction_idx += 1
            else:
                break

        # Process dividends for each portfolio and stock
        for pf in portfolios:
            for ticker in tickers:
                state = portfolio_state[pf][ticker]
                if ticker in dividends and len(dividends[ticker]) > 0:
                    curr = ticker_currency[ticker]
                    for div_date in dividends[ticker].index:
                        div_date_ts = pd.Timestamp(div_date)
                        if div_date_ts.date() <= date_ts.date() and div_date not in processed_dividends[pf][ticker]:
                            if state['holdings'] > 0:
                                div_amount_local = float(dividends[ticker][div_date]) * state['holdings']
                                div_amount_base = float(convert_to_base_currency(div_amount_local, curr, exchange_rates, date_ts))
                                state['dividends'] += div_amount_base
                            processed_dividends[pf][ticker].add(div_date)

        # Calculate daily metrics for each portfolio and stock
        for pf in portfolios:
            for ticker in tickers:
                state = portfolio_state[pf][ticker]

                # Calculate portfolio value for this stock
                stock_value = 0.0
                if state['holdings'] > 0 and ticker in prices.columns and pd.notna(prices.loc[date, ticker]):
                    price_local = float(prices.loc[date, ticker])
                    curr = ticker_currency[ticker]
                    price_base = float(convert_to_base_currency(price_local, curr, exchange_rates, date_ts))
                    stock_value = state['holdings'] * price_base

                cost_basis = float(state['cost_basis'])
                unrealized_gain = stock_value - cost_basis
                total_gain = unrealized_gain + state['realized_gain'] + state['dividends']

                history_by_portfolio_stock[pf][ticker].append({
                    'date': date,
                    'portfolio_value': stock_value,
                    'cost_basis': cost_basis,
                    'total_invested': state['total_invested'],
                    'total_withdrawn': state['total_withdrawn'],
                    'unrealized_gain': unrealized_gain,
                    'realized_gain': state['realized_gain'],
                    'dividends': state['dividends'],
                    'total_gain': total_gain
                })

    # Convert to DataFrames
    for pf in portfolios:
        for ticker in tickers:
            history_by_portfolio_stock[pf][ticker] = pd.DataFrame(history_by_portfolio_stock[pf][ticker])

    # Also create aggregated portfolio data (sum across all stocks in each portfolio)
    history_by_portfolio = {}
    for pf in portfolios:
        pf_data = None
        for ticker in tickers:
            stock_df = history_by_portfolio_stock[pf][ticker]
            if pf_data is None:
                pf_data = stock_df.copy()
            else:
                for col in ['portfolio_value', 'cost_basis', 'total_invested', 'total_withdrawn',
                           'unrealized_gain', 'realized_gain', 'dividends', 'total_gain']:
                    pf_data[col] = pf_data[col] + stock_df[col]
        history_by_portfolio[pf] = pf_data

    # Prepare exchange rate data for export
    exchange_rate_data = {}
    if 'KRW' in exchange_rates and exchange_rates['KRW'] is not None:
        # Create a dict of date -> rate
        krw_rates = exchange_rates['KRW']
        exchange_rate_data['KRW'] = {}
        for d, r in krw_rates.items():
            # Handle both datetime and string keys
            if hasattr(d, 'strftime'):
                key = d.strftime('%Y-%m-%d')
            else:
                key = str(d)[:10]  # Take first 10 chars if already string
            exchange_rate_data['KRW'][key] = r

    if 'CAD' in exchange_rates and exchange_rates['CAD'] is not None:
        # Create a dict of date -> rate
        cad_rates = exchange_rates['CAD']
        exchange_rate_data['CAD'] = {}
        for d, r in cad_rates.items():
            # Handle both datetime and string keys
            if hasattr(d, 'strftime'):
                key = d.strftime('%Y-%m-%d')
            else:
                key = str(d)[:10]  # Take first 10 chars if already string
            exchange_rate_data['CAD'][key] = r

    return history_by_portfolio, history_by_portfolio_stock, portfolios, tickers, exchange_rate_data


def create_interactive_chart(history_by_portfolio: dict, history_by_portfolio_stock: dict,
                            portfolios: list, tickers: list, display_names: dict = None,
                            exchange_rate_data: dict = None, output_path: str = None):
    """Create interactive chart with portfolio and stock checkboxes and currency selector."""

    # Use empty dict if no display names provided
    if display_names is None:
        display_names = {}

    # Get date range
    first_pf = portfolios[0]
    min_date = history_by_portfolio[first_pf]['date'].min().strftime('%Y-%m-%d')
    max_date = history_by_portfolio[first_pf]['date'].max().strftime('%Y-%m-%d')
    dates = history_by_portfolio[first_pf]['date'].dt.strftime('%Y-%m-%d').tolist()

    # Prepare exchange rates for JavaScript (default rate if not available)
    if exchange_rate_data is None:
        exchange_rate_data = {}

    # Fill in exchange rates for all dates
    exchange_rates_js = {}
    if 'KRW' in exchange_rate_data:
        exchange_rates_js['KRW'] = []
        for d in dates:
            if d in exchange_rate_data['KRW']:
                val = exchange_rate_data['KRW'][d]
                # Convert to native Python float
                exchange_rates_js['KRW'].append(float(val) if hasattr(val, 'item') else float(val))
            else:
                # Use closest previous rate or default
                exchange_rates_js['KRW'].append(1300.0)
    else:
        # Default exchange rate if not available
        exchange_rates_js['KRW'] = [1300.0] * len(dates)

    if 'CAD' in exchange_rate_data:
        exchange_rates_js['CAD'] = []
        for d in dates:
            if d in exchange_rate_data['CAD']:
                val = exchange_rate_data['CAD'][d]
                # Convert to native Python float
                exchange_rates_js['CAD'].append(float(val) if hasattr(val, 'item') else float(val))
            else:
                # Use closest previous rate or default
                exchange_rates_js['CAD'].append(1.36)
    else:
        # Default exchange rate if not available
        exchange_rates_js['CAD'] = [1.36] * len(dates)

    # Helper function to convert Series/array to list of native Python floats
    def to_float_list(series):
        """Convert pandas Series to list of native Python floats."""
        return [float(x) for x in series.values]

    # Prepare data for JavaScript
    portfolio_data_js = {}
    for pf in portfolios:
        df = history_by_portfolio[pf]
        portfolio_data_js[pf] = {
            'portfolio_value': to_float_list(df['portfolio_value']),
            'cost_basis': to_float_list(df['cost_basis']),
            'unrealized_gain': to_float_list(df['unrealized_gain']),
            'realized_gain': to_float_list(df['realized_gain']),
            'dividends': to_float_list(df['dividends']),
            'total_gain': to_float_list(df['total_gain']),
            'total_invested': to_float_list(df['total_invested'])
        }

    # Prepare per-stock data for JavaScript
    portfolio_stock_data_js = {}
    for pf in portfolios:
        portfolio_stock_data_js[pf] = {}
        for ticker in tickers:
            df = history_by_portfolio_stock[pf][ticker]
            portfolio_stock_data_js[pf][ticker] = {
                'portfolio_value': to_float_list(df['portfolio_value']),
                'cost_basis': to_float_list(df['cost_basis']),
                'unrealized_gain': to_float_list(df['unrealized_gain']),
                'realized_gain': to_float_list(df['realized_gain']),
                'dividends': to_float_list(df['dividends']),
                'total_gain': to_float_list(df['total_gain']),
                'total_invested': to_float_list(df['total_invested'])
            }

    # Create a simple placeholder figure (actual data managed by JS)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[], y=[], name='Cost Basis', line=dict(color='#E94F37', width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(x=[], y=[], name='Portfolio Value', line=dict(color='#2E86AB', width=2.5), fill='tonexty', fillcolor='rgba(46, 134, 171, 0.2)'))

    fig.update_layout(
        xaxis=dict(title='', showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat='%b<br>%Y', dtick='M1', tickangle=0, automargin=True),
        yaxis=dict(title='Value ($)', showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickprefix='$', tickformat=',.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=30, t=40, b=50),
        autosize=True
    )

    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(label='All', step='all')
            ],
            y=1.0, x=0, xanchor='left', font=dict(size=11)
        )
    )

    if output_path:
        html_str = fig.to_html(include_plotlyjs=True, full_html=False, div_id='portfolio-chart')

        # Generate portfolio checkbox HTML
        portfolio_checkbox_html = ''
        for pf in portfolios:
            portfolio_checkbox_html += f'''
                <label class="checkbox-label">
                    <input type="checkbox" id="cb-pf-{pf}" value="{pf}" checked onchange="onPortfolioChange()">
                    <span class="checkmark"></span>
                    {pf}
                </label>'''

        # Generate stock checkbox HTML
        stock_checkbox_html = ''
        for ticker in tickers:
            # Use human-readable display name from dictionary
            ticker_display = display_names.get(ticker, get_ticker_display_name(ticker, display_names))
            stock_checkbox_html += f'''
                <label class="checkbox-label" title="{ticker}">
                    <input type="checkbox" id="cb-tk-{ticker.replace(".", "_")}" value="{ticker}" checked onchange="onStockChange()">
                    <span class="checkmark"></span>
                    {ticker_display}
                </label>'''

        full_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Portfolio Performance Tracker</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            display: flex;
            flex-direction: row;
        }}

        /* Left Sidebar */
        .sidebar {{
            width: 200px;
            min-width: 200px;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 12px;
        }}
        .sidebar-section {{
            margin-bottom: 16px;
        }}
        .sidebar-section-title {{
            font-size: 11px;
            font-weight: 700;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sidebar-buttons {{
            display: flex;
            gap: 4px;
        }}
        .sidebar-btn {{
            font-size: 9px;
            padding: 2px 6px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: #f8f8f8;
            color: #666;
            cursor: pointer;
            text-transform: none;
            letter-spacing: 0;
        }}
        .sidebar-btn:hover {{
            background: #e8e8e8;
            border-color: #ccc;
        }}
        .sidebar .checkbox-label {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            cursor: pointer;
            padding: 6px 8px;
            border-radius: 4px;
            transition: background 0.2s;
            margin-bottom: 2px;
        }}
        .sidebar .checkbox-label:hover {{ background: #f0f4f8; }}
        .sidebar .checkbox-label input {{ cursor: pointer; margin: 0; }}

        /* Main Content Area */
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 12px;
            gap: 12px;
            min-width: 0;
            overflow: hidden;
        }}

        /* Top Controls Bar */
        .controls {{
            background: white;
            padding: 10px 16px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            flex-shrink: 0;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .control-group label {{
            font-size: 11px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }}
        .control-group input[type="date"], .control-group select {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            background: white;
        }}
        .btn {{
            padding: 6px 16px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }}
        .btn-primary {{ background: #2E86AB; color: white; }}
        .btn-primary:hover {{ background: #236b8a; }}
        .btn-secondary {{ background: #e8e8e8; color: #333; }}
        .btn-secondary:hover {{ background: #ddd; }}
        .divider {{ width: 1px; height: 24px; background: #e0e0e0; margin: 0 4px; }}

        /* Metrics */
        .metrics {{
            display: flex;
            gap: 12px;
            margin-left: auto;
            flex-wrap: wrap;
        }}
        .metric {{ text-align: center; min-width: 70px; }}
        .metric-value {{ font-size: 14px; font-weight: 700; color: #333; }}
        .metric-label {{ font-size: 9px; color: #888; text-transform: uppercase; }}
        .metric-value.positive {{ color: #22863a; }}
        .metric-value.negative {{ color: #cb2431; }}
        .metric-value.unrealized {{ color: #28a745; }}
        .metric-value.realized {{ color: #fd7e14; }}
        .metric-value.dividends {{ color: #6f42c1; }}
        .metric-value.total {{ color: #20c997; }}
        .gains-breakdown {{
            display: flex;
            gap: 8px;
            padding: 4px 8px;
            background: #f8f9fa;
            border-radius: 6px;
        }}

        /* Chart Container */
        .chart-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #portfolio-chart {{ width: 100% !important; height: 100% !important; flex: 1; min-height: 0; }}

        @media (max-width: 1000px) {{
            .metrics {{ margin-left: 0; width: 100%; justify-content: flex-start; order: 10; }}
        }}
        @media (max-width: 768px) {{
            body {{ flex-direction: column; }}
            .sidebar {{ width: 100%; min-width: 100%; flex-direction: row; flex-wrap: wrap; border-right: none; border-bottom: 1px solid #e0e0e0; padding: 8px; }}
            .sidebar-section {{ margin-bottom: 0; margin-right: 16px; }}
            .sidebar-section-title {{ margin-bottom: 4px; }}
        }}
    </style>
</head>
<body>
    <!-- Left Sidebar -->
    <div class="sidebar">
        <div class="sidebar-section">
            <div class="sidebar-section-title">
                Portfolios
                <span class="sidebar-buttons">
                    <button class="sidebar-btn" onclick="selectAllPortfolios()" title="Select All">All</button>
                    <button class="sidebar-btn" onclick="clearAllPortfolios()" title="Clear All">None</button>
                </span>
            </div>
            {portfolio_checkbox_html}
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">
                Stocks
                <span class="sidebar-buttons">
                    <button class="sidebar-btn" onclick="selectAllStocks()" title="Select All">All</button>
                    <button class="sidebar-btn" onclick="clearAllStocks()" title="Clear All">None</button>
                </span>
            </div>
            {stock_checkbox_html}
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <div class="controls">
            <div class="control-group">
                <label>From</label>
                <input type="date" id="start-date" value="{min_date}" min="{min_date}" max="{max_date}">
            </div>
            <div class="control-group">
                <label>To</label>
                <input type="date" id="end-date" value="{max_date}" min="{min_date}" max="{max_date}">
            </div>
            <div class="control-group">
                <label>Ticks</label>
                <select id="tick-format">
                    <option value="auto">Auto</option>
                    <option value="D1">Daily</option>
                    <option value="D7">Weekly</option>
                    <option value="M1">Monthly</option>
                    <option value="M3">Quarterly</option>
                    <option value="M12">Yearly</option>
                </select>
            </div>
            <div class="control-group">
                <label>Currency</label>
                <select id="currency-select" onchange="updateChart()">
                    <option value="USD">USD ($)</option>
                    <option value="CAD">CAD (C$)</option>
                    <option value="KRW">KRW (₩)</option>
                </select>
            </div>
            <button class="btn btn-primary" onclick="applyDateRange()">Apply</button>
            <button class="btn btn-secondary" onclick="resetDateRange()">Reset</button>
            <div class="divider"></div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="metric-portfolio">-</div>
                    <div class="metric-label">Portfolio</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-cost">-</div>
                <div class="metric-label">Cost Basis</div>
            </div>
            <div class="divider"></div>
            <div class="gains-breakdown">
                <div class="metric">
                    <div class="metric-value unrealized" id="metric-unrealized">-</div>
                    <div class="metric-label">Unrealized</div>
                </div>
                <div class="metric">
                    <div class="metric-value realized" id="metric-realized">-</div>
                    <div class="metric-label">Realized</div>
                </div>
                <div class="metric">
                    <div class="metric-value dividends" id="metric-dividends">-</div>
                    <div class="metric-label">Dividends</div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="metric">
                <div class="metric-value total" id="metric-total">-</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metric-total-pct">-</div>
                <div class="metric-label">Total %</div>
            </div>
            <div class="divider"></div>
            <div class="metric">
                <div class="metric-value" id="metric-period">-</div>
                <div class="metric-label">Period Return</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metric-period-pct">-</div>
                <div class="metric-label">Period %</div>
            </div>
        </div>
    </div>
    <div class="chart-container">
        {html_str}
    </div>
    </div>
    <script>
        const portfolioNames = {json.dumps(portfolios)};
        const tickerNames = {json.dumps(tickers)};
        const dates = {json.dumps(dates)};
        const portfolioData = {json.dumps(portfolio_data_js)};
        const portfolioStockData = {json.dumps(portfolio_stock_data_js)};
        const exchangeRates = {json.dumps(exchange_rates_js)};
        const minDate = '{min_date}';
        const maxDate = '{max_date}';

        function getSelectedPortfolios() {{
            return portfolioNames.filter(pf => document.getElementById('cb-pf-' + pf).checked);
        }}

        function getSelectedStocks() {{
            return tickerNames.filter(tk => {{
                const el = document.getElementById('cb-tk-' + tk.replace('.', '_'));
                return el && el.checked && el.parentElement.style.display !== 'none';
            }});
        }}

        function stockExistsInPortfolio(ticker, portfolio) {{
            // Check if a stock has any investment in the given portfolio
            const data = portfolioStockData[portfolio][ticker];
            if (!data) return false;
            // Check if total_invested ever becomes non-zero
            return data.total_invested.some(v => v > 0);
        }}

        function getFirstInvestmentDate(selectedPortfolios, selectedStocks) {{
            // Find the earliest date where any selected stock in any selected portfolio has total_invested > 0
            let firstDateIdx = -1;

            for (let i = 0; i < dates.length; i++) {{
                let hasInvestment = false;
                for (const pf of selectedPortfolios) {{
                    for (const tk of selectedStocks) {{
                        if (portfolioStockData[pf][tk] && portfolioStockData[pf][tk].total_invested[i] > 0) {{
                            hasInvestment = true;
                            break;
                        }}
                    }}
                    if (hasInvestment) break;
                }}
                if (hasInvestment) {{
                    firstDateIdx = i;
                    break;
                }}
            }}

            return firstDateIdx >= 0 ? dates[firstDateIdx] : minDate;
        }}

        function updatePeriodStartDate() {{
            const selectedPf = getSelectedPortfolios();
            const selectedTk = getSelectedStocks();
            if (selectedPf.length === 0 || selectedTk.length === 0) {{
                return;
            }}

            const firstDate = getFirstInvestmentDate(selectedPf, selectedTk);
            document.getElementById('start-date').value = firstDate;
        }}

        function updateStockVisibility() {{
            const selectedPf = getSelectedPortfolios();

            tickerNames.forEach(tk => {{
                const checkboxLabel = document.getElementById('cb-tk-' + tk.replace('.', '_')).parentElement;

                // Check if this stock exists in any of the selected portfolios
                let stockExistsInSelected = false;
                for (const pf of selectedPf) {{
                    if (stockExistsInPortfolio(tk, pf)) {{
                        stockExistsInSelected = true;
                        break;
                    }}
                }}

                // Show or hide the checkbox
                if (stockExistsInSelected) {{
                    checkboxLabel.style.display = 'flex';
                }} else {{
                    checkboxLabel.style.display = 'none';
                    // Uncheck hidden stocks
                    document.getElementById('cb-tk-' + tk.replace('.', '_')).checked = false;
                }}
            }});
        }}

        function onPortfolioChange() {{
            updateStockVisibility();
            updatePeriodStartDate();
            updateChart();

            // Apply the updated date range to the chart
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const tickMode = document.getElementById('tick-format').value;
            const {{ dtick, tickformat }} = getSmartTickFormat(startDate, endDate, tickMode);
            Plotly.relayout('portfolio-chart', {{
                'xaxis.range': [startDate, endDate],
                'xaxis.dtick': dtick,
                'xaxis.tickformat': tickformat
            }});
            updateMetrics(startDate, endDate);
        }}

        function selectAllPortfolios() {{
            portfolioNames.forEach(pf => document.getElementById('cb-pf-' + pf).checked = true);
            onPortfolioChange();
        }}

        function clearAllPortfolios() {{
            portfolioNames.forEach(pf => document.getElementById('cb-pf-' + pf).checked = false);
            onPortfolioChange();
        }}

        function onStockChange() {{
            updatePeriodStartDate();
            updateChart();

            // Apply the updated date range to the chart
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const tickMode = document.getElementById('tick-format').value;
            const {{ dtick, tickformat }} = getSmartTickFormat(startDate, endDate, tickMode);
            Plotly.relayout('portfolio-chart', {{
                'xaxis.range': [startDate, endDate],
                'xaxis.dtick': dtick,
                'xaxis.tickformat': tickformat
            }});
            updateMetrics(startDate, endDate);
        }}

        function selectAllStocks() {{
            // Only select visible stocks
            tickerNames.forEach(tk => {{
                const el = document.getElementById('cb-tk-' + tk.replace('.', '_'));
                if (el.parentElement.style.display !== 'none') {{
                    el.checked = true;
                }}
            }});
            onStockChange();
        }}

        function clearAllStocks() {{
            tickerNames.forEach(tk => document.getElementById('cb-tk-' + tk.replace('.', '_')).checked = false);
            onStockChange();
        }}

        function getSelectedCurrency() {{
            return document.getElementById('currency-select').value;
        }}

        function getExchangeRate(index, currency) {{
            // Returns the exchange rate at a given date index (USD to target currency)
            if (currency === 'KRW' && exchangeRates.KRW && exchangeRates.KRW[index]) {{
                return exchangeRates.KRW[index];
            }}
            if (currency === 'CAD' && exchangeRates.CAD && exchangeRates.CAD[index]) {{
                return exchangeRates.CAD[index];
            }}
            // Default fallbacks
            if (currency === 'KRW') return 1300.0;
            if (currency === 'CAD') return 1.36;
            return 1.0;
        }}

        function convertValue(valueUSD, index) {{
            const currency = getSelectedCurrency();
            if (currency === 'KRW') {{
                return valueUSD * getExchangeRate(index, 'KRW');
            }}
            if (currency === 'CAD') {{
                return valueUSD * getExchangeRate(index, 'CAD');
            }}
            return valueUSD;
        }}

        function aggregateData(selectedPortfolios, selectedStocks) {{
            const n = dates.length;
            const result = {{
                portfolio_value: new Array(n).fill(0),
                cost_basis: new Array(n).fill(0),
                unrealized_gain: new Array(n).fill(0),
                realized_gain: new Array(n).fill(0),
                dividends: new Array(n).fill(0),
                total_gain: new Array(n).fill(0),
                total_invested: new Array(n).fill(0)
            }};

            // Aggregate by selected portfolios AND selected stocks
            for (const pf of selectedPortfolios) {{
                for (const tk of selectedStocks) {{
                    const data = portfolioStockData[pf][tk];
                    for (let i = 0; i < n; i++) {{
                        result.portfolio_value[i] += data.portfolio_value[i];
                        result.cost_basis[i] += data.cost_basis[i];
                        result.unrealized_gain[i] += data.unrealized_gain[i];
                        result.realized_gain[i] += data.realized_gain[i];
                        result.dividends[i] += data.dividends[i];
                        result.total_gain[i] += data.total_gain[i];
                        result.total_invested[i] += data.total_invested[i];
                    }}
                }}
            }}
            return result;
        }}

        function formatCurrency(value, forceUSD = false) {{
            const currency = forceUSD ? 'USD' : getSelectedCurrency();
            const absVal = Math.abs(value);
            const sign = value < 0 ? '-' : '';

            if (currency === 'KRW') {{
                // Korean Won formatting
                if (absVal >= 100000000) return sign + '₩' + (absVal / 100000000).toFixed(1) + '억';
                if (absVal >= 10000) return sign + '₩' + (absVal / 10000).toFixed(1) + '만';
                return sign + '₩' + Math.round(absVal).toLocaleString('ko-KR');
            }} else if (currency === 'CAD') {{
                // Canadian Dollar formatting
                if (absVal >= 1000000) return sign + 'C$' + (absVal / 1000000).toFixed(1) + 'M';
                if (absVal >= 1000) return sign + 'C$' + (absVal / 1000).toFixed(1) + 'K';
                return sign + 'C$' + absVal.toFixed(0);
            }} else {{
                // USD formatting
                if (absVal >= 1000000) return sign + '$' + (absVal / 1000000).toFixed(1) + 'M';
                if (absVal >= 1000) return sign + '$' + (absVal / 1000).toFixed(1) + 'K';
                return sign + '$' + absVal.toFixed(0);
            }}
        }}

        function getCurrencySymbol() {{
            const currency = getSelectedCurrency();
            if (currency === 'KRW') return '₩';
            if (currency === 'CAD') return 'C$';
            return '$';
        }}

        function updateChart() {{
            const selectedPf = getSelectedPortfolios();
            const selectedTk = getSelectedStocks();
            const currency = getSelectedCurrency();
            const currencySymbol = getCurrencySymbol();

            if (selectedPf.length === 0 || selectedTk.length === 0) {{
                Plotly.react('portfolio-chart', [
                    {{ x: [], y: [], name: 'Cost Basis', line: {{ color: '#E94F37', width: 2.5, dash: 'dash' }} }},
                    {{ x: [], y: [], name: 'Portfolio Value', line: {{ color: '#2E86AB', width: 2.5 }}, fill: 'tonexty', fillcolor: 'rgba(46, 134, 171, 0.2)' }}
                ]);
                updateMetricsDisplay(null);
                return;
            }}

            const agg = aggregateData(selectedPf, selectedTk);
            const parsedDates = dates.map(d => d);

            // Convert values based on selected currency
            const convertedPortfolioValue = agg.portfolio_value.map((v, i) => convertValue(v, i));
            const convertedCostBasis = agg.cost_basis.map((v, i) => convertValue(v, i));

            // Get start index for period calculations
            const startDate = document.getElementById('start-date').value;
            const periodStartIdx = dates.findIndex(d => d >= startDate);

            // Calculate base values at period start for period return calculations
            // Period return measures gains from period start to each point
            const periodBaseIdx = periodStartIdx > 0 ? periodStartIdx - 1 : 0;
            const periodBaseGain = periodStartIdx > 0 ? agg.total_gain[periodBaseIdx] : 0;

            // Build custom hover text with all metrics
            const hoverTextPortfolio = [];
            for (let i = 0; i < dates.length; i++) {{
                const pv = convertValue(agg.portfolio_value[i], i);
                const cb = convertValue(agg.cost_basis[i], i);
                const ug = convertValue(agg.unrealized_gain[i], i);
                const rg = convertValue(agg.realized_gain[i], i);
                const dv = convertValue(agg.dividends[i], i);
                const tr = convertValue(agg.total_gain[i], i);
                const totalInvested = agg.total_invested[i];
                // Total return % = cumulative return from beginning to this point
                const totalReturnPct = totalInvested > 0 ? (agg.total_gain[i] / totalInvested * 100) : 0;
                const totalPctSign = totalReturnPct >= 0 ? '+' : '';

                // Period return: gains from period start date to this hover point
                const periodReturnUSD = agg.total_gain[i] - periodBaseGain;
                const periodReturn = convertValue(periodReturnUSD, i);
                // Period return % uses same denominator as total return (total invested)
                const periodReturnPct = totalInvested > 0 ? (periodReturnUSD / totalInvested * 100) : 0;
                const periodPctSign = periodReturnPct >= 0 ? '+' : '';

                const locale = currency === 'KRW' ? 'ko-KR' : (currency === 'CAD' ? 'en-CA' : 'en-US');
                const decimals = currency === 'KRW' ? 0 : 2;

                // Format date for display
                const dateObj = new Date(dates[i]);
                const dateStr = dateObj.toLocaleDateString(locale, {{ year: 'numeric', month: 'short', day: 'numeric' }});

                hoverTextPortfolio.push(
                    '<b>📅 ' + dateStr + '</b><br>' +
                    '<b>Portfolio Value:</b> ' + currencySymbol + pv.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + '<br>' +
                    '<b>Cost Basis:</b> ' + currencySymbol + cb.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + '<br>' +
                    '<b>Unrealized:</b> ' + currencySymbol + ug.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + '<br>' +
                    '<b>Realized:</b> ' + currencySymbol + rg.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + '<br>' +
                    '<b>Dividends:</b> ' + currencySymbol + dv.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + '<br>' +
                    '<b>Total Return:</b> ' + currencySymbol + tr.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + ' (' + totalPctSign + totalReturnPct.toFixed(2) + '%)' + '<br>' +
                    '<b>Period Return:</b> ' + currencySymbol + periodReturn.toLocaleString(locale, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + ' (' + periodPctSign + periodReturnPct.toFixed(2) + '%)'
                );
            }}

            Plotly.react('portfolio-chart', [
                {{
                    x: parsedDates,
                    y: convertedCostBasis,
                    name: 'Cost Basis',
                    line: {{ color: '#E94F37', width: 2.5, dash: 'dash' }},
                    hoverinfo: 'skip'
                }},
                {{
                    x: parsedDates,
                    y: convertedPortfolioValue,
                    name: 'Portfolio Value',
                    line: {{ color: '#2E86AB', width: 2.5 }},
                    fill: 'tonexty',
                    fillcolor: 'rgba(46, 134, 171, 0.2)',
                    text: hoverTextPortfolio,
                    hoverinfo: 'text',
                    hoverlabel: {{ bgcolor: 'white', font: {{ size: 12 }} }}
                }}
            ], {{
                xaxis: {{ title: '', showgrid: true, gridcolor: 'rgba(0,0,0,0.1)', tickangle: 0, automargin: true }},
                yaxis: {{ title: 'Value (' + currencySymbol + ')', showgrid: true, gridcolor: 'rgba(0,0,0,0.1)', tickprefix: currencySymbol, tickformat: ',.0f' }},
                legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'center', x: 0.5, bgcolor: 'rgba(255,255,255,0.8)' }},
                hovermode: 'x unified',
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: {{ l: 70, r: 30, t: 40, b: 50 }}
            }});

            const endDate = document.getElementById('end-date').value;
            updateMetrics(startDate, endDate);
        }}

        function updateMetrics(startDate, endDate) {{
            const selectedPf = getSelectedPortfolios();
            const selectedTk = getSelectedStocks();
            if (selectedPf.length === 0 || selectedTk.length === 0) {{
                updateMetricsDisplay(null);
                return;
            }}

            const agg = aggregateData(selectedPf, selectedTk);
            const startIdx = dates.findIndex(d => d >= startDate);
            const endIdx = dates.findLastIndex(d => d <= endDate);

            if (startIdx === -1 || endIdx === -1 || startIdx > endIdx) {{
                updateMetricsDisplay(null);
                return;
            }}

            // Get total invested at end date for return % calculation
            const totalInvestedEnd = agg.total_invested[endIdx];

            // Use day BEFORE period start so gains ON the start date are included
            const baseRealizedGain = startIdx > 0 ? agg.realized_gain[startIdx - 1] : 0;
            const baseDividends = startIdx > 0 ? agg.dividends[startIdx - 1] : 0;

            // Convert values based on selected currency using end date exchange rate
            const metrics = {{
                portfolio: convertValue(agg.portfolio_value[endIdx], endIdx),
                cost: convertValue(agg.cost_basis[endIdx], endIdx),
                unrealized: convertValue(agg.unrealized_gain[endIdx], endIdx),
                periodRealized: convertValue(agg.realized_gain[endIdx] - baseRealizedGain, endIdx),
                periodDividends: convertValue(agg.dividends[endIdx] - baseDividends, endIdx),
                totalInvested: totalInvestedEnd
            }};

            // Total cumulative return at end date (unrealized + realized + dividends)
            const totalReturnEnd = agg.total_gain[endIdx];
            metrics.totalReturn = convertValue(totalReturnEnd, endIdx);
            // Total return % based on total invested
            metrics.totalReturnPct = totalInvestedEnd > 0 ? (totalReturnEnd / totalInvestedEnd * 100) : 0;

            // Period return: change from period start to end
            // Use day BEFORE period start so gains ON the start date are included
            const baseGain = startIdx > 0 ? agg.total_gain[startIdx - 1] : 0;
            const periodReturnUSD = agg.total_gain[endIdx] - baseGain;
            metrics.periodTotal = convertValue(periodReturnUSD, endIdx);
            // Period return % uses same denominator as total return (total invested)
            metrics.periodReturnPct = totalInvestedEnd > 0 ? (periodReturnUSD / totalInvestedEnd * 100) : 0;

            updateMetricsDisplay(metrics);
        }}

        function updateMetricsDisplay(metrics) {{
            if (!metrics) {{
                document.getElementById('metric-portfolio').textContent = '-';
                document.getElementById('metric-cost').textContent = '-';
                document.getElementById('metric-unrealized').textContent = '-';
                document.getElementById('metric-realized').textContent = '-';
                document.getElementById('metric-dividends').textContent = '-';
                document.getElementById('metric-total').textContent = '-';
                document.getElementById('metric-total-pct').textContent = '-';
                document.getElementById('metric-period').textContent = '-';
                document.getElementById('metric-period-pct').textContent = '-';
                return;
            }}

            document.getElementById('metric-portfolio').textContent = formatCurrency(metrics.portfolio);
            document.getElementById('metric-cost').textContent = formatCurrency(metrics.cost);
            document.getElementById('metric-unrealized').textContent = (metrics.unrealized >= 0 ? '+' : '') + formatCurrency(metrics.unrealized);
            document.getElementById('metric-realized').textContent = (metrics.periodRealized >= 0 ? '+' : '') + formatCurrency(metrics.periodRealized);
            document.getElementById('metric-dividends').textContent = '+' + formatCurrency(metrics.periodDividends);

            // Total cumulative return
            document.getElementById('metric-total').textContent = (metrics.totalReturn >= 0 ? '+' : '') + formatCurrency(metrics.totalReturn);
            const totalPctEl = document.getElementById('metric-total-pct');
            totalPctEl.textContent = (metrics.totalReturnPct >= 0 ? '+' : '') + metrics.totalReturnPct.toFixed(1) + '%';
            totalPctEl.className = 'metric-value ' + (metrics.totalReturnPct >= 0 ? 'positive' : 'negative');

            // Period return
            document.getElementById('metric-period').textContent = (metrics.periodTotal >= 0 ? '+' : '') + formatCurrency(metrics.periodTotal);
            const periodPctEl = document.getElementById('metric-period-pct');
            periodPctEl.textContent = (metrics.periodReturnPct >= 0 ? '+' : '') + metrics.periodReturnPct.toFixed(1) + '%';
            periodPctEl.className = 'metric-value ' + (metrics.periodReturnPct >= 0 ? 'positive' : 'negative');
        }}

        function getSmartTickFormat(startDate, endDate, tickMode) {{
            const start = new Date(startDate);
            const end = new Date(endDate);
            const daysDiff = (end - start) / (1000 * 60 * 60 * 24);
            const sameYear = start.getFullYear() === end.getFullYear();
            const sameMonth = sameYear && start.getMonth() === end.getMonth();

            let dtick, tickformat;

            if (tickMode === 'auto') {{
                if (daysDiff <= 14) {{ dtick = 'D1'; tickformat = sameMonth ? '%d' : '%d<br>%b'; }}
                else if (daysDiff <= 60) {{ dtick = 604800000; tickformat = sameYear ? '%b %d' : '%d<br>%b %Y'; }}
                else if (daysDiff <= 400) {{ dtick = 'M1'; tickformat = '%b<br>%Y'; }}
                else if (daysDiff <= 800) {{ dtick = 'M3'; tickformat = '%b<br>%Y'; }}
                else {{ dtick = 'M12'; tickformat = '%Y'; }}
            }} else {{
                const formats = {{
                    'D1': {{ dtick: 'D1', tickformat: sameMonth ? '%d' : '%d<br>%b' }},
                    'D7': {{ dtick: 604800000, tickformat: sameYear ? '%b %d' : '%d<br>%b %Y' }},
                    'M1': {{ dtick: 'M1', tickformat: '%b<br>%Y' }},
                    'M3': {{ dtick: 'M3', tickformat: '%b<br>%Y' }},
                    'M12': {{ dtick: 'M12', tickformat: '%Y' }}
                }};
                const f = formats[tickMode] || formats['M1'];
                dtick = f.dtick;
                tickformat = f.tickformat;
            }}

            return {{ dtick, tickformat }};
        }}

        function applyDateRange() {{
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const tickMode = document.getElementById('tick-format').value;
            const {{ dtick, tickformat }} = getSmartTickFormat(startDate, endDate, tickMode);

            // Rebuild chart with new period start (this rebuilds hover text)
            updateChart();

            Plotly.relayout('portfolio-chart', {{
                'xaxis.range': [startDate, endDate],
                'xaxis.dtick': dtick,
                'xaxis.tickformat': tickformat
            }});

            updateMetrics(startDate, endDate);
        }}

        function resetDateRange() {{
            document.getElementById('start-date').value = minDate;
            document.getElementById('end-date').value = maxDate;
            document.getElementById('tick-format').value = 'auto';
            portfolioNames.forEach(pf => document.getElementById('cb-pf-' + pf).checked = true);
            tickerNames.forEach(tk => document.getElementById('cb-tk-' + tk.replace('.', '_')).checked = true);

            updateChart();

            const {{ dtick, tickformat }} = getSmartTickFormat(minDate, maxDate, 'auto');
            Plotly.relayout('portfolio-chart', {{
                'xaxis.range': [minDate, maxDate],
                'xaxis.dtick': dtick,
                'xaxis.tickformat': tickformat
            }});
        }}

        document.getElementById('portfolio-chart').on('plotly_relayout', function(eventData) {{
            let startDate, endDate;
            if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {{
                startDate = eventData['xaxis.range[0]'].split(' ')[0];
                endDate = eventData['xaxis.range[1]'].split(' ')[0];
            }} else if (eventData['xaxis.range']) {{
                startDate = eventData['xaxis.range'][0].split(' ')[0];
                endDate = eventData['xaxis.range'][1].split(' ')[0];
            }} else if (eventData['xaxis.autorange']) {{
                startDate = minDate;
                endDate = maxDate;
            }}
            if (startDate && endDate) {{
                updateMetrics(startDate, endDate);
                const tickMode = document.getElementById('tick-format').value;
                const {{ dtick, tickformat }} = getSmartTickFormat(startDate, endDate, tickMode);
                Plotly.relayout('portfolio-chart', {{ 'xaxis.dtick': dtick, 'xaxis.tickformat': tickformat }});
            }}
        }});

        function resizeChart() {{
            const container = document.querySelector('.chart-container');
            Plotly.relayout('portfolio-chart', {{ width: container.clientWidth, height: container.clientHeight }});
        }}

        window.addEventListener('resize', resizeChart);

        setTimeout(() => {{
            updateStockVisibility();
            updatePeriodStartDate();
            updateChart();
            resizeChart();
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const {{ dtick, tickformat }} = getSmartTickFormat(startDate, endDate, 'auto');
            Plotly.relayout('portfolio-chart', {{
                'xaxis.range': [startDate, endDate],
                'xaxis.dtick': dtick,
                'xaxis.tickformat': tickformat
            }});
            updateMetrics(startDate, endDate);
        }}, 100);
    </script>
</body>
</html>'''

        with open(output_path, 'w') as f:
            f.write(full_html)
        print(f"Chart saved to: {output_path}")

    return fig


def print_summary(history_by_portfolio: dict, portfolios: list):
    """Print portfolio summary statistics with gains breakdown."""
    print("\n" + "=" * 65)
    print("PORTFOLIO SUMMARY")
    print("=" * 65)

    totals = {'value': 0, 'cost': 0, 'invested': 0, 'unrealized': 0, 'realized': 0, 'dividends': 0, 'total': 0}

    for pf in portfolios:
        df = history_by_portfolio[pf]
        # Get the last row as a Series, then extract values
        # Use .values[-1] to always get a scalar even with duplicate indices
        portfolio_value = float(df['portfolio_value'].values[-1])
        cost_basis = float(df['cost_basis'].values[-1])
        unrealized_gain = float(df['unrealized_gain'].values[-1])
        realized_gain = float(df['realized_gain'].values[-1])
        dividends = float(df['dividends'].values[-1])
        total_gain = float(df['total_gain'].values[-1])
        total_invested = float(df['total_invested'].values[-1])

        print(f"\n[{pf}]")
        print(f"  Portfolio Value:    ${portfolio_value:>10,.2f}")
        print(f"  Cost Basis:         ${cost_basis:>10,.2f}")
        print(f"  Unrealized:         ${unrealized_gain:>10,.2f}")
        print(f"  Realized:           ${realized_gain:>10,.2f}")
        print(f"  Dividends:          ${dividends:>10,.2f}")
        print(f"  Total Gain:         ${total_gain:>10,.2f}")

        totals['value'] += portfolio_value
        totals['cost'] += cost_basis
        totals['invested'] += total_invested
        totals['unrealized'] += unrealized_gain
        totals['realized'] += realized_gain
        totals['dividends'] += dividends
        totals['total'] += total_gain

    return_pct = (totals['total'] / totals['invested'] * 100) if totals['invested'] > 0 else 0

    print("\n" + "-" * 65)
    print("COMBINED TOTALS:")
    print(f"  Portfolio Value:    ${totals['value']:>10,.2f}")
    print(f"  Cost Basis:         ${totals['cost']:>10,.2f}")
    print(f"  Unrealized:         ${totals['unrealized']:>10,.2f}")
    print(f"  Realized:           ${totals['realized']:>10,.2f}")
    print(f"  Dividends:          ${totals['dividends']:>10,.2f}")
    print(f"  TOTAL RETURN:       ${totals['total']:>10,.2f} ({return_pct:+.2f}%)")
    print("=" * 65)


if __name__ == '__main__':
    print("Loading transactions...")
    transactions = load_transactions(CSV_PATH)
    print(f"Loaded {len(transactions)} transactions")
    print(transactions.to_string(index=False))

    print("\nCalculating portfolio history...")
    history_by_portfolio, history_by_portfolio_stock, portfolios, tickers, exchange_rate_data = calculate_portfolio_history(transactions)

    # Fetch company names for display
    print()
    display_names = get_company_names(tickers)

    print_summary(history_by_portfolio, portfolios)

    print("\nGenerating interactive chart...")
    create_interactive_chart(history_by_portfolio, history_by_portfolio_stock, portfolios, tickers,
                            display_names, exchange_rate_data, OUTPUT_HTML)

    print(f"\nOpen '{OUTPUT_HTML}' in your browser to view the interactive chart.")