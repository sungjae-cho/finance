#!/usr/bin/env python3
"""
Portfolio Table Generator
Generates an interactive HTML page showing portfolio holdings with sorting and filtering.
Stocks appearing in multiple portfolios are combined into a single row.
Supports period-based performance analysis with date range and year selection.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

from parameters import CSV_PATH


def load_transactions(filepath: str) -> pd.DataFrame:
    """Load and parse transactions from CSV file."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_exchange_rates() -> dict:
    """Fetch current exchange rates for USD, KRW, CAD."""
    rates = {'USD': 1.0}

    # Fetch USD/KRW rate
    try:
        krw_ticker = yf.Ticker("KRW=X")
        krw_data = krw_ticker.history(period="1d")
        if not krw_data.empty:
            rates['KRW'] = float(krw_data['Close'].iloc[-1])
        else:
            rates['KRW'] = 1400.0  # Fallback
    except Exception:
        rates['KRW'] = 1400.0

    # Fetch USD/CAD rate
    try:
        cad_ticker = yf.Ticker("CAD=X")
        cad_data = cad_ticker.history(period="1d")
        if not cad_data.empty:
            rates['CAD'] = float(cad_data['Close'].iloc[-1])
        else:
            rates['CAD'] = 1.35  # Fallback
    except Exception:
        rates['CAD'] = 1.35

    return rates


def fetch_historical_exchange_rates(start_date: str) -> dict:
    """Fetch historical exchange rates from start_date to today."""
    historical_rates = {'USD': {}}  # USD is always 1.0
    
    currencies = ['KRW', 'CAD']
    
    for currency in currencies:
        try:
            symbol = f"{currency}=X"  # e.g., KRW=X gives USD to KRW rate
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date)
            
            if not hist.empty:
                rate_dict = {}
                for date, row in hist.iterrows():
                    if hasattr(date, 'tz') and date.tz is not None:
                        date_str = date.tz_convert('UTC').strftime('%Y-%m-%d')
                    else:
                        date_str = date.strftime('%Y-%m-%d')
                    rate_dict[date_str] = float(row['Close'])
                historical_rates[currency] = rate_dict
                print(f"  {currency}: {len(rate_dict)} days of exchange rate data")
            else:
                historical_rates[currency] = {}
        except Exception as e:
            print(f"Error fetching historical exchange rates for {currency}: {e}")
            historical_rates[currency] = {}
    
    return historical_rates


def get_stock_currency(ticker: str) -> str:
    """Determine the native currency of a stock based on its ticker."""
    if ticker.endswith('.KS') or ticker.endswith('.KQ'):
        return 'KRW'
    elif ticker.endswith('.TO') or ticker.endswith('.V') or ticker.endswith('.NE'):
        return 'CAD'
    else:
        return 'USD'


def get_split_history(ticker: str) -> pd.Series:
    """Fetch stock split history for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        splits = stock.splits
        if splits is not None and not splits.empty:
            return splits
    except Exception as e:
        print(f"Error fetching splits for {ticker}: {e}")
    return pd.Series(dtype=float)


def get_cumulative_split_ratio(splits: pd.Series, from_date: pd.Timestamp) -> float:
    """
    Calculate cumulative split ratio from a given date to present.
    """
    if splits.empty:
        return 1.0

    if splits.index.tz is not None:
        split_index_naive = splits.index.tz_convert('UTC').tz_localize(None)
    else:
        split_index_naive = splits.index
    splits_naive = pd.Series(splits.values, index=split_index_naive)

    if isinstance(from_date, pd.Timestamp):
        if from_date.tz is not None:
            from_date_naive = from_date.tz_convert('UTC').tz_localize(None)
        else:
            from_date_naive = from_date
    else:
        from_date_naive = pd.Timestamp(from_date)

    future_splits = splits_naive[splits_naive.index > from_date_naive]

    if future_splits.empty:
        return 1.0

    cumulative_ratio = future_splits.prod()
    return float(cumulative_ratio)


def calculate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate current holdings from transactions, accounting for stock splits."""
    holdings = []

    unique_tickers = df['ticker'].unique()
    split_histories = {}

    print("Fetching stock split histories...")
    for ticker in unique_tickers:
        split_histories[ticker] = get_split_history(ticker)
        if not split_histories[ticker].empty:
            print(f"  {ticker}: {len(split_histories[ticker])} splits found")

    grouped = df.groupby(['portfolio', 'ticker'])

    for (portfolio, ticker), group in grouped:
        total_shares = 0
        total_cost = 0
        splits = split_histories.get(ticker, pd.Series(dtype=float))

        group = group.sort_values('date')

        for _, row in group.iterrows():
            shares = float(row['shares'])
            price = float(row['price'])
            txn_date = row['date']

            split_ratio = get_cumulative_split_ratio(splits, txn_date)

            if row['action'] == 'buy':
                adjusted_shares = shares * split_ratio
                total_shares += adjusted_shares
                total_cost += shares * price
            elif row['action'] == 'sell':
                if total_shares > 0:
                    adjusted_shares = shares * split_ratio
                    avg_cost = total_cost / total_shares
                    total_shares -= adjusted_shares
                    total_cost = total_shares * avg_cost if total_shares > 0 else 0

        if total_shares > 0.0001:
            avg_cost_per_share = total_cost / total_shares
            holdings.append({
                'portfolio': portfolio,
                'ticker': ticker,
                'shares': total_shares,
                'avg_cost': avg_cost_per_share,
                'total_cost': total_cost,
                'currency': get_stock_currency(ticker)
            })

    return pd.DataFrame(holdings)


def fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices for all tickers."""
    prices = {}
    names = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = float(hist['Close'].iloc[-1])
            else:
                info = stock.info
                prices[ticker] = info.get('regularMarketPrice', 0) or info.get('previousClose', 0)

            try:
                info = stock.info
                names[ticker] = info.get('shortName') or info.get('longName') or ticker
            except:
                names[ticker] = ticker
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            prices[ticker] = 0
            names[ticker] = ticker

    return prices, names


def fetch_historical_prices(tickers: list, start_date: str) -> dict:
    """Fetch historical prices for all tickers from start_date to today."""
    historical = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date)
            if not hist.empty:
                # Convert to dict with date strings as keys
                price_dict = {}
                for date, row in hist.iterrows():
                    # Handle timezone-aware datetimes
                    if hasattr(date, 'tz') and date.tz is not None:
                        date_str = date.tz_convert('UTC').strftime('%Y-%m-%d')
                    else:
                        date_str = date.strftime('%Y-%m-%d')
                    price_dict[date_str] = float(row['Close'])
                historical[ticker] = price_dict
            else:
                historical[ticker] = {}
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            historical[ticker] = {}

    return historical


def generate_html(transactions_data: list, ticker_info: dict, exchange_rates: dict, portfolios: list,
                  historical_prices: dict, historical_fx_rates: dict, split_data: dict, 
                  available_years: list, min_date: str, max_date: str) -> str:
    """Generate the interactive HTML page."""

    transactions_json = json.dumps(transactions_data)
    ticker_info_json = json.dumps(ticker_info)
    rates_json = json.dumps(exchange_rates)
    portfolios_json = json.dumps(portfolios)
    historical_json = json.dumps(historical_prices)
    historical_fx_json = json.dumps(historical_fx_rates)
    splits_json = json.dumps(split_data)
    years_json = json.dumps(available_years)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Holdings</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            align-items: flex-end;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .control-group label {{
            font-weight: 600;
            color: #a0a0a0;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .portfolio-filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .portfolio-checkbox {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .portfolio-checkbox:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        .portfolio-checkbox input {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}

        .portfolio-checkbox span {{
            font-size: 0.95rem;
        }}

        .select-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }}

        .select-buttons button {{
            padding: 6px 12px;
            font-size: 0.8rem;
            background: rgba(58, 123, 213, 0.3);
            border: 1px solid rgba(58, 123, 213, 0.5);
            color: #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .select-buttons button:hover {{
            background: rgba(58, 123, 213, 0.5);
        }}

        select, input[type="date"] {{
            padding: 10px 15px;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #e0e0e0;
            cursor: pointer;
            min-width: 150px;
        }}

        input[type="date"] {{
            min-width: 160px;
        }}

        input[type="date"]::-webkit-calendar-picker-indicator {{
            filter: invert(0.8);
            cursor: pointer;
        }}

        select:focus, input[type="date"]:focus {{
            outline: none;
            border-color: #3a7bd5;
        }}

        .period-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: flex-end;
        }}

        .date-range {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .date-range span {{
            color: #808080;
        }}

        .period-presets {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .period-presets button {{
            padding: 8px 14px;
            font-size: 0.85rem;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .period-presets button:hover {{
            background: rgba(58, 123, 213, 0.3);
            border-color: rgba(58, 123, 213, 0.5);
        }}

        .period-presets button.active {{
            background: rgba(58, 123, 213, 0.5);
            border-color: #3a7bd5;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}

        .summary-card .label {{
            font-size: 0.85rem;
            color: #a0a0a0;
            margin-bottom: 8px;
        }}

        .summary-card .value {{
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .summary-card .value.positive {{
            color: #00c853;
        }}

        .summary-card .value.negative {{
            color: #ff5252;
        }}

        .summary-card .sub-value {{
            font-size: 0.85rem;
            color: #808080;
            margin-top: 4px;
        }}

        .table-container {{
            overflow-x: auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 5px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 15px 12px;
            text-align: left;
        }}

        th {{
            background: rgba(58, 123, 213, 0.3);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #c0c0c0;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            transition: background 0.2s;
            position: relative;
        }}

        th:hover {{
            background: rgba(58, 123, 213, 0.5);
        }}

        th.sortable::after {{
            content: '⇅';
            margin-left: 8px;
            opacity: 0.4;
            font-size: 0.75rem;
        }}

        th.sorted-asc::after {{
            content: '▲';
            opacity: 1;
            color: #00d2ff;
        }}

        th.sorted-desc::after {{
            content: '▼';
            opacity: 1;
            color: #00d2ff;
        }}

        th.sorted-asc, th.sorted-desc {{
            background: rgba(58, 123, 213, 0.5);
        }}

        tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: background 0.2s;
        }}

        tbody tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}

        td {{
            font-size: 0.95rem;
        }}

        .ticker {{
            font-weight: 600;
            color: #3a7bd5;
        }}

        .stock-name {{
            font-size: 0.8rem;
            color: #808080;
            margin-top: 2px;
        }}

        .portfolio-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}

        .portfolio-tag {{
            display: inline-block;
            padding: 4px 10px;
            background: rgba(58, 123, 213, 0.2);
            border-radius: 20px;
            font-size: 0.8rem;
            color: #7eb3ff;
        }}

        .positive {{
            color: #00c853;
        }}

        .negative {{
            color: #ff5252;
        }}

        .number {{
            text-align: right;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        }}

        .timestamp {{
            text-align: center;
            margin-top: 30px;
            font-size: 0.85rem;
            color: #606060;
        }}

        .sort-hint {{
            font-size: 0.8rem;
            color: #707070;
            margin-left: auto;
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }}

        .period-info {{
            font-size: 0.85rem;
            color: #a0a0a0;
            padding: 10px 15px;
            background: rgba(0, 210, 255, 0.1);
            border-radius: 8px;
            border-left: 3px solid #00d2ff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Holdings</h1>

        <div class="controls">
            <div class="control-group">
                <label>Filter by Portfolio</label>
                <div class="portfolio-filters" id="portfolioFilters"></div>
                <div class="select-buttons">
                    <button onclick="selectAll()">All</button>
                    <button onclick="selectNone()">None</button>
                </div>
            </div>

            <div class="control-group">
                <label>Currency</label>
                <select id="currencySelect" onchange="updateDisplay()">
                    <option value="USD">USD ($)</option>
                    <option value="KRW">KRW (₩)</option>
                    <option value="CAD">CAD (C$)</option>
                </select>
            </div>

            <div class="control-group">
                <label>Options</label>
                <label class="portfolio-checkbox">
                    <input type="checkbox" id="showFxBreakdown" onchange="updateDisplay()">
                    <span>Show Currency Effect</span>
                </label>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Performance Period</label>
                <div class="period-controls">
                    <div class="date-range">
                        <input type="date" id="startDate" onchange="onDateChange()">
                        <span>to</span>
                        <input type="date" id="endDate" onchange="onDateChange()">
                    </div>
                    <div class="period-presets" id="periodPresets">
                        <button onclick="setPeriod('ytd')" data-period="ytd">YTD</button>
                        <button onclick="setPeriod('1m')" data-period="1m">1M</button>
                        <button onclick="setPeriod('3m')" data-period="3m">3M</button>
                        <button onclick="setPeriod('6m')" data-period="6m">6M</button>
                        <button onclick="setPeriod('1y')" data-period="1y">1Y</button>
                        <button onclick="setPeriod('all')" data-period="all">All Time</button>
                    </div>
                </div>
            </div>

            <div class="control-group">
                <label>Quick Year Select</label>
                <select id="yearSelect" onchange="onYearChange()">
                    <option value="">-- Select Year --</option>
                </select>
            </div>

            <div class="sort-hint">
                Click column headers to sort
            </div>
        </div>

        <div class="summary" id="summary"></div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th class="sortable" data-field="ticker">Stock</th>
                        <th class="sortable" data-field="portfolios">Portfolios</th>
                        <th class="sortable number" data-field="shares">Shares</th>
                        <th class="sortable number" data-field="cost_basis_historical">Cost Basis</th>
                        <th class="sortable number" data-field="end_value">End Value</th>
                        <th class="sortable number" data-field="unrealized_return">Unrealized</th>
                        <th class="sortable number" data-field="annualized_return">Annualized</th>
                        <th class="sortable number" data-field="realized_return">Realized</th>
                    </tr>
                </thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>

        <div class="timestamp">
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <script>
        const transactionsData = {transactions_json};
        const tickerInfo = {ticker_info_json};
        const exchangeRates = {rates_json};
        const portfolios = {portfolios_json};
        const historicalPrices = {historical_json};
        const historicalFxRates = {historical_fx_json};
        const splitData = {splits_json};
        const availableYears = {years_json};
        const minDate = '{min_date}';
        const maxDate = '{max_date}';

        let selectedPortfolios = new Set(portfolios);
        let currentSort = {{ field: 'unrealized_return', ascending: false }};
        let currentPeriod = 'all';

        const currencySymbols = {{
            'USD': '$',
            'KRW': '₩',
            'CAD': 'C$'
        }};

        // Calculate cumulative split ratio from a given date to present
        function getFutureSplitRatio(ticker, fromDate) {{
            const splits = splitData[ticker];
            if (!splits || Object.keys(splits).length === 0) return 1.0;

            let ratio = 1.0;
            const splitDates = Object.keys(splits).sort();

            for (const splitDate of splitDates) {{
                if (splitDate > fromDate) {{
                    ratio *= splits[splitDate];
                }}
            }}
            return ratio;
        }}

        // Calculate holdings at a specific date for selected portfolios
        function calculateHoldingsAtDate(targetDate, selectedPfs) {{
            const holdings = {{}};  // ticker -> {{ shares, costBasis, costBasisUsd, portfolios }}

            // Filter and sort transactions up to targetDate
            const relevantTxns = transactionsData
                .filter(t => t.date <= targetDate && selectedPfs.has(t.portfolio))
                .sort((a, b) => a.date.localeCompare(b.date));

            // Process each transaction
            for (const txn of relevantTxns) {{
                const key = txn.ticker;

                if (!holdings[key]) {{
                    holdings[key] = {{
                        ticker: txn.ticker,
                        shares: 0,
                        costBasis: 0,      // In native currency
                        costBasisUsd: 0,   // In USD at purchase time FX rates
                        portfolios: new Set()
                    }};
                }}

                const h = holdings[key];
                const splitRatio = getFutureSplitRatio(txn.ticker, txn.date);
                const action = txn.action.toLowerCase();
                const txnCurrency = txn.currency || tickerInfo[txn.ticker]?.currency || 'USD';

                if (action === 'buy') {{
                    const adjustedShares = txn.shares * splitRatio;
                    const costNative = txn.shares * txn.price;
                    
                    // Convert to USD using FX rate at purchase date
                    const purchaseFxRate = getHistoricalFxRate(txnCurrency, txn.date, 'backward');
                    const costUsd = txnCurrency === 'USD' ? costNative : costNative / purchaseFxRate;
                    
                    h.shares += adjustedShares;
                    h.costBasis += costNative;
                    h.costBasisUsd += costUsd;
                    h.portfolios.add(txn.portfolio);
                }} else if (action === 'sell') {{
                    if (h.shares > 0) {{
                        const adjustedShares = txn.shares * splitRatio;
                        const sellRatio = adjustedShares / h.shares;
                        
                        h.shares -= adjustedShares;
                        h.costBasis *= (1 - sellRatio);
                        h.costBasisUsd *= (1 - sellRatio);
                        
                        if (h.shares < 0.0001) {{
                            h.costBasis = 0;
                            h.costBasisUsd = 0;
                        }}
                    }}
                }}
            }}

            // Clean up - remove holdings with zero shares
            for (const ticker in holdings) {{
                if (holdings[ticker].shares < 0.0001) {{
                    delete holdings[ticker];
                }} else {{
                    holdings[ticker].portfolios = Array.from(holdings[ticker].portfolios).sort();
                }}
            }}

            return holdings;
        }}

        // Get historical exchange rate (1 USD = X foreign currency)
        function getHistoricalFxRate(currency, targetDate, direction = 'backward') {{
            if (currency === 'USD') return 1.0;
            
            const rates = historicalFxRates[currency];
            if (!rates || Object.keys(rates).length === 0) {{
                // Fallback to current rate
                return exchangeRates[currency] || 1.0;
            }}
            
            const dates = Object.keys(rates).sort();
            
            if (rates[targetDate]) return rates[targetDate];
            
            if (direction === 'backward') {{
                for (let i = dates.length - 1; i >= 0; i--) {{
                    if (dates[i] <= targetDate) {{
                        return rates[dates[i]];
                    }}
                }}
                return rates[dates[0]];
            }} else {{
                for (let i = 0; i < dates.length; i++) {{
                    if (dates[i] >= targetDate) {{
                        return rates[dates[i]];
                    }}
                }}
                return rates[dates[dates.length - 1]];
            }}
        }}

        // Convert currency using historical rate
        function convertCurrencyHistorical(amount, fromCurrency, toCurrency, date) {{
            if (fromCurrency === toCurrency) return amount;
            
            const fromRate = getHistoricalFxRate(fromCurrency, date);
            const toRate = getHistoricalFxRate(toCurrency, date);
            
            // Convert to USD first, then to target
            const usdAmount = fromCurrency === 'USD' ? amount : amount / fromRate;
            return toCurrency === 'USD' ? usdAmount : usdAmount * toRate;
        }}

        // Calculate cash flows during a period for a specific ticker
        // Returns: {{ purchases, sales, soldCostBasis, purchasesNative, salesNative, soldCostBasisNative }} 
        // purchases/sales/soldCostBasis in USD, *Native in stock's native currency
        // Tracks cost basis per portfolio to handle multi-portfolio scenarios correctly
        function calculatePeriodCashFlows(ticker, startDate, endDate, selectedPfs) {{
            const info = tickerInfo[ticker] || {{ currency: 'USD' }};
            const nativeCurrency = info.currency;
            
            // Build per-portfolio holdings at start of period
            const portfolioHoldings = {{}};  // portfolio -> {{ shares, costBasisUsd, costBasisNative }}
            
            // Get all transactions up to startDate to build per-portfolio holdings
            const priorTxns = transactionsData
                .filter(t => t.ticker === ticker && 
                            t.date <= startDate && 
                            selectedPfs.has(t.portfolio))
                .sort((a, b) => {{
                    const dateCompare = a.date.localeCompare(b.date);
                    if (dateCompare !== 0) return dateCompare;
                    const aIsBuy = a.action.toLowerCase() === 'buy' ? 0 : 1;
                    const bIsBuy = b.action.toLowerCase() === 'buy' ? 0 : 1;
                    return aIsBuy - bIsBuy;
                }});
            
            for (const txn of priorTxns) {{
                const pf = txn.portfolio;
                if (!portfolioHoldings[pf]) {{
                    portfolioHoldings[pf] = {{ shares: 0, costBasisUsd: 0, costBasisNative: 0 }};
                }}
                
                const txnCurrency = txn.currency || nativeCurrency;
                const txnAmount = txn.shares * txn.price;  // In native currency
                const fxRate = getHistoricalFxRate(txnCurrency, txn.date, 'backward');
                const txnAmountUsd = txnCurrency === 'USD' ? txnAmount : txnAmount / fxRate;
                const action = txn.action.toLowerCase();
                const splitRatio = getFutureSplitRatio(ticker, txn.date);
                const adjustedShares = txn.shares * splitRatio;
                
                if (action === 'buy') {{
                    portfolioHoldings[pf].shares += adjustedShares;
                    portfolioHoldings[pf].costBasisUsd += txnAmountUsd;
                    portfolioHoldings[pf].costBasisNative += txnAmount;
                }} else if (action === 'sell') {{
                    const h = portfolioHoldings[pf];
                    if (h.shares > 0.0001) {{
                        const sellRatio = Math.min(adjustedShares / h.shares, 1);
                        h.shares -= adjustedShares;
                        h.costBasisUsd *= (1 - sellRatio);
                        h.costBasisNative *= (1 - sellRatio);
                        if (h.shares < 0.0001) {{
                            h.costBasisUsd = 0;
                            h.costBasisNative = 0;
                        }}
                    }}
                }}
            }}
            
            // Get transactions during the period
            const periodTxns = transactionsData
                .filter(t => t.ticker === ticker && 
                            t.date > startDate && 
                            t.date <= endDate && 
                            selectedPfs.has(t.portfolio))
                .sort((a, b) => {{
                    const dateCompare = a.date.localeCompare(b.date);
                    if (dateCompare !== 0) return dateCompare;
                    const aIsBuy = a.action.toLowerCase() === 'buy' ? 0 : 1;
                    const bIsBuy = b.action.toLowerCase() === 'buy' ? 0 : 1;
                    return aIsBuy - bIsBuy;
                }});
            
            let purchases = 0;           // Total spent on buys (in USD)
            let sales = 0;               // Total received from sells (in USD)
            let soldCostBasis = 0;       // Cost basis of shares sold (in USD)
            let purchasesNative = 0;     // Total spent on buys (in native currency)
            let salesNative = 0;         // Total received from sells (in native currency)
            let soldCostBasisNative = 0; // Cost basis of shares sold (in native currency)
            
            for (const txn of periodTxns) {{
                const pf = txn.portfolio;
                if (!portfolioHoldings[pf]) {{
                    portfolioHoldings[pf] = {{ shares: 0, costBasisUsd: 0, costBasisNative: 0 }};
                }}
                
                const txnCurrency = txn.currency || nativeCurrency;
                const txnAmount = txn.shares * txn.price;  // In native currency
                const fxRate = getHistoricalFxRate(txnCurrency, txn.date, 'backward');
                const txnAmountUsd = txnCurrency === 'USD' ? txnAmount : txnAmount / fxRate;
                const action = txn.action.toLowerCase();
                const splitRatio = getFutureSplitRatio(ticker, txn.date);
                const adjustedShares = txn.shares * splitRatio;
                
                if (action === 'buy') {{
                    purchases += txnAmountUsd;
                    purchasesNative += txnAmount;
                    portfolioHoldings[pf].shares += adjustedShares;
                    portfolioHoldings[pf].costBasisUsd += txnAmountUsd;
                    portfolioHoldings[pf].costBasisNative += txnAmount;
                }} else if (action === 'sell') {{
                    sales += txnAmountUsd;
                    salesNative += txnAmount;
                    
                    // Calculate cost basis of sold shares from this specific portfolio
                    const h = portfolioHoldings[pf];
                    if (h.shares > 0.0001) {{
                        const sellRatio = Math.min(adjustedShares / h.shares, 1);
                        const soldCostUsd = h.costBasisUsd * sellRatio;
                        const soldCostNative = h.costBasisNative * sellRatio;
                        soldCostBasis += soldCostUsd;
                        soldCostBasisNative += soldCostNative;
                        
                        h.shares -= adjustedShares;
                        h.costBasisUsd -= soldCostUsd;
                        h.costBasisNative -= soldCostNative;
                        
                        if (h.shares < 0.0001) {{
                            h.costBasisUsd = 0;
                            h.costBasisNative = 0;
                        }}
                    }}
                }}
            }}
            
            return {{ purchases, sales, soldCostBasis, purchasesNative, salesNative, soldCostBasisNative }};
        }}

        // Calculate cost basis and sales at historical (transaction-time) FX rates for any display currency
        // This properly tracks FX effects by using the FX rate at the time of each transaction
        function calculateHistoricalFxValues(ticker, endDate, selectedPfs, displayCurrency) {{
            const info = tickerInfo[ticker] || {{ currency: 'USD' }};
            const nativeCurrency = info.currency;
            
            // Get all transactions up to endDate
            const txns = transactionsData
                .filter(t => t.ticker === ticker && 
                            t.date <= endDate && 
                            selectedPfs.has(t.portfolio))
                .sort((a, b) => {{
                    const dateCompare = a.date.localeCompare(b.date);
                    if (dateCompare !== 0) return dateCompare;
                    const aIsBuy = a.action.toLowerCase() === 'buy' ? 0 : 1;
                    const bIsBuy = b.action.toLowerCase() === 'buy' ? 0 : 1;
                    return aIsBuy - bIsBuy;
                }});
            
            let shares = 0;
            let costBasisDisplay = 0;  // In display currency at historical (purchase-time) rates
            let soldCostBasisDisplay = 0;  // Cost basis of sold shares at historical (purchase-time) rates
            let salesDisplay = 0;  // Sales proceeds at historical (sale-time) rates
            
            for (const txn of txns) {{
                const txnCurrency = txn.currency || nativeCurrency;
                const amountNative = txn.shares * txn.price;
                const splitRatio = getFutureSplitRatio(ticker, txn.date);
                const adjustedShares = txn.shares * splitRatio;
                
                // Get FX rates at transaction time
                const nativeToUsdRate = getHistoricalFxRate(nativeCurrency, txn.date, 'backward');
                const displayFxRate = getHistoricalFxRate(displayCurrency, txn.date, 'backward');
                
                // Convert: Native → USD → Display (all at transaction-time rates)
                const amountUsd = nativeCurrency === 'USD' ? amountNative : amountNative / nativeToUsdRate;
                const amountDisplay = displayCurrency === 'USD' ? amountUsd : amountUsd * displayFxRate;
                
                const action = txn.action.toLowerCase();
                
                if (action === 'buy') {{
                    shares += adjustedShares;
                    costBasisDisplay += amountDisplay;
                }} else if (action === 'sell') {{
                    // Sales proceeds at sale-time FX rate
                    salesDisplay += amountDisplay;
                    
                    // Cost basis of sold shares at purchase-time FX rate
                    if (shares > 0.0001) {{
                        const sellRatio = Math.min(adjustedShares / shares, 1);
                        const soldCost = costBasisDisplay * sellRatio;
                        soldCostBasisDisplay += soldCost;
                        shares -= adjustedShares;
                        costBasisDisplay -= soldCost;
                        if (shares < 0.0001) {{
                            costBasisDisplay = 0;
                        }}
                    }}
                }}
            }}
            
            return {{ costBasisDisplay, soldCostBasisDisplay, salesDisplay }};
        }}

        function initFilters() {{
            const container = document.getElementById('portfolioFilters');
            portfolios.forEach(p => {{
                const label = document.createElement('label');
                label.className = 'portfolio-checkbox';
                label.innerHTML = `
                    <input type="checkbox" value="${{p}}" checked onchange="togglePortfolio('${{p}}')">
                    <span>${{p}}</span>
                `;
                container.appendChild(label);
            }});

            document.querySelectorAll('th.sortable').forEach(th => {{
                th.addEventListener('click', () => {{
                    const field = th.dataset.field;
                    sortBy(field);
                }});
            }});

            // Initialize year selector
            const yearSelect = document.getElementById('yearSelect');
            availableYears.forEach(year => {{
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearSelect.appendChild(option);
            }});

            // Set date range
            document.getElementById('startDate').value = minDate;
            document.getElementById('startDate').min = minDate;
            document.getElementById('startDate').max = maxDate;
            document.getElementById('endDate').value = maxDate;
            document.getElementById('endDate').min = minDate;
            document.getElementById('endDate').max = maxDate;

            // Default to All Time
            setPeriod('all');
        }}

        function selectAll() {{
            selectedPortfolios = new Set(portfolios);
            document.querySelectorAll('.portfolio-checkbox input').forEach(cb => cb.checked = true);
            updateStartDateForSelectedPortfolios();
            updateDisplay();
        }}

        function selectNone() {{
            selectedPortfolios.clear();
            document.querySelectorAll('.portfolio-checkbox input').forEach(cb => cb.checked = false);
            updateDisplay();
        }}

        function getEarliestPurchaseDate(selectedPfs) {{
            // Find the earliest purchase date among selected portfolios
            let earliest = null;
            for (const txn of transactionsData) {{
                if (txn.action.toLowerCase() === 'buy' && selectedPfs.has(txn.portfolio)) {{
                    if (earliest === null || txn.date < earliest) {{
                        earliest = txn.date;
                    }}
                }}
            }}
            return earliest || minDate;
        }}

        function updateStartDateForSelectedPortfolios() {{
            const earliestDate = getEarliestPurchaseDate(selectedPortfolios);
            document.getElementById('startDate').value = earliestDate;
            
            // Reset to 'all' period when portfolios change
            currentPeriod = 'all';
            document.getElementById('yearSelect').value = '';
            document.querySelectorAll('.period-presets button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.period === 'all');
            }});
        }}

        function togglePortfolio(portfolio) {{
            if (selectedPortfolios.has(portfolio)) {{
                selectedPortfolios.delete(portfolio);
            }} else {{
                selectedPortfolios.add(portfolio);
            }}
            updateStartDateForSelectedPortfolios();
            updateDisplay();
        }}

        function setPeriod(period) {{
            currentPeriod = period;
            const today = new Date(maxDate);
            let startDate;

            switch(period) {{
                case 'ytd':
                    startDate = new Date(today.getFullYear(), 0, 1);
                    break;
                case '1m':
                    startDate = new Date(today);
                    startDate.setMonth(startDate.getMonth() - 1);
                    break;
                case '3m':
                    startDate = new Date(today);
                    startDate.setMonth(startDate.getMonth() - 3);
                    break;
                case '6m':
                    startDate = new Date(today);
                    startDate.setMonth(startDate.getMonth() - 6);
                    break;
                case '1y':
                    startDate = new Date(today);
                    startDate.setFullYear(startDate.getFullYear() - 1);
                    break;
                case 'all':
                    // Use earliest purchase date from selected portfolios
                    const earliestDate = getEarliestPurchaseDate(selectedPortfolios);
                    startDate = new Date(earliestDate);
                    break;
            }}

            // Ensure start date is not before min date
            const minDateObj = new Date(minDate);
            if (startDate < minDateObj) {{
                startDate = minDateObj;
            }}

            document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
            document.getElementById('endDate').value = maxDate;
            document.getElementById('yearSelect').value = '';

            // Update active button
            document.querySelectorAll('.period-presets button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.period === period);
            }});

            updateDisplay();
        }}

        function onYearChange() {{
            const year = document.getElementById('yearSelect').value;
            if (year) {{
                currentPeriod = 'year';
                const startDate = `${{year}}-01-01`;
                const endDate = `${{year}}-12-31`;

                // Clamp to available data range
                const minDateObj = new Date(minDate);
                const maxDateObj = new Date(maxDate);
                let start = new Date(startDate);
                let end = new Date(endDate);

                if (start < minDateObj) start = minDateObj;
                if (end > maxDateObj) end = maxDateObj;

                document.getElementById('startDate').value = start.toISOString().split('T')[0];
                document.getElementById('endDate').value = end.toISOString().split('T')[0];

                // Clear active preset buttons
                document.querySelectorAll('.period-presets button').forEach(btn => {{
                    btn.classList.remove('active');
                }});

                updateDisplay();
            }}
        }}

        function onDateChange() {{
            currentPeriod = 'custom';
            document.getElementById('yearSelect').value = '';
            document.querySelectorAll('.period-presets button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            updateDisplay();
        }}

        function getClosestPrice(ticker, targetDate, direction = 'backward') {{
            const prices = historicalPrices[ticker];
            if (!prices || Object.keys(prices).length === 0) return null;

            const dates = Object.keys(prices).sort();
            const target = targetDate;

            if (prices[target]) return prices[target];

            if (direction === 'backward') {{
                // Find closest date before or on target
                for (let i = dates.length - 1; i >= 0; i--) {{
                    if (dates[i] <= target) {{
                        return prices[dates[i]];
                    }}
                }}
                // If no earlier date, use first available
                return prices[dates[0]];
            }} else {{
                // Find closest date after or on target
                for (let i = 0; i < dates.length; i++) {{
                    if (dates[i] >= target) {{
                        return prices[dates[i]];
                    }}
                }}
                // If no later date, use last available
                return prices[dates[dates.length - 1]];
            }}
        }}

        function convertCurrency(amount, fromCurrency, toCurrency) {{
            if (fromCurrency === toCurrency) return amount;

            let usdAmount = amount;
            if (fromCurrency === 'KRW') {{
                usdAmount = amount / exchangeRates.KRW;
            }} else if (fromCurrency === 'CAD') {{
                usdAmount = amount / exchangeRates.CAD;
            }}

            if (toCurrency === 'KRW') {{
                return usdAmount * exchangeRates.KRW;
            }} else if (toCurrency === 'CAD') {{
                return usdAmount * exchangeRates.CAD;
            }}
            return usdAmount;
        }}

        function formatNumber(num, currency, decimals = 2) {{
            if (num === null || num === undefined || isNaN(num)) return '-';

            const symbol = currencySymbols[currency];
            const absNum = Math.abs(num);

            if (currency === 'KRW') {{
                decimals = 0;
            }}

            const formatted = absNum.toLocaleString('en-US', {{
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            }});

            if (num < 0) {{
                return `-${{symbol}}${{formatted}}`;
            }}
            return `${{symbol}}${{formatted}}`;
        }}

        function formatSignedNumber(num, currency, decimals = 2) {{
            if (num === null || num === undefined || isNaN(num)) return '-';

            const symbol = currencySymbols[currency];
            const absNum = Math.abs(num);

            if (currency === 'KRW') {{
                decimals = 0;
            }}

            const formatted = absNum.toLocaleString('en-US', {{
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            }});

            if (num < 0) {{
                return `-${{symbol}}${{formatted}}`;
            }}
            return `+${{symbol}}${{formatted}}`;
        }}

        function formatPercent(num) {{
            if (num === null || num === undefined || isNaN(num)) return '-';
            const sign = num >= 0 ? '+' : '';
            return `${{sign}}${{num.toFixed(2)}}%`;
        }}

        function sortBy(field) {{
            if (currentSort.field === field) {{
                currentSort.ascending = !currentSort.ascending;
            }} else {{
                currentSort.field = field;
                currentSort.ascending = (field === 'ticker' || field === 'portfolios');
            }}
            updateDisplay();
        }}

        function updateSortIndicators() {{
            document.querySelectorAll('th.sortable').forEach(th => {{
                th.classList.remove('sorted-asc', 'sorted-desc');
            }});

            const currentTh = document.querySelector(`th[data-field="${{currentSort.field}}"]`);
            if (currentTh) {{
                currentTh.classList.add(currentSort.ascending ? 'sorted-asc' : 'sorted-desc');
            }}
        }}

        function updateDisplay() {{
            const currency = document.getElementById('currencySelect').value;
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            const isAllTime = currentPeriod === 'all';
            const showFxBreakdown = document.getElementById('showFxBreakdown').checked;

            // Calculate holdings at start and end dates
            const startHoldings = calculateHoldingsAtDate(startDate, selectedPortfolios);
            const endHoldings = calculateHoldingsAtDate(endDate, selectedPortfolios);

            // Get all tickers that had holdings at start, end, OR had transactions during the period
            const tickersWithActivity = new Set();
            Object.keys(startHoldings).forEach(t => tickersWithActivity.add(t));
            Object.keys(endHoldings).forEach(t => tickersWithActivity.add(t));
            
            // Also include tickers that had transactions during the period (for realized gains)
            transactionsData
                .filter(t => t.date > startDate && t.date <= endDate && selectedPortfolios.has(t.portfolio))
                .forEach(t => tickersWithActivity.add(t.ticker));

            // Build combined data
            let combined = [];

            for (const ticker of tickersWithActivity) {{
                const info = tickerInfo[ticker] || {{ name: ticker, currency: 'USD' }};
                const nativeCurrency = info.currency;
                const startH = startHoldings[ticker];
                const endH = endHoldings[ticker];

                // Get shares and cost basis at each point
                const startShares = startH ? startH.shares : 0;
                const endShares = endH ? endH.shares : 0;
                const startCostBasis = startH ? startH.costBasis : 0;
                const endCostBasis = endH ? endH.costBasis : 0;
                const startCostBasisUsd = startH ? startH.costBasisUsd : 0;
                const endCostBasisUsd = endH ? endH.costBasisUsd : 0;

                // Calculate cash flows during the period (purchases and sales in USD and native)
                const cashFlows = calculatePeriodCashFlows(ticker, startDate, endDate, selectedPortfolios);
                const purchasesUsd = cashFlows.purchases;
                const salesUsd = cashFlows.sales;
                const soldCostBasisUsd = cashFlows.soldCostBasis;
                const purchasesNative = cashFlows.purchasesNative;
                const salesNative = cashFlows.salesNative;
                const soldCostBasisNative = cashFlows.soldCostBasisNative;

                // Get prices in native currency
                const startPriceNative = getClosestPrice(ticker, startDate, 'forward') || 0;
                const endPriceNative = getClosestPrice(ticker, endDate, 'backward') || 0;

                // Calculate values in NATIVE currency
                const startValueNative = startShares * startPriceNative;
                const endValueNative = endShares * endPriceNative;

                // Get historical FX rates
                const startFxRate = getHistoricalFxRate(nativeCurrency, startDate, 'forward');
                const endFxRate = getHistoricalFxRate(nativeCurrency, endDate, 'backward');

                // Calculate start and end values in USD
                let startValueUsd, endValueUsd;
                
                if (nativeCurrency === 'USD') {{
                    startValueUsd = startValueNative;
                    endValueUsd = endValueNative;
                }} else {{
                    startValueUsd = startShares > 0 ? startValueNative / startFxRate : 0;
                    endValueUsd = endShares > 0 ? endValueNative / endFxRate : 0;
                }}

                // PERIOD RETURN (in USD):
                // = (End Value + Sales) - (Start Value + Purchases)
                // = End Value - Start Value + Sales - Purchases
                // This captures: unrealized gains + realized gains from sales
                const periodReturnUsd = endValueUsd - startValueUsd + salesUsd - purchasesUsd;

                // For display purposes, determine effective start value
                const boughtDuringPeriod = startShares < 0.0001 && endShares > 0.0001;
                const soldDuringPeriod = startShares > 0.0001 && endShares < 0.0001;
                const hadActivity = purchasesUsd > 0 || salesUsd > 0;

                // Skip if no holdings and no activity
                if (startShares < 0.0001 && endShares < 0.0001 && !hadActivity) continue;

                // Calculate what we "put in" at start of period for return % calculation
                let effectiveStartValueUsd;
                if (isAllTime) {{
                    // All Time: use cost basis of what we invested
                    // For stocks bought and sold entirely in the period, use purchases as the base
                    if (endShares < 0.0001 && startShares < 0.0001) {{
                        // Bought and sold entirely within the period
                        effectiveStartValueUsd = purchasesUsd;
                    }} else if (endShares < 0.0001 && startShares > 0.0001) {{
                        // Held at start, sold during period - use start cost basis
                        effectiveStartValueUsd = startCostBasisUsd;
                    }} else {{
                        // Still holding - use end cost basis
                        effectiveStartValueUsd = endCostBasisUsd;
                    }}
                }} else {{
                    // Period: use start market value + purchases
                    effectiveStartValueUsd = startValueUsd + purchasesUsd;
                }}

                const periodReturnPctUsd = effectiveStartValueUsd > 0 ? (periodReturnUsd / effectiveStartValueUsd) * 100 : 0;

                // Convert to display currency
                const endToDisplay = currency === 'USD' ? 1 : getHistoricalFxRate(currency, endDate, 'backward');
                const startToDisplay = currency === 'USD' ? 1 : getHistoricalFxRate(currency, startDate, 'forward');

                let startValueConverted, endValueConverted, periodReturnConverted, effectiveStartConverted;
                
                if (currency === nativeCurrency) {{
                    // Display in native currency
                    startValueConverted = startValueNative;
                    endValueConverted = endValueNative;
                    // For native currency, recalculate period return
                    const purchasesNative = purchasesUsd * (nativeCurrency === 'USD' ? 1 : endFxRate);
                    const salesNative = salesUsd * (nativeCurrency === 'USD' ? 1 : endFxRate);
                    periodReturnConverted = endValueNative - startValueNative + salesNative - purchasesNative;
                    effectiveStartConverted = startValueNative + purchasesNative;
                }} else if (isAllTime || boughtDuringPeriod) {{
                    // Use end rate for both to match Accumulated Return
                    startValueConverted = effectiveStartValueUsd * endToDisplay;
                    endValueConverted = endValueUsd * endToDisplay;
                    periodReturnConverted = periodReturnUsd * endToDisplay;
                    effectiveStartConverted = effectiveStartValueUsd * endToDisplay;
                }} else {{
                    // Normal period - use historical rates
                    startValueConverted = startValueUsd * startToDisplay;
                    endValueConverted = endValueUsd * endToDisplay;
                    const salesConverted = salesUsd * endToDisplay;
                    const purchasesConverted = purchasesUsd * startToDisplay;
                    periodReturnConverted = endValueConverted - startValueConverted + salesConverted - purchasesConverted;
                    effectiveStartConverted = startValueConverted + purchasesConverted;
                }}

                const periodReturnPct = effectiveStartConverted > 0 ? (periodReturnConverted / effectiveStartConverted) * 100 : 0;

                // Display values
                let displayStartPrice, displayStartValue;
                if (isAllTime) {{
                    displayStartPrice = endShares > 0 ? convertCurrency(endCostBasis / endShares, nativeCurrency, currency) : 0;
                    displayStartValue = effectiveStartConverted;
                }} else if (boughtDuringPeriod) {{
                    displayStartPrice = endShares > 0 ? convertCurrency(endCostBasis / endShares, nativeCurrency, currency) : 0;
                    displayStartValue = effectiveStartConverted;
                }} else {{
                    displayStartPrice = convertCurrency(startPriceNative, nativeCurrency, currency);
                    displayStartValue = startValueConverted;
                }}
                
                const endPrice = convertCurrency(endPriceNative, nativeCurrency, currency);
                const totalCost = convertCurrency(endCostBasis, nativeCurrency, currency);
                const avgCost = endShares > 0 ? convertCurrency(endCostBasis / endShares, nativeCurrency, currency) : 0;

                // Cost basis in historical terms (uses purchase-time FX rates)
                // This is the cost basis converted to display currency at the FX rate when each purchase was made
                let costBasisHistorical, soldCostBasisHistorical, salesHistorical;
                if (currency === 'USD') {{
                    costBasisHistorical = endCostBasisUsd;
                    soldCostBasisHistorical = soldCostBasisUsd;
                    salesHistorical = salesUsd;
                }} else if (currency === nativeCurrency) {{
                    costBasisHistorical = endCostBasis;
                    soldCostBasisHistorical = soldCostBasisNative;
                    salesHistorical = salesNative;
                }} else {{
                    // Use the new function to get values at historical (transaction-time) FX rates
                    const historicalValues = calculateHistoricalFxValues(ticker, endDate, selectedPortfolios, currency);
                    costBasisHistorical = historicalValues.costBasisDisplay;
                    soldCostBasisHistorical = historicalValues.soldCostBasisDisplay;
                    salesHistorical = historicalValues.salesDisplay;
                }}

                // Cost basis at END rate (for Local Return - no FX effect)
                // This is what the cost basis would be if converted at today's rate
                let costBasisAtEndRate;
                if (currency === 'USD') {{
                    // USD display: convert native cost basis at end rate
                    costBasisAtEndRate = nativeCurrency === 'USD' ? endCostBasis : endCostBasis / endFxRate;
                }} else if (currency === nativeCurrency) {{
                    // Native display: no conversion needed, no FX effect
                    costBasisAtEndRate = endCostBasis;
                }} else {{
                    // Other currency display: convert native to USD at end rate, then to display
                    const costInUsdAtEndRate = nativeCurrency === 'USD' ? endCostBasis : endCostBasis / endFxRate;
                    costBasisAtEndRate = costInUsdAtEndRate * endToDisplay;
                }}

                // Calculate Realized and Unrealized returns
                // When display currency = native currency, use native values directly (no FX conversion)
                let salesConverted, soldCostBasisConverted, purchasesConverted;
                
                if (currency === nativeCurrency) {{
                    // Native display: use native values directly - NO FX effect
                    salesConverted = salesNative;
                    soldCostBasisConverted = soldCostBasisNative;
                    purchasesConverted = purchasesNative;
                }} else if (currency === 'USD') {{
                    // USD display: use USD values (already converted at transaction-time rates)
                    salesConverted = salesUsd;
                    soldCostBasisConverted = soldCostBasisUsd;
                    purchasesConverted = purchasesUsd;
                }} else {{
                    // Other currency display: use historical rates
                    // Sales at sale-time FX rate (locked in when sold)
                    salesConverted = salesHistorical;
                    // Cost basis at purchase-time FX rate
                    soldCostBasisConverted = soldCostBasisHistorical;
                    // Purchases at purchase-time FX rate (use the historical function result indirectly via cost basis)
                    purchasesConverted = purchasesUsd * endToDisplay;  // For period calculations, still need this
                }}
                
                // Sold cost basis at END rate (for Local Return)
                // Now using actual native currency value
                let soldCostBasisAtEndRate, salesAtEndRate;
                if (currency === 'USD') {{
                    // USD display: convert native sold cost basis at end rate
                    soldCostBasisAtEndRate = nativeCurrency === 'USD' ? soldCostBasisNative : soldCostBasisNative / endFxRate;
                    salesAtEndRate = nativeCurrency === 'USD' ? salesNative : salesNative / endFxRate;
                }} else if (currency === nativeCurrency) {{
                    // Native display: use native value directly
                    soldCostBasisAtEndRate = soldCostBasisNative;
                    salesAtEndRate = salesNative;
                }} else {{
                    // Other currency display: convert native to USD at end rate, then to display
                    const soldCostInUsdAtEndRate = nativeCurrency === 'USD' ? soldCostBasisNative : soldCostBasisNative / endFxRate;
                    soldCostBasisAtEndRate = soldCostInUsdAtEndRate * endToDisplay;
                    const salesInUsdAtEndRate = nativeCurrency === 'USD' ? salesNative : salesNative / endFxRate;
                    salesAtEndRate = salesInUsdAtEndRate * endToDisplay;
                }}
                
                // Total invested = cost basis of current holdings + cost basis of sold shares
                const totalInvestedForTicker = costBasisHistorical + soldCostBasisConverted;
                const totalInvestedAtEndRate = costBasisAtEndRate + soldCostBasisAtEndRate;
                
                // Realized Return = Sales proceeds (at sale-time rate) - Cost basis of sold shares (at purchase-time rate)
                const realizedReturn = salesConverted - soldCostBasisConverted;
                
                // Unrealized Return = Current value - Cost basis of current holdings
                const unrealizedReturn = endValueConverted - costBasisHistorical;
                
                // Total Return = Realized + Unrealized (accounting identity)
                const totalReturnCalc = realizedReturn + unrealizedReturn;
                const totalReturnPct = totalInvestedForTicker > 0 ? (totalReturnCalc / totalInvestedForTicker) * 100 : 0;

                // LOCAL RETURN: What the return would be if FX rates hadn't changed
                // Use END rate for all values to remove FX effect
                const localUnrealized = endValueConverted - costBasisAtEndRate;
                const localRealized = salesAtEndRate - soldCostBasisAtEndRate;
                const localReturnCalc = localUnrealized + localRealized;
                const localReturnPctCalc = totalInvestedAtEndRate > 0 ? (localReturnCalc / totalInvestedAtEndRate) * 100 : 0;

                // FX RETURN = Total Return - Local Return
                // This captures the effect of FX rate changes on cost basis
                // FX = 0 when display currency = native currency (no conversion)
                const fxReturnCalc = totalReturnCalc - localReturnCalc;
                const fxReturnPctCalc = totalInvestedForTicker > 0 ? (fxReturnCalc / totalInvestedForTicker) * 100 : 0;
                
                // =========== PERIOD-SPECIFIC FX CALCULATIONS ===========
                // Rules for FX rates:
                // 1. Bought during/after period start → use buying time FX (already in purchasesUsd)
                // 2. Bought before period start (existing holdings) → use period start FX
                // 3. Still holding at period end → use period end FX
                // 4. Sold during period → use selling time FX (already in salesUsd)
                // 5. Same currency → FX return = 0
                
                // Calculate period return in native currency (pure investment performance, no FX)
                const periodReturnNative = endValueNative - startValueNative + salesNative - purchasesNative;
                
                // Split into unrealized and realized components in native currency
                // Unrealized: End Value - (Start Value still held + Purchases)
                // Realized: Sales - Cost Basis of Sold
                
                // Calculate what portion of start holdings is still held
                const startSharesStillHeld = Math.min(startShares, endShares);
                const startSharesSold = Math.max(0, startShares - endShares);
                
                // Value of start holdings still held (in native currency at START price)
                const startValueStillHeldNative = startShares > 0 ? 
                    (startSharesStillHeld / startShares) * startValueNative : 0;
                
                // Period Unrealized (in native currency):
                // = End value - (start value still held + purchases)
                const periodUnrealizedNative = endValueNative - startValueStillHeldNative - purchasesNative;
                
                // Period Realized (in native currency):
                // = Sales - sold cost basis
                const periodRealizedNative = salesNative - soldCostBasisNative;
                
                let periodLocalReturn, periodFxReturn, actualPeriodReturnDisplay;
                let periodUnrealizedLocal, periodUnrealizedFx, periodUnrealizedActual;
                let periodRealizedLocal, periodRealizedFx, periodRealizedActual;
                
                if (currency === nativeCurrency) {{
                    // Rule 5: No FX effect when display currency matches native currency
                    periodLocalReturn = periodReturnNative;
                    periodFxReturn = 0;
                    actualPeriodReturnDisplay = periodReturnNative;
                    
                    periodUnrealizedLocal = periodUnrealizedNative;
                    periodUnrealizedFx = 0;
                    periodUnrealizedActual = periodUnrealizedNative;
                    
                    periodRealizedLocal = periodRealizedNative;
                    periodRealizedFx = 0;
                    periodRealizedActual = periodRealizedNative;
                }} else {{
                    // Calculate actual return with proper FX rates per rules
                    
                    // Native to USD rates
                    const nativeToUsdStart = nativeCurrency === 'USD' ? 1 : 1 / startFxRate;
                    const nativeToUsdEnd = nativeCurrency === 'USD' ? 1 : 1 / endFxRate;
                    
                    // USD to display currency rates
                    const usdToDisplayStart = currency === 'USD' ? 1 : startToDisplay;
                    const usdToDisplayEnd = currency === 'USD' ? 1 : endToDisplay;
                    
                    // Combined rates
                    const rateStart = nativeToUsdStart * usdToDisplayStart;
                    const rateEnd = nativeToUsdEnd * usdToDisplayEnd;
                    
                    // Start value: holdings from before period → use period START FX (Rule 2)
                    const startValueDisplay = startValueNative * rateStart;
                    const startValueStillHeldDisplay = startValueStillHeldNative * rateStart;
                    
                    // End value: still holding at period end → use period END FX (Rule 3)
                    const endValueDisplay = endValueNative * rateEnd;
                    
                    // Purchases during period: use buying time FX (Rule 1)
                    // purchasesUsd already converted at purchase time
                    const purchasesDisplay = purchasesUsd * usdToDisplayEnd;
                    
                    // Sales during period: use selling time FX (Rule 4)
                    // salesUsd already converted at sale time
                    const salesDisplay = salesUsd * usdToDisplayEnd;
                    
                    // Sold cost basis: use buying time FX
                    const soldCostBasisDisplay = soldCostBasisUsd * usdToDisplayEnd;
                    
                    // Actual period return with proper FX rates
                    actualPeriodReturnDisplay = endValueDisplay - startValueDisplay + salesDisplay - purchasesDisplay;
                    
                    // Actual unrealized: End value - start value still held - purchases
                    periodUnrealizedActual = endValueDisplay - startValueStillHeldDisplay - purchasesDisplay;
                    
                    // Actual realized: Sales - sold cost basis
                    periodRealizedActual = salesDisplay - soldCostBasisDisplay;
                    
                    // Local returns: native currency returns converted at START rate (no FX movement effect)
                    periodLocalReturn = periodReturnNative * rateStart;
                    periodUnrealizedLocal = periodUnrealizedNative * rateStart;
                    periodRealizedLocal = periodRealizedNative * rateStart;
                    
                    // FX returns = Actual - Local
                    periodFxReturn = actualPeriodReturnDisplay - periodLocalReturn;
                    periodUnrealizedFx = periodUnrealizedActual - periodUnrealizedLocal;
                    periodRealizedFx = periodRealizedActual - periodRealizedLocal;
                }}
                
                // Period base for percentage calculation
                // Start value (at start FX) + purchases (at purchase-time FX converted to display)
                let periodBaseDisplay;
                if (currency === nativeCurrency) {{
                    periodBaseDisplay = startValueNative + purchasesNative;
                }} else {{
                    const nativeToUsdStart = nativeCurrency === 'USD' ? 1 : 1 / startFxRate;
                    const usdToDisplayStart = currency === 'USD' ? 1 : startToDisplay;
                    const usdToDisplayEnd = currency === 'USD' ? 1 : endToDisplay;
                    
                    const startValueDisplay = startValueNative * nativeToUsdStart * usdToDisplayStart;
                    const purchasesDisplay = purchasesUsd * usdToDisplayEnd;
                    periodBaseDisplay = startValueDisplay + purchasesDisplay;
                }}
                
                const periodLocalReturnPct = periodBaseDisplay > 0 ? (periodLocalReturn / periodBaseDisplay) * 100 : 0;
                const periodFxReturnPct = periodBaseDisplay > 0 ? (periodFxReturn / periodBaseDisplay) * 100 : 0;

                // Get portfolio list - check transactions if no holdings data
                let portfolioList = endH ? endH.portfolios : (startH ? startH.portfolios : []);
                if (portfolioList.length === 0) {{
                    // Get portfolios from transactions during this period
                    const txnPortfolios = new Set();
                    transactionsData
                        .filter(t => t.ticker === ticker && selectedPortfolios.has(t.portfolio))
                        .forEach(t => txnPortfolios.add(t.portfolio));
                    portfolioList = Array.from(txnPortfolios).sort();
                }}

                // Calculate percentage returns for unrealized and realized
                // Unrealized % = (unrealized return / cost basis of current holdings) * 100
                const unrealizedPct = costBasisHistorical > 0 ? (unrealizedReturn / costBasisHistorical) * 100 : 0;
                
                // Realized % = (realized return / cost basis of sold shares) * 100
                const realizedPct = soldCostBasisConverted > 0 ? (realizedReturn / soldCostBasisConverted) * 100 : 0;
                
                // Calculate local and FX breakdown percentages (based on same cost basis as total)
                // Unrealized breakdown
                const unrealizedLocalPct = costBasisHistorical > 0 ? (localUnrealized / costBasisHistorical) * 100 : 0;
                const unrealizedFxPct = costBasisHistorical > 0 ? ((unrealizedReturn - localUnrealized) / costBasisHistorical) * 100 : 0;
                
                // Realized breakdown
                const realizedLocalPct = soldCostBasisConverted > 0 ? (localRealized / soldCostBasisConverted) * 100 : 0;
                const realizedFxPct = soldCostBasisConverted > 0 ? ((realizedReturn - localRealized) / soldCostBasisConverted) * 100 : 0;
                
                // Calculate annualized return for unrealized gains
                // Period rules:
                // Start: Use S_p (period start) if S_s < S_p; otherwise S_s (first purchase date)
                // End: Use E_p (period end) if E_p < E_s or E_s doesn't exist; otherwise E_s (last sale date when fully sold)
                const tickerTxns = transactionsData
                    .filter(t => t.ticker === ticker && selectedPortfolios.has(t.portfolio))
                    .sort((a, b) => a.date.localeCompare(b.date));
                
                const buyTxns = tickerTxns.filter(t => t.action.toLowerCase() === 'buy');
                const sellTxns = tickerTxns.filter(t => t.action.toLowerCase() === 'sell');
                
                let annualizedReturn = 0;
                if (buyTxns.length > 0 && (costBasisHistorical > 0 || soldCostBasisConverted > 0)) {{
                    // S_s: First purchase date
                    const firstPurchaseDate = new Date(buyTxns[0].date);
                    // S_p: Period start date
                    const periodStartDate = new Date(startDate);
                    
                    // Start: Use S_p if S_s < S_p; otherwise S_s
                    const effectiveStart = firstPurchaseDate < periodStartDate ? periodStartDate : firstPurchaseDate;
                    
                    // E_p: Period end date
                    const periodEndDate = new Date(endDate);
                    
                    // E_s: Last sale date when fully sold (only if shares = 0)
                    let effectiveEnd = periodEndDate;
                    if (endShares < 0.0001 && sellTxns.length > 0) {{
                        // Stock is fully sold, find E_s (last sale date)
                        const lastSaleDate = new Date(sellTxns[sellTxns.length - 1].date);
                        // End: Use E_p if E_p < E_s; otherwise E_s
                        effectiveEnd = periodEndDate < lastSaleDate ? periodEndDate : lastSaleDate;
                    }}
                    
                    const holdingDays = Math.max(1, Math.round((effectiveEnd - effectiveStart) / (1000 * 60 * 60 * 24)));
                    
                    // Calculate value at effective end date
                    let endValueForAnnualized, costBasisForAnnualized;
                    
                    if (endShares < 0.0001) {{
                        // Fully sold: use realized return
                        endValueForAnnualized = salesConverted;
                        costBasisForAnnualized = soldCostBasisConverted;
                    }} else {{
                        // Still holding: use unrealized return at effective end date
                        const effectiveEndStr = effectiveEnd.toISOString().split('T')[0];
                        const priceAtEnd = getClosestPrice(ticker, effectiveEndStr, 'backward') || endPriceNative;
                        const valueNativeAtEnd = endShares * priceAtEnd;
                        
                        // Convert to display currency at effective end date FX rate
                        const fxRateAtEnd = getHistoricalFxRate(nativeCurrency, effectiveEndStr, 'backward');
                        const displayRateAtEnd = currency === 'USD' ? 1 : getHistoricalFxRate(currency, effectiveEndStr, 'backward');
                        
                        if (currency === nativeCurrency) {{
                            endValueForAnnualized = valueNativeAtEnd;
                        }} else if (currency === 'USD') {{
                            endValueForAnnualized = nativeCurrency === 'USD' ? valueNativeAtEnd : valueNativeAtEnd / fxRateAtEnd;
                        }} else {{
                            const valueUsd = nativeCurrency === 'USD' ? valueNativeAtEnd : valueNativeAtEnd / fxRateAtEnd;
                            endValueForAnnualized = valueUsd * displayRateAtEnd;
                        }}
                        costBasisForAnnualized = costBasisHistorical;
                    }}
                    
                    // Calculate return percentage
                    const returnPctForAnnualized = costBasisForAnnualized > 0 ? 
                        ((endValueForAnnualized - costBasisForAnnualized) / costBasisForAnnualized) * 100 : 0;
                    
                    // Annualized return formula: ((1 + r)^(365/days) - 1) * 100
                    const returnMultiplier = 1 + returnPctForAnnualized / 100;
                    if (returnMultiplier > 0) {{
                        annualizedReturn = 100 * (Math.pow(returnMultiplier, 365 / holdingDays) - 1);
                    }}
                }}

                combined.push({{
                    ticker: ticker,
                    name: info.name,
                    currency: nativeCurrency,
                    shares: endShares,
                    start_shares: startShares,
                    portfolios: portfolioList,
                    start_price: displayStartPrice,
                    end_price: endPrice,
                    start_value: displayStartValue,
                    end_value: endValueConverted,
                    total_cost: totalCost,
                    cost_basis_historical: costBasisHistorical,
                    total_invested: totalInvestedForTicker,
                    avg_cost: avgCost,
                    // All-time returns (for "All Time" view)
                    total_return: totalReturnCalc,
                    total_return_pct: totalReturnPct,
                    all_time_local_return: localReturnCalc,
                    all_time_local_return_pct: localReturnPctCalc,
                    all_time_fx_return: fxReturnCalc,
                    all_time_fx_return_pct: fxReturnPctCalc,
                    all_time_unrealized: unrealizedReturn,
                    all_time_realized: realizedReturn,
                    // Period-specific returns (FX based on period start/end rates)
                    period_local_return: periodLocalReturn,
                    period_local_return_pct: periodLocalReturnPct,
                    period_fx_return: periodFxReturn,
                    period_fx_return_pct: periodFxReturnPct,
                    // Period-specific unrealized and realized with FX breakdown
                    period_unrealized_local: periodUnrealizedLocal,
                    period_unrealized_fx: periodUnrealizedFx,
                    period_unrealized_actual: periodUnrealizedActual,
                    period_realized_local: periodRealizedLocal,
                    period_realized_fx: periodRealizedFx,
                    period_realized_actual: periodRealizedActual,
                    // Main return field - used for display
                    period_return: isAllTime ? totalReturnCalc : actualPeriodReturnDisplay,
                    period_return_pct: isAllTime ? totalReturnPct : (periodBaseDisplay > 0 ? (actualPeriodReturnDisplay / periodBaseDisplay) * 100 : 0),
                    // FX breakdown fields (switches based on view mode)
                    local_return: isAllTime ? localReturnCalc : periodLocalReturn,
                    local_return_pct: isAllTime ? localReturnPctCalc : periodLocalReturnPct,
                    fx_return: isAllTime ? fxReturnCalc : periodFxReturn,
                    fx_return_pct: isAllTime ? fxReturnPctCalc : periodFxReturnPct,
                    // Unrealized/Realized (switches based on view mode)
                    unrealized_return: isAllTime ? unrealizedReturn : periodUnrealizedActual,
                    realized_return: isAllTime ? realizedReturn : periodRealizedActual,
                    unrealized_local: isAllTime ? localUnrealized : periodUnrealizedLocal,
                    unrealized_fx: isAllTime ? (unrealizedReturn - localUnrealized) : periodUnrealizedFx,
                    realized_local: isAllTime ? localRealized : periodRealizedLocal,
                    realized_fx: isAllTime ? (realizedReturn - localRealized) : periodRealizedFx,
                    purchases: purchasesConverted,
                    sales: salesConverted,
                    sold_cost_basis: soldCostBasisConverted,
                    bought_during_period: boughtDuringPeriod,
                    sold_during_period: soldDuringPeriod,
                    has_unrealized: isAllTime ? (costBasisHistorical > 0) : (endShares > 0.0001),
                    has_realized: isAllTime ? (soldCostBasisConverted > 0) : (salesNative > 0),
                    // New percentage fields
                    unrealized_pct: unrealizedPct,
                    realized_pct: realizedPct,
                    unrealized_local_pct: unrealizedLocalPct,
                    unrealized_fx_pct: unrealizedFxPct,
                    realized_local_pct: realizedLocalPct,
                    realized_fx_pct: realizedFxPct,
                    annualized_return: annualizedReturn
                }});
            }}

            // Sort
            const field = currentSort.field;
            combined.sort((a, b) => {{
                let aVal = a[field];
                let bVal = b[field];

                if (field === 'portfolios') {{
                    aVal = a.portfolios.join(', ').toLowerCase();
                    bVal = b.portfolios.join(', ').toLowerCase();
                }} else if (typeof aVal === 'string') {{
                    aVal = aVal.toLowerCase();
                    bVal = bVal.toLowerCase();
                }}

                if (aVal < bVal) return currentSort.ascending ? -1 : 1;
                if (aVal > bVal) return currentSort.ascending ? 1 : -1;
                return 0;
            }});

            updateSortIndicators();

            // Calculate summary
            const totalStartValue = combined.reduce((sum, h) => sum + h.start_value, 0);
            const totalEndValue = combined.reduce((sum, h) => sum + h.end_value, 0);
            const totalCost = combined.reduce((sum, h) => sum + h.total_cost, 0);
            const totalCostHistorical = combined.reduce((sum, h) => sum + h.cost_basis_historical, 0);
            const totalInvested = combined.reduce((sum, h) => sum + (h.total_invested || 0), 0);
            const totalPeriodReturn = combined.reduce((sum, h) => sum + h.period_return, 0);
            const totalLocalReturn = combined.reduce((sum, h) => sum + h.local_return, 0);
            const totalFxReturn = combined.reduce((sum, h) => sum + h.fx_return, 0);
            const totalSales = combined.reduce((sum, h) => sum + (h.sales || 0), 0);
            const totalSoldCostBasis = combined.reduce((sum, h) => sum + (h.sold_cost_basis || 0), 0);
            const totalRealizedReturn = combined.reduce((sum, h) => sum + (h.realized_return || 0), 0);
            const totalUnrealizedReturn = combined.reduce((sum, h) => sum + (h.unrealized_return || 0), 0);
            
            // Unrealized and Realized FX breakdown totals
            const totalUnrealizedLocal = combined.reduce((sum, h) => sum + (h.unrealized_local || 0), 0);
            const totalUnrealizedFx = combined.reduce((sum, h) => sum + (h.unrealized_fx || 0), 0);
            const totalRealizedLocal = combined.reduce((sum, h) => sum + (h.realized_local || 0), 0);
            const totalRealizedFx = combined.reduce((sum, h) => sum + (h.realized_fx || 0), 0);
            
            // Total Return = Realized + Unrealized, percentage based on total invested
            const totalPeriodReturnPct = totalInvested > 0 ? (totalPeriodReturn / totalInvested) * 100 : 0;
            const totalLocalReturnPct = totalInvested > 0 ? (totalLocalReturn / totalInvested) * 100 : 0;
            const totalFxReturnPct = totalInvested > 0 ? (totalFxReturn / totalInvested) * 100 : 0;
            
            // Accumulated Return = Realized + Unrealized (same as totalPeriodReturn now)
            const accumulatedReturn = totalRealizedReturn + totalUnrealizedReturn;
            const accumulatedReturnPct = totalInvested > 0 ? (accumulatedReturn / totalInvested) * 100 : 0;
            const realizedReturnPct = totalSoldCostBasis > 0 ? (totalRealizedReturn / totalSoldCostBasis) * 100 : 0;
            const unrealizedReturnPct = totalCostHistorical > 0 ? (totalUnrealizedReturn / totalCostHistorical) * 100 : 0;
            
            // FX breakdown percentages for summary (based on cost basis)
            const unrealizedLocalPct = totalCostHistorical > 0 ? (totalUnrealizedLocal / totalCostHistorical) * 100 : 0;
            const unrealizedFxPct = totalCostHistorical > 0 ? (totalUnrealizedFx / totalCostHistorical) * 100 : 0;
            const realizedLocalPct = totalSoldCostBasis > 0 ? (totalRealizedLocal / totalSoldCostBasis) * 100 : 0;
            const realizedFxPct = totalSoldCostBasis > 0 ? (totalRealizedFx / totalSoldCostBasis) * 100 : 0;

            // Calculate average annual return
            // Formula: (1 + r/100) = (1 + x/100)^(D/365)
            // Solving for x: x = 100 * ((1 + r/100)^(365/D) - 1)
            const startDateObj = new Date(startDate);
            const endDateObj = new Date(endDate);
            const periodDays = Math.round((endDateObj - startDateObj) / (1000 * 60 * 60 * 24));
            
            let annualizedReturnPct = 0;
            if (periodDays > 0 && totalPeriodReturnPct > -100) {{
                const totalReturnMultiplier = 1 + totalPeriodReturnPct / 100;
                annualizedReturnPct = 100 * (Math.pow(totalReturnMultiplier, 365 / periodDays) - 1);
            }}

            // Build summary HTML - use totalInvested since returns are based on cost basis
            const startLabel = 'Cost Basis';
            const startSubValue = 'Total invested';
            const returnLabel = 'Total Return';

            let summaryHTML = `
                <div class="summary-card">
                    <div class="label">${{startLabel}}</div>
                    <div class="value">${{formatNumber(totalCostHistorical, currency)}}</div>
                    <div class="sub-value">${{startSubValue}}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Current Value</div>
                    <div class="value">${{formatNumber(totalEndValue, currency)}}</div>
                    <div class="sub-value">${{endDate}}</div>
                </div>
                <div class="summary-card">
                    <div class="label">${{returnLabel}}</div>
                    <div class="value ${{totalPeriodReturn >= 0 ? 'positive' : 'negative'}}">
                        ${{formatSignedNumber(totalPeriodReturn, currency)}}
                    </div>
                    <div class="sub-value ${{totalPeriodReturnPct >= 0 ? 'positive' : 'negative'}}">
                        ${{formatPercent(totalPeriodReturnPct)}}
                    </div>
                </div>
                <div class="summary-card">
                    <div class="label">Avg Annual Return</div>
                    <div class="value ${{annualizedReturnPct >= 0 ? 'positive' : 'negative'}}">
                        ${{formatPercent(annualizedReturnPct)}}
                    </div>
                    <div class="sub-value">${{periodDays}} days</div>
                </div>
            `;

            // Show FX breakdown in summary if enabled
            if (showFxBreakdown) {{
                summaryHTML += `
                    <div class="summary-card">
                        <div class="label">Unrealized Local</div>
                        <div class="value ${{totalUnrealizedLocal >= 0 ? 'positive' : 'negative'}}">
                            ${{formatSignedNumber(totalUnrealizedLocal, currency)}}
                        </div>
                        <div class="sub-value ${{unrealizedLocalPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(unrealizedLocalPct)}}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Unrealized FX</div>
                        <div class="value ${{totalUnrealizedFx >= 0 ? 'positive' : 'negative'}}">
                            ${{formatSignedNumber(totalUnrealizedFx, currency)}}
                        </div>
                        <div class="sub-value ${{unrealizedFxPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(unrealizedFxPct)}}</div>
                    </div>
                `;
            }}

            // Show Unrealized returns
            summaryHTML += `
                <div class="summary-card">
                    <div class="label">Unrealized</div>
                    <div class="value ${{totalUnrealizedReturn >= 0 ? 'positive' : 'negative'}}">
                        ${{formatSignedNumber(totalUnrealizedReturn, currency)}}
                    </div>
                    <div class="sub-value ${{unrealizedReturnPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(unrealizedReturnPct)}}</div>
                </div>
            `;
            
            // Show FX breakdown for Realized if enabled
            if (showFxBreakdown && totalSoldCostBasis > 0) {{
                summaryHTML += `
                    <div class="summary-card">
                        <div class="label">Realized Local</div>
                        <div class="value ${{totalRealizedLocal >= 0 ? 'positive' : 'negative'}}">
                            ${{formatSignedNumber(totalRealizedLocal, currency)}}
                        </div>
                        <div class="sub-value ${{realizedLocalPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(realizedLocalPct)}}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Realized FX</div>
                        <div class="value ${{totalRealizedFx >= 0 ? 'positive' : 'negative'}}">
                            ${{formatSignedNumber(totalRealizedFx, currency)}}
                        </div>
                        <div class="sub-value ${{realizedFxPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(realizedFxPct)}}</div>
                    </div>
                `;
            }}
            
            if (totalSoldCostBasis > 0) {{
                summaryHTML += `
                    <div class="summary-card">
                        <div class="label">Realized</div>
                        <div class="value ${{totalRealizedReturn >= 0 ? 'positive' : 'negative'}}">
                            ${{formatSignedNumber(totalRealizedReturn, currency)}}
                        </div>
                        <div class="sub-value ${{realizedReturnPct >= 0 ? 'positive' : 'negative'}}">${{formatPercent(realizedReturnPct)}}</div>
                    </div>
                `;
            }}

            document.getElementById('summary').innerHTML = summaryHTML;

            // Update table - add FX breakdown columns if enabled
            const tbody = document.getElementById('tableBody');
            
            if (showFxBreakdown) {{
                // Update header row to include FX breakdown for Unrealized and Realized
                const headerRow = document.querySelector('thead tr');
                headerRow.innerHTML = `
                    <th class="sortable" data-field="ticker">Stock</th>
                    <th class="sortable" data-field="portfolios">Portfolios</th>
                    <th class="sortable number" data-field="shares">Shares</th>
                    <th class="sortable number" data-field="cost_basis_historical">Cost Basis</th>
                    <th class="sortable number" data-field="end_value">End Value</th>
                    <th class="sortable number" data-field="unrealized_local">Unreal. Local</th>
                    <th class="sortable number" data-field="unrealized_fx">Unreal. FX</th>
                    <th class="sortable number" data-field="unrealized_return">Unrealized</th>
                    <th class="sortable number" data-field="annualized_return">Annualized</th>
                    <th class="sortable number" data-field="realized_local">Real. Local</th>
                    <th class="sortable number" data-field="realized_fx">Real. FX</th>
                    <th class="sortable number" data-field="realized_return">Realized</th>
                `;
                
                // Re-attach click handlers
                document.querySelectorAll('th.sortable').forEach(th => {{
                    th.addEventListener('click', () => sortBy(th.dataset.field));
                }});
                
                tbody.innerHTML = combined.map(h => `
                    <tr>
                        <td>
                            <div class="ticker">${{h.ticker}}${{h.shares < 0.0001 ? ' <span style="color: #ff9800; font-size: 0.8em;">(Sold)</span>' : ''}}</div>
                            <div class="stock-name">${{h.name}}</div>
                        </td>
                        <td>
                            <div class="portfolio-tags">
                                ${{h.portfolios.map(p => `<span class="portfolio-tag">${{p}}</span>`).join('')}}
                            </div>
                        </td>
                        <td class="number">${{h.shares.toLocaleString('en-US', {{maximumFractionDigits: 4}})}}</td>
                        <td class="number">${{formatNumber(h.cost_basis_historical, currency)}}</td>
                        <td class="number">${{formatNumber(h.end_value, currency)}}</td>
                        <td class="number ${{h.unrealized_local >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? `${{formatSignedNumber(h.unrealized_local, currency)}} (${{formatPercent(h.unrealized_local_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.unrealized_fx >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? `${{formatSignedNumber(h.unrealized_fx, currency)}} (${{formatPercent(h.unrealized_fx_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.unrealized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? `${{formatSignedNumber(h.unrealized_return, currency)}} (${{formatPercent(h.unrealized_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.annualized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? formatPercent(h.annualized_return) : '-'}}
                        </td>
                        <td class="number ${{h.realized_local >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_realized ? `${{formatSignedNumber(h.realized_local, currency)}} (${{formatPercent(h.realized_local_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.realized_fx >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_realized ? `${{formatSignedNumber(h.realized_fx, currency)}} (${{formatPercent(h.realized_fx_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.realized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_realized ? `${{formatSignedNumber(h.realized_return, currency)}} (${{formatPercent(h.realized_pct)}})` : '-'}}
                        </td>
                    </tr>
                `).join('');
            }} else {{
                // Standard view without FX breakdown
                const headerRow = document.querySelector('thead tr');
                headerRow.innerHTML = `
                    <th class="sortable" data-field="ticker">Stock</th>
                    <th class="sortable" data-field="portfolios">Portfolios</th>
                    <th class="sortable number" data-field="shares">Shares</th>
                    <th class="sortable number" data-field="cost_basis_historical">Cost Basis</th>
                    <th class="sortable number" data-field="end_value">End Value</th>
                    <th class="sortable number" data-field="unrealized_return">Unrealized</th>
                    <th class="sortable number" data-field="annualized_return">Annualized</th>
                    <th class="sortable number" data-field="realized_return">Realized</th>
                `;
                
                // Re-attach click handlers
                document.querySelectorAll('th.sortable').forEach(th => {{
                    th.addEventListener('click', () => sortBy(th.dataset.field));
                }});
                
                tbody.innerHTML = combined.map(h => `
                    <tr>
                        <td>
                            <div class="ticker">${{h.ticker}}${{h.shares < 0.0001 ? ' <span style="color: #ff9800; font-size: 0.8em;">(Sold)</span>' : ''}}</div>
                            <div class="stock-name">${{h.name}}</div>
                        </td>
                        <td>
                            <div class="portfolio-tags">
                                ${{h.portfolios.map(p => `<span class="portfolio-tag">${{p}}</span>`).join('')}}
                            </div>
                        </td>
                        <td class="number">${{h.shares.toLocaleString('en-US', {{maximumFractionDigits: 4}})}}</td>
                        <td class="number">${{formatNumber(h.cost_basis_historical, currency)}}</td>
                        <td class="number">${{formatNumber(h.end_value, currency)}}</td>
                        <td class="number ${{h.unrealized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? `${{formatSignedNumber(h.unrealized_return, currency)}} (${{formatPercent(h.unrealized_pct)}})` : '-'}}
                        </td>
                        <td class="number ${{h.annualized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_unrealized ? formatPercent(h.annualized_return) : '-'}}
                        </td>
                        <td class="number ${{h.realized_return >= 0 ? 'positive' : 'negative'}}">
                            ${{h.has_realized ? `${{formatSignedNumber(h.realized_return, currency)}} (${{formatPercent(h.realized_pct)}})` : '-'}}
                        </td>
                    </tr>
                `).join('');
            }}
            
            updateSortIndicators();
        }}

        // Initialize
        initFilters();
    </script>
</body>
</html>'''

    return html


def main():
    # Load transactions
    print("Loading transactions...")
    df = load_transactions(CSV_PATH)

    # Get unique tickers and portfolios
    tickers = df['ticker'].unique().tolist()
    portfolios = sorted(df['portfolio'].unique().tolist())

    # Determine date range from transactions
    min_txn_date = df['date'].min()
    start_date = min_txn_date.strftime('%Y-%m-%d')

    # Fetch current prices and names
    print("Fetching current stock prices...")
    prices, names = fetch_current_prices(tickers)

    # Fetch historical prices
    print(f"Fetching historical prices from {start_date}...")
    historical_prices = fetch_historical_prices(tickers, start_date)

    # Fetch split histories for all tickers
    print("Fetching stock split histories...")
    split_data = {}
    for ticker in tickers:
        splits = get_split_history(ticker)
        if not splits.empty:
            # Convert to dict with date string keys
            split_dict = {}
            for date, ratio in splits.items():
                if hasattr(date, 'tz') and date.tz is not None:
                    date_str = date.tz_convert('UTC').strftime('%Y-%m-%d')
                else:
                    date_str = date.strftime('%Y-%m-%d')
                split_dict[date_str] = float(ratio)
            split_data[ticker] = split_dict
            print(f"  {ticker}: {len(split_dict)} splits found")
        else:
            split_data[ticker] = {}

    # Determine available years from historical data
    all_dates = set()
    for ticker, price_dict in historical_prices.items():
        all_dates.update(price_dict.keys())

    if all_dates:
        all_dates = sorted(all_dates)
        min_date = all_dates[0]
        max_date = all_dates[-1]
        available_years = sorted(set(d[:4] for d in all_dates), reverse=True)
    else:
        min_date = datetime.now().strftime('%Y-%m-%d')
        max_date = min_date
        available_years = [str(datetime.now().year)]

    print(f"Data range: {min_date} to {max_date}")
    print(f"Available years: {available_years}")

    # Fetch exchange rates
    print("Fetching exchange rates...")
    exchange_rates = get_exchange_rates()
    print(f"Current rates: USD=1, KRW={exchange_rates['KRW']:.2f}, CAD={exchange_rates['CAD']:.4f}")

    # Fetch historical exchange rates
    print("Fetching historical exchange rates...")
    historical_fx_rates = fetch_historical_exchange_rates(start_date)

    # Prepare transaction data for JSON
    transactions_data = []
    for _, row in df.iterrows():
        transactions_data.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'portfolio': row['portfolio'],
            'ticker': row['ticker'],
            'action': row['action'],
            'shares': float(row['shares']),
            'price': float(row['price']),
            'currency': get_stock_currency(row['ticker'])
        })

    # Prepare ticker info (names, currencies)
    ticker_info = {}
    for ticker in tickers:
        ticker_info[ticker] = {
            'name': names.get(ticker, ticker),
            'currency': get_stock_currency(ticker)
        }

    # Generate HTML
    print("Generating HTML...")
    html = generate_html(transactions_data, ticker_info, exchange_rates, portfolios,
                         historical_prices, historical_fx_rates, split_data, available_years, min_date, max_date)

    # Save to file
    output_path = 'portfolio_table.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nPortfolio table saved to: {output_path}")
    print(f"Total tickers: {len(tickers)} across {len(portfolios)} portfolios")


if __name__ == '__main__':
    main()
