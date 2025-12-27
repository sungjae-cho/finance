"""
Ticker Display Names Override
Manual overrides for stock ticker display names.

These names take priority over API-fetched names.
Use this to:
- Override API names with preferred names
- Add names for tickers that don't have API info
- Use Korean/localized names instead of English

The portfolio tracker will:
1. First check this file for a name
2. Then check the local cache (company_names_cache.json)
3. Then fetch from yfinance API
4. Fall back to the ticker symbol if all else fails
"""

CSV_PATH = '../my-stocks/transactions.csv'
OUTPUT_HTML = 'portfolio_chart.html'

# Set to False to use real yfinance data (requires internet)
USE_MOCK_DATA = False

# Cache file for company names
COMPANY_NAMES_CACHE_FILE = 'company_names_cache.json'

# Base currency for portfolio valuation
BASE_CURRENCY = 'USD'

TICKER_DISPLAY_NAMES = {
    # =========================================================================
    # Override examples - uncomment and modify as needed
    # =========================================================================

    # Korean names for Korean stocks (override English API names)
    # '005930.KS': '삼성전자',        # Samsung Electronics
    # '000660.KS': 'SK하이닉스',      # SK Hynix
    # '035420.KS': '네이버',          # Naver

    # Custom short names for US stocks
    # 'GOOGL': 'Google',
    # 'META': 'Meta',

    # =========================================================================
    # Add your custom overrides below
    # =========================================================================

}

# Currency mapping for stock exchanges
EXCHANGE_CURRENCY = {
    '.KS': 'KRW',   # Korea Stock Exchange (KOSPI)
    '.KQ': 'KRW',   # Korea KOSDAQ
    '.T': 'JPY',    # Tokyo Stock Exchange
    '.HK': 'HKD',   # Hong Kong
    '.L': 'GBP',    # London
    '.PA': 'EUR',   # Paris
    '.DE': 'EUR',   # Germany (XETRA)
    '.TO': 'CAD',   # Toronto
    '.AX': 'AUD',   # Australia
}