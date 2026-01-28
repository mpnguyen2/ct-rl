# data/trading/config.py

GROUPS = {
    "Tech": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "Finance": ["BAC", "WFC", "C", "USB", "TFC"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "MPC"],
    "Healthcare": ["JNJ", "ABBV", "MRK", "ABT", "MDT"],
}

GROUP_ORDER = list(GROUPS.keys())

# Flat ticker list used by download/preprocess
TICKERS = [t for g in GROUP_ORDER for t in GROUPS[g]]

# Company display name + 1-line description
TICKER_INFO = {
    # Tech
    "AAPL": (
        "Apple",
        "Consumer electronics and services ecosystem (iPhone, Mac, App Store).",
    ),
    "MSFT": (
        "Microsoft",
        "Enterprise software, cloud (Azure), and productivity tools (Office).",
    ),
    "AMZN": ("Amazon", "E-commerce and cloud computing leader (AWS)."),
    "GOOGL": ("Alphabet (Google)", "Search, ads, YouTube, and cloud services."),
    "META": (
        "Meta",
        "Social platforms (Facebook/Instagram) and advertising.",
    ),
    # Finance
    "BAC": ("Bank of America", "Large U.S. bank with consumer and investment banking."),
    "WFC": (
        "Wells Fargo",
        "Large U.S. bank focused on consumer and commercial banking.",
    ),
    "C": ("Citigroup", "Global bank with consumer, corporate, and markets businesses."),
    "USB": ("U.S. Bank", "Large regional bank with payments and consumer banking."),
    "TFC": ("Truist Financial", "U.S. bank formed by BB&T and SunTrust merger."),
    # Energy
    "XOM": (
        "Exxon Mobil",
        "Integrated oil & gas with upstream, refining, and chemicals.",
    ),
    "CVX": ("Chevron", "Integrated oil & gas producer and refiner."),
    "COP": ("ConocoPhillips", "Upstream-focused oil & gas exploration and production."),
    "EOG": ("EOG Resources", "Shale oil & gas exploration and production."),
    "MPC": ("Marathon Petroleum", "Refining and midstream operations."),
    # Healthcare
    "JNJ": ("Johnson & Johnson", "Pharma and medtech diversified healthcare company."),
    "ABBV": ("AbbVie", "Biopharma (immunology, oncology; e.g., Humira/Skyrizi)."),
    "MRK": ("Merck & Co.", "Pharma, vaccines, oncology (Keytruda)."),
    "ABT": ("Abbott Labs", "Medical devices, diagnostics, nutrition."),
    "MDT": ("Medtronic", "Medical devices (cardiac, neuro, diabetes tech)."),
}


def ticker_label(ticker: str) -> str:
    name = TICKER_INFO.get(ticker, (ticker, ""))[0]
    return f"{ticker}-{name}"


# Time + feature constants
NY_TZ = "America/New_York"
SESSION_MIN_PER_DAY = 390  # 09:30..15:59 (regular session minutes)
SESSION_START_HHMM = (9, 30)  # local NY time
PRICE_SCALE = 100.0  # close_scaled = close / 100

# 23 close-only features: current + lagged closes
# last 10 mins (1..10), last 10 hours in trading minutes (60..600), last 2 days (390, 780)
LAGS = [0] + list(range(1, 11)) + [60 * i for i in range(1, 11)] + [390, 780]
MAX_LAG = max(LAGS)  # 780

# API keys
API_KEY = "YOUR API KEY FOR ALPACA API HERE"
API_SECRET = "YOUR API SECRET FOR ALPACA API HERE"

# Data directory
TRAIN_NPZ = "data/trading/processed_data/train.npz"
EVAL_NPZ = "data/trading/processed_data/eval.npz"
