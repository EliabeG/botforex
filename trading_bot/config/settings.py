# config/settings.py
import os
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class TradingConfig:
    """Configurações principais do bot de trading"""
    
    # Credenciais TickTrader
    ACCOUNT_TYPE = "Gross"
    PLATFORM = "TickTrader"
    LOGIN = "28502146"
    PASSWORD = "FmpQ32rW"
    SERVER = "ttdemomarginal.fxopen.net"
    LEVERAGE = 500
    CURRENCY = "USD"
    
    # Web API Tokens
    WEB_API_TOKEN_ID = "a7e46c03-fc43-4cff-870c-d2a425b196f3"
    WEB_API_TOKEN_KEY = "4J7j5Y8DZ27d8KJp"
    WEB_API_TOKEN_SECRET = "d8fJKY4JCqZCdFP7ttTmtq3hbN9ZdwytXxa4fEWz4r3ZNtY7Mz4aXBzXf8G8YdEg"
    WEB_API_AUTH_TYPE = "HMAC"
    WEB_API_RIGHTS = "Full"
    
    # WebSocket URLs
    WS_FEED_URL = "wss://marginalttdemowebapi.fxopen.net/feed"
    WS_TRADE_URL = "wss://marginalttdemowebapi.fxopen.net/trade"
    REST_API_URL = "https://marginalttdemowebapi.fxopen.net"
    
    # Trading Parameters
    SYMBOL = "EURUSD"
    TARGET_DAILY_PROFIT = 0.05  # 5% da banca
    MAX_DRAWDOWN = 0.20  # 20% máximo
    MAX_RISK_PER_TRADE = 0.01  # 1% por trade
    DAILY_LOSS_LIMIT = 0.03  # 3% limite diário
    
    # Execution Parameters
    MAX_SIMULTANEOUS_ORDERS = 3
    MAX_SLIPPAGE_PIPS = 0.2
    MAX_SPREAD_PIPS = 1.0
    ORDER_TIMEOUT_MS = 10000  # 10 segundos
    
    # Data Settings
    TICK_HISTORY_YEARS = 3
    REDIS_TTL_HOURS = 24
    DOM_LEVELS = 200
    DOM_SNAPSHOT_MS = 100
    
    # Optimization Settings
    WALK_FORWARD_TRAIN_MONTHS = 6
    WALK_FORWARD_TEST_MONTHS = 1
    OPTIMIZATION_SCHEDULE = "0 21 * * 0"  # Domingo 21:00 UTC
    
    # Regime Detection
    REGIME_UPDATE_MS = 500
    REGIME_CONFIDENCE_THRESHOLD = 0.60
    
    # Strategy Selection
    MAX_ACTIVE_STRATEGIES = 3
    SCORE_UPDATE_TRADES = 50
    SCORE_UPDATE_MINUTES = 30
    
    # Monitoring
    NTP_SYNC_MINUTES = 5
    METRICS_PORT = 9090
    ALERT_LATENCY_MS = 20
    ALERT_DD_WEEKLY = 0.10
    
    # Infrastructure
    VPS_LOCATION = "LD4"  # London Equinix
    TARGET_PING_MS = 5
    
    # Database
    PARQUET_PATH = "./data/ticks"
    SQLITE_PATH = "./data/strategy_meta.db"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_PATH = "./logs"
    LOG_ROTATION = "1 day"
    LOG_RETENTION = "30 days"

class RegimeThresholds:
    """Limiares para detecção de regime de mercado"""
    
    # Tendência Forte
    TREND_ADX_MIN = 25
    TREND_R2_MIN = 0.8
    TREND_WINDOW = 250  # ticks
    
    # Lateral/Range
    RANGE_BB_LOW = 0.2
    RANGE_BB_HIGH = 0.8
    RANGE_WINDOW = 500  # ticks
    
    # Alta Volatilidade
    VOLATILITY_ATR_PERCENTILE = 95
    VOLATILITY_SPREAD_DELTA = 5  # ticks
    VOLATILITY_WINDOW = 30  # segundos
    
    # Baixo Volume
    LOW_VOLUME_DEPTH = 50000  # USD
    LOW_VOLUME_WINDOW = 300  # segundos (5 min)

class StrategyCategories:
    """Categorias e quantidade de estratégias"""
    
    CATEGORIES = {
        "momentum": {
            "count": 25,
            "examples": [
                "DonchianBreak55",
                "EMAStack_8_21_50",
                "IchimokuKumo",
                "CCI_ADX",
                "HeikinAshiTrend",
                "ParabolicSAR",
                "SuperTrend",
                "MACD_Signal",
                "RSI_Divergence",
                "DMI_CrossOver"
            ]
        },
        "mean_reversion": {
            "count": 20,
            "examples": [
                "ZScoreVWAP",
                "RSI2",
                "BollingerBandFade",
                "KeltnerTouch",
                "StochasticOversold",
                "MeanReversionChannel",
                "PairDeviation",
                "VolumeWeightedReversal",
                "GapFade",
                "ExtremeBounce"
            ]
        },
        "breakout": {
            "count": 15,
            "examples": [
                "RangeExpansionIndex",
                "ATRChannelBreakout",
                "LondonOpeningRange",
                "VolatilityBreakout",
                "DonchianChannel",
                "PivotPointBreak",
                "FibonacciBreakout",
                "TriangleBreak",
                "FlagPattern",
                "WedgeBreakout"
            ]
        },
        "orderflow": {
            "count": 15,
            "examples": [
                "OrderFlowImbalance",
                "CumulativeDeltaFlip",
                "QueuePositionEdge",
                "VolumeProfilePOC",
                "FootprintPattern",
                "DOMImbalance",
                "TradeIntensity",
                "LargeOrderDetection",
                "AbsorptionPattern",
                "SweepDetector"
            ]
        },
        "ml_predictive": {
            "count": 15,
            "examples": [
                "GradientBoostSHAP",
                "CNN1DTicks",
                "TFTSeqToOne",
                "LSTMPricePredict",
                "RandomForestFeatures",
                "XGBoostSignals",
                "NeuralNetEnsemble",
                "AutoEncoder",
                "ReinforcementLearning",
                "TransformerModel"
            ]
        },
        "arbitrage": {
            "count": 5,
            "examples": [
                "EURUSD_DXY_Hedge",
                "EURJPY_GBPUSD_Ratio",
                "TriangularArb",
                "CrossPairSpread",
                "CorrelationTrade"
            ]
        },
        "news_aware": {
            "count": 5,
            "examples": [
                "NewsGapFade",
                "EconomicCalendarFilter",
                "SentimentShift",
                "NewsSpikeFade",
                "PreNewsPositioning"
            ]
        },
        "liquidity_hunt": {
            "count": 5,
            "examples": [
                "DepthSweepDetector",
                "LastLookReversal",
                "StopHuntPattern",
                "LiquidityVoid",
                "HiddenOrderDetection"
            ]
        },
        "overnight_carry": {
            "count": 5,
            "examples": [
                "SwapBiasNYClose",
                "RolloverArbitrage",
                "CarryTrade",
                "OvernightGap",
                "SessionTransition"
            ]
        }
    }

def load_config() -> TradingConfig:
    """Carrega configurações do ambiente ou arquivo"""
    config = TradingConfig()
    
    # Override com variáveis de ambiente se existirem
    for field in config.__dataclass_fields__:
        env_value = os.getenv(f"TRADING_{field}")
        if env_value:
            field_type = config.__dataclass_fields__[field].type
            if field_type in (int, float):
                setattr(config, field, field_type(env_value))
            else:
                setattr(config, field, env_value)
    
    return config

def save_config(config: TradingConfig, filepath: str = "config.json"):
    """Salva configurações em arquivo JSON"""
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)

# Instância global de configuração
CONFIG = load_config()
REGIME_CONFIG = RegimeThresholds()
STRATEGY_CONFIG = StrategyCategories()