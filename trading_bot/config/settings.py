# config/settings.py
import os
from dataclasses import dataclass
from typing import Dict, Any
import json
from pathlib import Path # Adicionado para melhor manipulação de caminhos

# Define um diretório base para o projeto, se necessário, ou use caminhos absolutos
# Por exemplo, se o bot está em /app no container Docker:
APP_BASE_DIR = Path(os.getenv("APP_BASE_DIR", "/app"))

@dataclass
class TradingConfig:
    """Configurações principais do bot de trading"""

    # Credenciais TickTrader (CARREGAR DE VARIÁVEIS DE AMBIENTE)
    ACCOUNT_TYPE: str = os.getenv("TRADING_ACCOUNT_TYPE", "Gross")
    PLATFORM: str = os.getenv("TRADING_PLATFORM", "TickTrader")
    LOGIN: str = os.getenv("TRADING_LOGIN", "SUA_LOGIN_AQUI") # Placeholder
    PASSWORD: str = os.getenv("TRADING_PASSWORD", "SUA_SENHA_AQUI") # Placeholder
    SERVER: str = os.getenv("TRADING_SERVER", "ttdemomarginal.fxopen.net")
    LEVERAGE: int = int(os.getenv("TRADING_LEVERAGE", "500"))
    CURRENCY: str = os.getenv("TRADING_CURRENCY", "USD")

    # Web API Tokens (CARREGAR DE VARIÁVEIS DE AMBIENTE)
    WEB_API_TOKEN_ID: str = os.getenv("TRADING_WEB_API_TOKEN_ID", "SEU_TOKEN_ID_AQUI") # Placeholder
    WEB_API_TOKEN_KEY: str = os.getenv("TRADING_WEB_API_TOKEN_KEY", "SEU_TOKEN_KEY_AQUI") # Placeholder
    WEB_API_TOKEN_SECRET: str = os.getenv("TRADING_WEB_API_TOKEN_SECRET", "SEU_TOKEN_SECRET_AQUI") # Placeholder
    WEB_API_AUTH_TYPE: str = os.getenv("TRADING_WEB_API_AUTH_TYPE", "HMAC")
    WEB_API_RIGHTS: str = os.getenv("TRADING_WEB_API_RIGHTS", "Full")

    # WebSocket URLs
    WS_FEED_URL: str = os.getenv("TRADING_WS_FEED_URL", "wss://marginalttdemowebapi.fxopen.net/feed")
    WS_TRADE_URL: str = os.getenv("TRADING_WS_TRADE_URL", "wss://marginalttdemowebapi.fxopen.net/trade")
    REST_API_URL: str = os.getenv("TRADING_REST_API_URL", "https://marginalttdemowebapi.fxopen.net")

    # Trading Parameters
    SYMBOL: str = os.getenv("TRADING_SYMBOL", "EURUSD")
    TARGET_DAILY_PROFIT: float = float(os.getenv("TRADING_TARGET_DAILY_PROFIT", "0.05"))  # 5% da banca
    MAX_DRAWDOWN: float = float(os.getenv("TRADING_MAX_DRAWDOWN", "0.20"))  # 20% máximo
    MAX_RISK_PER_TRADE: float = float(os.getenv("TRADING_MAX_RISK_PER_TRADE", "0.01"))  # 1% por trade
    DAILY_LOSS_LIMIT: float = float(os.getenv("TRADING_DAILY_LOSS_LIMIT", "0.03"))  # 3% limite diário

    # Execution Parameters
    MAX_SIMULTANEOUS_ORDERS: int = int(os.getenv("TRADING_MAX_SIMULTANEOUS_ORDERS", "3"))
    MAX_SLIPPAGE_PIPS: float = float(os.getenv("TRADING_MAX_SLIPPAGE_PIPS", "0.2"))
    MAX_SPREAD_PIPS: float = float(os.getenv("TRADING_MAX_SPREAD_PIPS", "1.0"))
    ORDER_TIMEOUT_MS: int = int(os.getenv("TRADING_ORDER_TIMEOUT_MS", "10000"))  # 10 segundos

    # Data Settings
    TICK_HISTORY_YEARS: int = int(os.getenv("TRADING_TICK_HISTORY_YEARS", "3"))
    REDIS_TTL_HOURS: int = int(os.getenv("TRADING_REDIS_TTL_HOURS", "24"))
    DOM_LEVELS: int = int(os.getenv("TRADING_DOM_LEVELS", "200"))
    DOM_SNAPSHOT_MS: int = int(os.getenv("TRADING_DOM_SNAPSHOT_MS", "100"))

    # Optimization Settings
    WALK_FORWARD_TRAIN_MONTHS: int = int(os.getenv("TRADING_WALK_FORWARD_TRAIN_MONTHS", "6"))
    WALK_FORWARD_TEST_MONTHS: int = int(os.getenv("TRADING_WALK_FORWARD_TEST_MONTHS", "1"))
    OPTIMIZATION_SCHEDULE: str = os.getenv("TRADING_OPTIMIZATION_SCHEDULE", "0 21 * * 0")  # Domingo 21:00 UTC

    # Regime Detection
    REGIME_UPDATE_MS: int = int(os.getenv("TRADING_REGIME_UPDATE_MS", "500"))
    REGIME_CONFIDENCE_THRESHOLD: float = float(os.getenv("TRADING_REGIME_CONFIDENCE_THRESHOLD", "0.60"))

    # Strategy Selection
    MAX_ACTIVE_STRATEGIES: int = int(os.getenv("TRADING_MAX_ACTIVE_STRATEGIES", "3"))
    SCORE_UPDATE_TRADES: int = int(os.getenv("TRADING_SCORE_UPDATE_TRADES", "50"))
    SCORE_UPDATE_MINUTES: int = int(os.getenv("TRADING_SCORE_UPDATE_MINUTES", "30"))

    # Monitoring
    NTP_SYNC_MINUTES: int = int(os.getenv("TRADING_NTP_SYNC_MINUTES", "5"))
    METRICS_PORT: int = int(os.getenv("TRADING_METRICS_PORT", "9090"))
    ALERT_LATENCY_MS: int = int(os.getenv("TRADING_ALERT_LATENCY_MS", "200")) # Aumentado para exemplo, ajuste conforme necessário
    ALERT_DD_WEEKLY: float = float(os.getenv("TRADING_ALERT_DD_WEEKLY", "0.10"))

    # Infrastructure
    VPS_LOCATION: str = os.getenv("TRADING_VPS_LOCATION", "LD4")  # London Equinix
    TARGET_PING_MS: int = int(os.getenv("TRADING_TARGET_PING_MS", "50")) # Aumentado para exemplo

    # Database (Caminhos ajustados para Docker)
    PARQUET_PATH: str = os.getenv("TRADING_PARQUET_PATH", str(APP_BASE_DIR / "data" / "ticks"))
    SQLITE_PATH: str = os.getenv("TRADING_SQLITE_PATH", str(APP_BASE_DIR / "data" / "strategy_meta.db"))
    MODELS_PATH: str = os.getenv("TRADING_MODELS_PATH", str(APP_BASE_DIR / "models")) # Adicionado para modelos
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis") # Nome do serviço no docker-compose
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Logging (Caminho ajustado para Docker)
    LOG_LEVEL: str = os.getenv("TRADING_LOG_LEVEL", "INFO")
    LOG_PATH: str = os.getenv("TRADING_LOG_PATH", str(APP_BASE_DIR / "logs"))
    LOG_ROTATION: str = os.getenv("TRADING_LOG_ROTATION", "1 day")
    LOG_RETENTION: str = os.getenv("TRADING_LOG_RETENTION", "30 days")

    # Adicione um diretório base para dados se não estiver usando APP_BASE_DIR para tudo
    DATA_DIR: str = os.getenv("TRADING_DATA_DIR", str(APP_BASE_DIR / "data"))
    LOG_DIR: str = os.getenv("TRADING_LOG_DIR", str(APP_BASE_DIR / "logs"))


@dataclass
class RegimeThresholds:
    """Limiares para detecção de regime de mercado"""

    # Tendência Forte
    TREND_ADX_MIN: int = 25
    TREND_R2_MIN: float = 0.8
    TREND_WINDOW: int = 250  # ticks

    # Lateral/Range
    RANGE_BB_LOW: float = 0.2
    RANGE_BB_HIGH: float = 0.8
    RANGE_WINDOW: int = 500  # ticks

    # Alta Volatilidade
    VOLATILITY_ATR_PERCENTILE: int = 95
    VOLATILITY_SPREAD_DELTA: int = 5  # ticks # Ajustado de float para int baseado no nome da variável
    VOLATILITY_WINDOW: int = 30  # segundos

    # Baixo Volume
    LOW_VOLUME_DEPTH: int = 50000  # USD
    LOW_VOLUME_WINDOW: int = 300  # segundos (5 min)

@dataclass
class StrategyCategories:
    """Categorias e quantidade de estratégias"""

    CATEGORIES: Dict[str, Dict[str, Any]] = {
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

# Removida a função load_config e save_config daqui,
# pois a dataclass agora carrega diretamente de os.getenv.
# A instanciação direta é suficiente.

# Instância global de configuração
CONFIG = TradingConfig()
REGIME_CONFIG = RegimeThresholds()
STRATEGY_CONFIG = StrategyCategories()

# Para garantir que os diretórios existam ao carregar as settings
Path(CONFIG.PARQUET_PATH).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(CONFIG.SQLITE_PATH)).mkdir(parents=True, exist_ok=True)
Path(CONFIG.MODELS_PATH).mkdir(parents=True, exist_ok=True)
Path(CONFIG.LOG_PATH).mkdir(parents=True, exist_ok=True)
Path(CONFIG.DATA_DIR).mkdir(parents=True, exist_ok=True) # Garante que o diretório de dados geral exista
Path(CONFIG.LOG_DIR).mkdir(parents=True, exist_ok=True)   # Garante que o diretório de logs geral exista