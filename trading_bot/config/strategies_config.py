# config/strategies_config.py
"""Configurações específicas para cada categoria de estratégia"""
from typing import Dict, Any, Tuple, List # Adicionado Tuple e List

class StrategyConfig:
    """Configurações base para estratégias"""

    # Configurações globais aplicadas a todas as estratégias
    GLOBAL_SETTINGS: Dict[str, Any] = { # Adicionada tipagem
        'max_simultaneous_positions': 3,
        'min_time_between_trades': 60,  # segundos
        'use_dynamic_stops': True,
        'allow_partial_exits': True,
        'max_position_duration': 86400,  # 24 horas
        'min_profit_to_close': 0.0005,  # 5 pips
        'enable_news_filter': True,
        'enable_session_filter': True
    }

    # Configurações por categoria
    MOMENTUM_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'default_atr_multiplier': 2.0,
        'trend_confirmation_required': True,
        'min_adx_threshold': 25,
        'use_volume_confirmation': True,
        'scale_in_enabled': False,
        'max_positions_per_trend': 2,
        'exit_on_momentum_loss': True,
        'use_trailing_stop': True,
        'trailing_stop_activation': 0.002,  # 20 pips profit
        'trailing_stop_distance': 0.001     # 10 pips
    }

    MEAN_REVERSION_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'default_zscore_entry': 2.5,
        'default_zscore_exit': 0.5,
        'max_deviation_multiplier': 4.0,
        'use_bollinger_bands': True,
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'require_volume_spike': False,
        'exit_at_mean': True,
        'allow_averaging_down': True, # CUIDADO: Averaging down pode ser arriscado
        'max_average_positions': 3,
        'time_decay_factor': 0.95  # Reduz tamanho com o tempo
    }

    BREAKOUT_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'min_consolidation_periods': 20,
        'breakout_confirmation_bars': 2,
        'volume_surge_multiplier': 1.5,
        'use_donchian_channel': True,
        'donchian_period': 20,
        'atr_filter_multiplier': 1.5,
        'false_breakout_protection': True,
        'retest_entry_enabled': True,
        'momentum_confirmation': True,
        'stop_below_structure': True
    }

    ORDERFLOW_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'min_order_imbalance': 0.6,
        'volume_profile_periods': 100,
        'delta_divergence_threshold': 1000, # Unidade? (ex: volume, ticks)
        'absorption_detection': True,
        'large_order_threshold': 50000, # Unidade? (ex: unidades de moeda base, lotes)
        'sweep_detection_levels': 5,
        'tape_reading_window': 50, # Número de trades no tape
        'footprint_analysis': True,
        'cvd_confirmation': True, # Cumulative Volume Delta
        'dom_depth_required': 10
    }

    ML_PREDICTIVE_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'feature_window': 100,
        'prediction_horizon': 5, # Número de ticks/barras à frente
        'min_model_confidence': 0.7,
        'ensemble_models': ['rf', 'xgboost', 'lstm'],
        'retrain_frequency': 1440,  # minutos (24h)
        'min_training_samples': 10000,
        'feature_importance_threshold': 0.05,
        'use_online_learning': True,
        'anomaly_detection': True,
        'max_model_staleness': 7200  # segundos (2 horas)
    }

    ARBITRAGE_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'min_correlation': 0.8,
        'zscore_entry_threshold': 2.0,
        'zscore_exit_threshold': 0.5,
        'lookback_period': 1000, # Para cointegração e cálculo de spread
        'cointegration_test': True,
        'half_life_max': 50, # Períodos máximos para half-life
        'hedge_ratio_dynamic': True,
        'rebalance_frequency': 60,  # segundos
        'max_pair_deviation': 0.05, # 5% de desvio máximo permitido no spread
        'transaction_cost_buffer': 0.0002 # 2 pips de buffer para custos
    }

    NEWS_AWARE_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'news_sources': ['forexfactory', 'investing.com'], # Exemplos
        'high_impact_only': True,
        'pre_news_minutes': 30,
        'post_news_minutes': 60,
        'volatility_expansion_expected': 2.0, # Multiplicador da volatilidade normal
        'fade_spike_strategy': True,
        'sentiment_analysis': True, # Requereria integração com análise de sentimento
        'economic_calendar_filter': True,
        'avoid_major_speeches': True,
        'weekend_gap_trading': True
    }

    LIQUIDITY_HUNT_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'stop_cluster_threshold': 0.001,  # 10 pips
        'min_cluster_size': 1000000,  # $1M em volume de stops (estimado)
        'sweep_velocity_threshold': 100,  # ticks/segundo ou pips/segundo?
        'reversal_confirmation_time': 30,  # segundos
        'dom_imbalance_ratio': 3.0, # Ratio de desequilíbrio no DOM
        'hidden_liquidity_detection': True,
        'institutional_levels': True, # Considerar níveis como 00, 20, 50, 80
        'round_number_bias': 0.0001,  # 1 pip de buffer para números redondos
        'session_liquidity_profile': True, # Analisar liquidez por sessão
        'trap_detection': True # Detectar armadilhas de liquidez
    }

    OVERNIGHT_CARRY_CONFIG: Dict[str, Any] = { # Adicionada tipagem
        'min_positive_swap': 0.5,  # pips por dia
        'entry_time_window': (21, 23),  # UTC hours
        'exit_time_window': (6, 8),    # UTC hours
        'rollover_time': "21:00",      # UTC (verificar exato do broker, geralmente 21:00 ou 22:00 UTC)
        'position_hold_hours': 8,      # Duração máxima se não sair na janela de exit
        'hedge_negative_carry': True, # Se a estratégia de hedge estiver implementada
        'correlation_filter': 0.3,   # Correlação máxima com outros pares de carry
        'volatility_adjustment': True, # Ajustar tamanho da posição pela volatilidade
        'max_overnight_risk': 0.005,   # 0.5% da conta
        'weekend_positions': False
    }

    # Session-specific settings
    SESSION_CONFIG: Dict[str, Dict[str, Any]] = { # Adicionada tipagem
        'ASIA': {
            'start_hour': 23, # UTC
            'end_hour': 8,    # UTC
            'preferred_strategies': ['mean_reversion', 'overnight_carry'],
            'max_spread': 1.5, # pips
            'volatility_factor': 0.7 # Multiplicador de risco/tamanho
        },
        'LONDON': {
            'start_hour': 7,  # UTC
            'end_hour': 16, # UTC
            'preferred_strategies': ['momentum', 'breakout', 'orderflow'],
            'max_spread': 1.0, # pips
            'volatility_factor': 1.2
        },
        'NEWYORK': {
            'start_hour': 13, # UTC
            'end_hour': 22, # UTC
            'preferred_strategies': ['momentum', 'news_aware', 'liquidity_hunt'],
            'max_spread': 0.8, # pips
            'volatility_factor': 1.5
        },
        'OVERLAP': {  # London/NY overlap
            'start_hour': 13, # UTC
            'end_hour': 16,   # UTC
            'preferred_strategies': ['all'], # 'all' precisaria ser tratado especialmente
            'max_spread': 0.7, # pips
            'volatility_factor': 2.0
        }
    }

    @classmethod
    def get_strategy_config(cls, strategy_category: str) -> Dict[str, Any]:
        """Retorna configuração para categoria específica, combinada com globais."""
        configs: Dict[str, Dict[str, Any]] = { # Adicionada tipagem
            'momentum': cls.MOMENTUM_CONFIG,
            'mean_reversion': cls.MEAN_REVERSION_CONFIG,
            'breakout': cls.BREAKOUT_CONFIG,
            'orderflow': cls.ORDERFLOW_CONFIG,
            'ml_predictive': cls.ML_PREDICTIVE_CONFIG,
            'arbitrage': cls.ARBITRAGE_CONFIG,
            'news_aware': cls.NEWS_AWARE_CONFIG,
            'liquidity_hunt': cls.LIQUIDITY_HUNT_CONFIG,
            'overnight_carry': cls.OVERNIGHT_CARRY_CONFIG
        }

        base_config = cls.GLOBAL_SETTINGS.copy()
        category_specific_config = configs.get(strategy_category.lower()) # Usar lower para consistência

        if category_specific_config:
            base_config.update(category_specific_config)
        else:
            # Log ou aviso se uma categoria desconhecida for solicitada
            print(f"Aviso: Categoria de estratégia '{strategy_category}' não encontrada. Usando configurações globais.")


        return base_config

    @classmethod
    def get_session_config(cls, current_hour: int) -> Dict[str, Any]:
        """Retorna configuração baseada na sessão atual (hora UTC)."""
        for session_name, config in cls.SESSION_CONFIG.items():
            start = config['start_hour']
            end = config['end_hour']

            # Handle overnight sessions (e.g., Asia: 23:00-08:00)
            if start > end: # A sessão cruza a meia-noite
                if current_hour >= start or current_hour < end:
                    return config
            else: # A sessão é no mesmo dia
                if start <= current_hour < end:
                    return config

        # Fallback se nenhuma sessão corresponder (pode indicar um gap ou erro de configuração)
        # Poderia retornar uma configuração padrão ou a da Ásia como no original
        # print(f"Aviso: Nenhuma configuração de sessão encontrada para a hora {current_hour} UTC. Usando Ásia como padrão.")
        return cls.SESSION_CONFIG['ASIA']