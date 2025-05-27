# strategies/arbitrage/statistical_arbitrage.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timezone 
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm # Adicionado import
from sklearn.linear_model import LinearRegression

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
# from config.settings import CONFIG 

logger = setup_logger("stat_arb_eurusd_dxy") 

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Estrategia de arbitragem estatistica entre dois ativos (ex: EURUSD vs DXY simulado).
    Busca reverter a media do spread cointegrado entre os ativos.
    """

    def __init__(self):
        super().__init__("StatArb_EURUSD_vs_HedgedAsset") 
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.TREND] 
        self.min_time_between_signals_sec = 600  # Corrigido nome (era min_time_between_signals)

        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std_dev: float = 1.0 
        self.spread_half_life: float = 20.0 

        self.spread_history_series: pd.Series = pd.Series(dtype=float) 
        self.cointegration_is_valid: bool = False 
        self.last_coint_check_time: Optional[datetime] = None 

        self.main_asset_symbol: str = "EURUSD" 
        self.hedge_asset_symbol: str = "DXY_SIMULATED" 

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'lookback_period_coint': 1000, 
            'cointegration_pvalue_threshold': 0.05, 
            'min_spread_half_life': 5, 
            'max_spread_half_life': 100, 
            'spread_rolling_window': 100, 
            'entry_zscore_threshold': 2.0, 
            'exit_zscore_threshold': 0.5, 
            'stop_loss_zscore_threshold': 3.5, 
            'rebalance_hedge_threshold_pct': 0.1, 
            'max_holding_duration_periods': 200, 
            'check_stationarity_of_spread': True, 
            'coint_check_frequency_hours': 6, 
        }

    # A assinatura de calculate_indicators em BaseStrategy e `async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:`
    # Esta estrategia esta retornando um Dict. Deve atualizar self.current_indicators.
    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula o spread, Z-Score e outros indicadores de arbitragem."""
        self.current_indicators = {} # Resetar indicadores

        main_asset_ticks_df: Optional[pd.DataFrame] = market_context.get(f'recent_ticks_{self.main_asset_symbol}')
        hedge_asset_ticks_df: Optional[pd.DataFrame] = market_context.get(f'recent_ticks_{self.hedge_asset_symbol}')

        if not isinstance(main_asset_ticks_df, pd.DataFrame) or main_asset_ticks_df.empty:
            eurusd_ticks_list = market_context.get('recent_ticks', []) 
            if not eurusd_ticks_list or not (isinstance(eurusd_ticks_list[0], dict) or hasattr(eurusd_ticks_list[0], 'to_dict')):
                logger.warning("Formato de 'recent_ticks' inesperado para StatArb.")
                return 
            
            main_asset_ticks_df = pd.DataFrame([t.to_dict() if hasattr(t, 'to_dict') else t for t in eurusd_ticks_list]) # Lidar com TickData ou dict

            if 'mid' not in main_asset_ticks_df.columns or 'timestamp' not in main_asset_ticks_df.columns:
                 logger.warning(f"Colunas 'mid' ou 'timestamp' ausentes nos ticks de {self.main_asset_symbol}")
                 return 
            main_asset_ticks_df['timestamp'] = pd.to_datetime(main_asset_ticks_df['timestamp'], utc=True)
            main_asset_ticks_df = main_asset_ticks_df.set_index('timestamp').sort_index()


        hedge_asset_prices_series: pd.Series
        if not isinstance(hedge_asset_ticks_df, pd.DataFrame) or hedge_asset_ticks_df.empty:
            if self.hedge_asset_symbol == "DXY_SIMULATED": 
                hedge_asset_prices_series = self._simulate_hedge_asset_prices(main_asset_ticks_df['mid'])
                if hedge_asset_prices_series.empty: return 
            else:
                logger.warning(f"Dados para o ativo de hedge '{self.hedge_asset_symbol}' nao encontrados no market_context.")
                return 
        else: 
            if 'mid' not in hedge_asset_ticks_df.columns or 'timestamp' not in hedge_asset_ticks_df.columns:
                 logger.warning(f"Colunas 'mid' ou 'timestamp' ausentes nos ticks de {self.hedge_asset_symbol}")
                 return 
            hedge_asset_ticks_df['timestamp'] = pd.to_datetime(hedge_asset_ticks_df['timestamp'], utc=True)
            hedge_asset_prices_series = hedge_asset_ticks_df.set_index('timestamp').sort_index()['mid']


        main_asset_prices = main_asset_ticks_df['mid']
        aligned_main, aligned_hedge = main_asset_prices.align(hedge_asset_prices_series, join='inner', copy=False) 

        if len(aligned_main) < self.parameters['lookback_period_coint']:
            logger.debug(f"Dados alinhados insuficientes para StatArb: {len(aligned_main)}/{self.parameters['lookback_period_coint']}")
            return 

        now_utc = datetime.now(timezone.utc)
        if not self.cointegration_is_valid or \
           (self.last_coint_check_time and \
            (now_utc - self.last_coint_check_time) > timedelta(hours=self.parameters['coint_check_frequency_hours'])):
            
            self._perform_cointegration_check(aligned_main.values, aligned_hedge.values) 
            self.last_coint_check_time = now_utc

        if not self.cointegration_is_valid:
            self.current_indicators = {'cointegration_valid': False} 
            return 

        current_spread = aligned_main.iloc[-1] - (self.hedge_ratio * aligned_hedge.iloc[-1])
        new_spread_entry = pd.Series([current_spread], index=[aligned_main.index[-1]])
        
        if self.spread_history_series.empty: # Inicializacao
            self.spread_history_series = new_spread_entry
        else:
            self.spread_history_series = pd.concat([self.spread_history_series, new_spread_entry])
        
        if len(self.spread_history_series) > self.parameters['lookback_period_coint'] * 1.2: 
            self.spread_history_series = self.spread_history_series.iloc[-self.parameters['lookback_period_coint']:]

        zscore_val = 0.0 
        if len(self.spread_history_series) >= self.parameters['spread_rolling_window']:
            rolling_spreads = self.spread_history_series.iloc[-self.parameters['spread_rolling_window']:]
            self.spread_mean = rolling_spreads.mean()
            self.spread_std_dev = rolling_spreads.std()

            if self.spread_std_dev > 1e-9: 
                zscore_val = (current_spread - self.spread_mean) / self.spread_std_dev
            
            self.spread_half_life = self._calculate_ornstein_uhlenbeck_half_life(self.spread_history_series.values) 
        else:
            if len(self.spread_history_series) > 1:
                self.spread_mean = self.spread_history_series.mean()
                self.spread_std_dev = self.spread_history_series.std()
                if self.spread_std_dev is not None and self.spread_std_dev > 1e-9: # Checar se std_dev nao e None
                    zscore_val = (current_spread - self.spread_mean) / self.spread_std_dev
            self.spread_half_life = self.parameters['max_spread_half_life'] 


        self.current_indicators = {
            'current_spread_value': current_spread, 
            'spread_mean_rolling': self.spread_mean, 
            'spread_std_dev_rolling': self.spread_std_dev, 
            'current_zscore': zscore_val, 
            'calculated_half_life': self.spread_half_life, 
            'current_hedge_ratio': self.hedge_ratio, 
            'cointegration_valid': self.cointegration_is_valid,
            f'{self.main_asset_symbol}_price': aligned_main.iloc[-1],
            f'{self.hedge_asset_symbol}_price': aligned_hedge.iloc[-1],
            'spread_history_length': len(self.spread_history_series) 
        }
        return # Nao retornar, apenas atualizar self.current_indicators

    def _simulate_hedge_asset_prices(self, main_asset_prices_series: pd.Series) -> pd.Series: 
        """Simula dados do ativo de hedge (ex: DXY) com correlacao negativa ao ativo principal."""
        if main_asset_prices_series.empty:
            return pd.Series(dtype=float)

        dxy_base_price = 100.0  
        target_correlation = -0.85 

        main_log_returns = np.log(main_asset_prices_series / main_asset_prices_series.shift(1)).dropna()
        if main_log_returns.empty: 
            return pd.Series(dxy_base_price + np.random.normal(0, 0.01, len(main_asset_prices_series)), index=main_asset_prices_series.index)

        noise = np.random.normal(0, main_log_returns.std() * 0.5, len(main_log_returns)) 
        hedge_log_returns = target_correlation * main_log_returns + np.sqrt(1 - target_correlation**2) * noise

        sim_hedge_prices_list: List[float] = [dxy_base_price] 
        for log_ret in hedge_log_returns:
            sim_hedge_prices_list.append(sim_hedge_prices_list[-1] * np.exp(log_ret))
        
        sim_hedge_prices_series: pd.Series
        if len(sim_hedge_prices_list) > len(main_log_returns.index): 
            sim_hedge_prices_series = pd.Series(sim_hedge_prices_list[1:], index=main_log_returns.index)
            first_timestamp = main_asset_prices_series.index[0]
            sim_hedge_prices_series = pd.concat([pd.Series([dxy_base_price], index=[first_timestamp]), sim_hedge_prices_series])
        else: 
             sim_hedge_prices_series = pd.Series(sim_hedge_prices_list, index=main_asset_prices_series.index[:len(sim_hedge_prices_list)])

        return sim_hedge_prices_series.reindex(main_asset_prices_series.index, method='ffill').fillna(method='bfill')


    def _perform_cointegration_check(self, main_series_arr: np.ndarray, hedge_series_arr: np.ndarray): 
        """Verifica cointegracao entre as series de precos e atualiza hedge_ratio."""
        if len(main_series_arr) < self.parameters['lookback_period_coint'] or \
           len(hedge_series_arr) < self.parameters['lookback_period_coint']:
            logger.debug("Dados insuficientes para checagem de cointegracao.")
            self.cointegration_is_valid = False
            return

        try:
            s1 = main_series_arr[-self.parameters['lookback_period_coint']:]
            s2 = hedge_series_arr[-self.parameters['lookback_period_coint']:]

            score, pvalue, crit_values = coint(s1, s2, trend='c', autolag='AIC') 

            if pvalue < self.parameters['cointegration_pvalue_threshold']:
                X_reg = s2.reshape(-1, 1) # Usar s2 (hedge) para X
                y_reg = s1                # Usar s1 (main) para y

                model = LinearRegression()
                model.fit(X_reg, y_reg)

                self.hedge_ratio = model.coef_[0]
                self.cointegration_is_valid = True
                logger.info(f"Cointegracao VALIDA detectada. P-valor: {pvalue:.4f}, Hedge Ratio: {self.hedge_ratio:.4f}")
                
                # Recalcular historico de spread com novo hedge_ratio e timestamps corretos
                # Se main_series_arr e hedge_series_arr foram alinhados de DataFrames com DatetimeIndex:
                # Precisamos de um DatetimeIndex para o spread_history_series.
                # Se os dados originais (main_asset_prices e hedge_asset_prices_series) tinham DatetimeIndex, usa-los.
                # Como nao temos acesso direto aos timestamps aqui, uma solucao seria passar os timestamps
                # ou reconstruir o spread_history_series no metodo calculate_indicators apos a checagem.
                # Por enquanto, limpando o historico para ser reconstruido.
                self.spread_history_series = pd.Series(dtype=float) 
                # Idealmente:
                # aligned_main_full, aligned_hedge_full = pd.Series(main_series_arr).align(pd.Series(hedge_series_arr), join='inner')
                # recalculated_spreads = aligned_main_full - self.hedge_ratio * aligned_hedge_full
                # self.spread_history_series = recalculated_spreads.iloc[-self.parameters['lookback_period_coint']:]


            else:
                self.cointegration_is_valid = False
                logger.warning(f"Cointegracao INVALIDA. P-valor: {pvalue:.4f} (Limite: {self.parameters['cointegration_pvalue_threshold']})")

        except Exception as e_coint: 
            logger.exception("Erro ao verificar cointegracao:") 
            self.cointegration_is_valid = False


    def _calculate_ornstein_uhlenbeck_half_life(self, spread_values_arr: np.ndarray) -> float: 
        """Calcula o half-life do spread usando o modelo de Ornstein-Uhlenbeck."""
        if len(spread_values_arr) < 20: 
            return self.parameters['max_spread_half_life']

        try:
            spread_lagged = spread_values_arr[:-1]
            delta_spread = np.diff(spread_values_arr) 

            X_hl = sm.add_constant(spread_lagged) 
            model_hl = sm.OLS(delta_spread, X_hl).fit() 

            gamma_coeff = model_hl.params[1] 

            if gamma_coeff >= 0: 
                return self.parameters['max_spread_half_life']

            half_life_val = -np.log(2) / gamma_coeff 

            return np.clip(half_life_val,
                          self.parameters['min_spread_half_life'],
                          self.parameters['max_spread_half_life'])

        except Exception as e_hl: 
            logger.debug(f"Erro ao calcular half-life do spread: {e_hl}. Usando default.")
            return self.parameters['max_spread_half_life']


    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinais de entrada/saida para a arbitragem estatistica."""
        indicators = self.current_indicators # Usar self.current_indicators

        if not indicators or not indicators.get('cointegration_valid', False):
            return None

        zscore_current = indicators.get('current_zscore', 0.0) 
        half_life_current = indicators.get('calculated_half_life', self.parameters['max_spread_half_life']) 

        if not (self.parameters['min_spread_half_life'] <= half_life_current <= self.parameters['max_spread_half_life']):
            logger.debug(f"Half-life ({half_life_current:.1f}) fora dos limites. Nenhum sinal gerado.")
            return None

        entry_thresh = self.parameters['entry_zscore_threshold'] 
        signal_trade_type: Optional[str] = None 

        if abs(zscore_current) >= entry_thresh:
            if zscore_current >= entry_thresh:
                signal_trade_type = 'sell' 
            elif zscore_current <= -entry_thresh:
                signal_trade_type = 'buy' 

        if signal_trade_type:
            return self._create_stat_arb_signal(signal_trade_type, indicators, market_context)
        return None


    def _create_stat_arb_signal(self, signal_side: str, indicators: Dict[str, Any], 
                                market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para o ativo principal da arbitragem."""
        main_asset_price = indicators.get(f'{self.main_asset_symbol}_price', 0.0)
        if main_asset_price == 0.0: 
            logger.error(f"Preco do ativo principal {self.main_asset_symbol} nao disponivel para criar sinal.")
            # Criar um sinal que provavelmente falhara na validacao ou sera ignorado
            return Signal(timestamp=datetime.now(timezone.utc), strategy_name=self.name, symbol=self.main_asset_symbol, side=signal_side, confidence=0.0, stop_loss=0.0, take_profit=0.0, reason="Preco invalido")


        spread_m = indicators['spread_mean_rolling'] 
        spread_s = indicators.get('spread_std_dev_rolling', 0.0) # Default para 0.0 se nao existir
        hedge_r = indicators['current_hedge_ratio'] 
        hedge_asset_px = indicators.get(f'{self.hedge_asset_symbol}_price', 0.0) 

        target_z_exit = self.parameters['exit_zscore_threshold']
        stop_z = self.parameters['stop_loss_zscore_threshold'] 

        spread_target_tp: float
        spread_target_sl: float

        if signal_side == 'buy': 
            spread_target_tp = spread_m - (spread_s * target_z_exit) 
            spread_target_sl = spread_m - (spread_s * stop_z)       
        else: 
            spread_target_tp = spread_m + (spread_s * target_z_exit)
            spread_target_sl = spread_m + (spread_s * stop_z)


        take_profit_price = spread_target_tp + (hedge_r * hedge_asset_px) if hedge_asset_px > 0 and spread_s > 0 else main_asset_price * (1.01 if signal_side == 'buy' else 0.99) 
        stop_loss_price = spread_target_sl + (hedge_r * hedge_asset_px) if hedge_asset_px > 0 and spread_s > 0 else main_asset_price * (0.98 if signal_side == 'buy' else 1.02) 

        if signal_side == 'buy':
            if take_profit_price <= main_asset_price: take_profit_price = main_asset_price * 1.005 
            if stop_loss_price >= main_asset_price: stop_loss_price = main_asset_price * 0.995
        else: 
            if take_profit_price >= main_asset_price: take_profit_price = main_asset_price * 0.995
            if stop_loss_price <= main_asset_price: stop_loss_price = main_asset_price * 1.005


        confidence_val = self._calculate_stat_arb_confidence(indicators) 

        return Signal(
            timestamp=datetime.now(timezone.utc), 
            strategy_name=self.name,
            symbol=self.main_asset_symbol, # Adicionado simbolo ao sinal
            side=signal_side, 
            confidence=confidence_val,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            entry_price=main_asset_price, 
            reason=f"StatArb {signal_side.upper()} {self.main_asset_symbol} vs {self.hedge_asset_symbol}. Z: {indicators.get('current_zscore', 0.0):.2f}",
            metadata={
                'zscore': indicators.get('current_zscore', 0.0),
                'current_spread': indicators.get('current_spread_value', 0.0),
                'spread_mean': spread_m,
                'spread_std_dev': spread_s,
                'half_life': indicators.get('calculated_half_life', 0.0),
                'hedge_ratio': hedge_r,
                f'{self.hedge_asset_symbol}_price_at_signal': hedge_asset_px
            }
        )


    def _calculate_stat_arb_confidence(self, indicators: Dict[str, Any]) -> float: 
        """Calcula a confianca do sinal de arbitragem estatistica."""
        confidence = 0.55 
        zscore_abs_val = abs(indicators.get('current_zscore', 0.0)) 
        half_life_val = indicators.get('calculated_half_life', self.parameters['max_spread_half_life']) 

        if zscore_abs_val > self.parameters['entry_zscore_threshold'] * 1.25: 
            confidence += 0.15
        elif zscore_abs_val > self.parameters['entry_zscore_threshold']:
            confidence += 0.05

        if half_life_val < self.parameters['min_spread_half_life'] * 2: 
            confidence += 0.15
        elif half_life_val < (self.parameters['min_spread_half_life'] + self.parameters['max_spread_half_life']) / 2 : 
            confidence += 0.05
        
        if indicators.get('spread_history_length', 0) >= self.parameters['spread_rolling_window'] * 0.8: 
            confidence += 0.05
        
        return round(np.clip(confidence, 0.5, 0.95), 4) 


    # A assinatura de evaluate_exit_conditions em BaseStrategy e:
    # async def evaluate_exit_conditions(self, open_position: Position, market_context: Dict[str, Any]) -> Optional[ExitSignal]:
    # Esta implementacao precisa ser async e usar os argumentos corretos.
    async def evaluate_exit_conditions(self, open_position: Position, 
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]: 
        """Condicoes de saida para a posicao de arbitragem."""
        
        current_indicators = self.current_indicators 
        if not current_indicators or not current_indicators.get('cointegration_valid', False):
            logger.warning(f"Saindo da posicao {open_position.id} (StatArb): Cointegracao tornou-se invalida.")
            return ExitSignal(position_id_to_close=open_position.id, reason="Cointegracao quebrada") # Corrigido nome do parametro

        current_zscore_val = current_indicators.get('current_zscore', 0.0) 
        exit_z_thresh = self.parameters['exit_zscore_threshold'] 
        stop_z_thresh = self.parameters['stop_loss_zscore_threshold'] 

        if open_position.side.lower() == 'buy': 
            if current_zscore_val >= -exit_z_thresh: 
                return ExitSignal(position_id_to_close=open_position.id, reason=f"Z-Score do spread normalizado para {current_zscore_val:.2f} (alvo > {-exit_z_thresh})")
            elif current_zscore_val <= -stop_z_thresh:
                 return ExitSignal(position_id_to_close=open_position.id, reason=f"Stop Loss por Z-Score do spread: {current_zscore_val:.2f} (limite < {-stop_z_thresh})")

        elif open_position.side.lower() == 'sell': 
            if current_zscore_val <= exit_z_thresh: 
                return ExitSignal(position_id_to_close=open_position.id, reason=f"Z-Score do spread normalizado para {current_zscore_val:.2f} (alvo < {exit_z_thresh})")
            elif current_zscore_val >= stop_z_thresh:
                return ExitSignal(position_id_to_close=open_position.id, reason=f"Stop Loss por Z-Score do spread: {current_zscore_val:.2f} (limite > {stop_z_thresh})")

        time_held_seconds = (datetime.now(timezone.utc) - open_position.open_time).total_seconds() 
        max_duration_config = self.parameters.get('max_holding_duration_periods', 200) # Em numero de "periodos"
        
        # A conversao de 'max_holding_duration_periods' para segundos depende da frequencia dos dados.
        # Se cada "periodo" e, por exemplo, 1 minuto:
        # max_duration_seconds = max_duration_config * 60
        # if time_held_seconds > max_duration_seconds:
        #     return ExitSignal(position_id_to_close=open_position.id, reason=f"Tempo maximo de holding ({time_held_seconds/3600:.1f}h) excedido.")

        return None