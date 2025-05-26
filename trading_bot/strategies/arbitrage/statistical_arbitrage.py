# strategies/arbitrage/statistical_arbitrage.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timezone # Adicionado timezone
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
# from config.settings import CONFIG # Para configurações específicas se necessário

logger = setup_logger("stat_arb_eurusd_dxy") # Nome do logger mais específico

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Estratégia de arbitragem estatística entre dois ativos (ex: EURUSD vs DXY simulado).
    Busca reverter à média do spread cointegrado entre os ativos.
    """

    def __init__(self):
        # O nome da estratégia deve ser único e descritivo
        super().__init__("StatArb_EURUSD_vs_HedgedAsset") # Nome mais genérico
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.TREND] # Pode funcionar em ambos se a relação se mantém
        self.min_time_between_signals = 600  # 10 minutos, configurável

        # Parâmetros de cointegração e spread
        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std_dev: float = 1.0 # Renomeado de spread_std
        self.spread_half_life: float = 20.0 # Renomeado de half_life

        # Histórico do spread (para cálculos rolling)
        self.spread_history_series: pd.Series = pd.Series(dtype=float) # Usar pd.Series para facilidade
        self.cointegration_is_valid: bool = False # Renomeado de cointegration_valid
        self.last_coint_check_time: Optional[datetime] = None # Para controlar frequência de checagem

        # Ativo principal (ex: EURUSD) e ativo de hedge (ex: DXY)
        # Estes poderiam ser parâmetros da estratégia
        self.main_asset_symbol: str = "EURUSD" # Exemplo
        self.hedge_asset_symbol: str = "DXY_SIMULATED" # Exemplo

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Cointegration parameters
            'lookback_period_coint': 1000, # Renomeado lookback_period para clareza
            'cointegration_pvalue_threshold': 0.05, # Renomeado
            'min_spread_half_life': 5, # Renomeado
            'max_spread_half_life': 100, # Aumentado de 50, pois half-life pode ser maior
            'spread_rolling_window': 100, # Janela para média e std dev do spread

            # Entry/Exit Z-Score thresholds
            'entry_zscore_threshold': 2.0, # Renomeado
            'exit_zscore_threshold': 0.5, # Renomeado
            'stop_loss_zscore_threshold': 3.5, # Renomeado (stop se o Z-score piorar muito)

            # Risk and Position Sizing
            # 'position_size_pct_capital': 0.02, # Esta lógica deve estar no RiskManager/PositionSizer
            'rebalance_hedge_threshold_pct': 0.1, # 10% de desvio no hedge ratio para rebalancear (se implementado)
            'max_holding_duration_periods': 200, # Ex: 200 ticks/barras (relativo ao half-life)

            # Pair configuration (quais ativos usar)
            # Estes idealmente seriam parte da configuração da instância da estratégia
            # 'main_asset': "EURUSD", # Exemplo
            # 'hedge_asset_1': "DXY_SIMULATED",
            # 'use_dxy': True, # Substituído por configuração de ativos
            # 'use_gbpusd': True,
            # 'use_eurjpy': False,

            # Filters
            # 'min_spread_quality': 0.8, # Spread de negociação, não o spread da cointegração
            'check_stationarity_of_spread': True, # Adicional: verificar estacionariedade do spread (ex: teste ADF)
            'coint_check_frequency_hours': 6, # Com que frequência re-checar cointegração
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula o spread, Z-Score e outros indicadores de arbitragem."""
        # Assegurar que recent_ticks é um DataFrame e tem as colunas necessárias
        # O market_context deveria fornecer DataFrames para cada ativo do par.
        # Ex: market_context['recent_ticks_EURUSD'] e market_context['recent_ticks_DXY_SIMULATED']
        # Por enquanto, o código original simula DXY a partir de EURUSD.

        main_asset_ticks_df: Optional[pd.DataFrame] = market_context.get(f'recent_ticks_{self.main_asset_symbol}')
        hedge_asset_ticks_df: Optional[pd.DataFrame] = market_context.get(f'recent_ticks_{self.hedge_asset_symbol}')

        # Se não houver DataFrames separados, usar a simulação original
        if not isinstance(main_asset_ticks_df, pd.DataFrame) or main_asset_ticks_df.empty:
            eurusd_ticks_list = market_context.get('recent_ticks', []) # Lista de TickData ou dicts
            if not eurusd_ticks_list or not isinstance(eurusd_ticks_list[0], dict): # Checar se é lista de dicts
                logger.warning("Formato de 'recent_ticks' inesperado para StatArb.")
                return {}
            main_asset_ticks_df = pd.DataFrame(eurusd_ticks_list)
            # Garantir que 'mid' e 'timestamp' existam
            if 'mid' not in main_asset_ticks_df.columns or 'timestamp' not in main_asset_ticks_df.columns:
                 logger.warning(f"Colunas 'mid' ou 'timestamp' ausentes nos ticks de {self.main_asset_symbol}")
                 return {}
            main_asset_ticks_df['timestamp'] = pd.to_datetime(main_asset_ticks_df['timestamp'], utc=True)
            main_asset_ticks_df = main_asset_ticks_df.set_index('timestamp').sort_index()


        # Simulação de DXY (ativo de hedge) se não fornecido
        if not isinstance(hedge_asset_ticks_df, pd.DataFrame) or hedge_asset_ticks_df.empty:
            if self.hedge_asset_symbol == "DXY_SIMULATED": # Apenas simular se for o DXY simulado
                hedge_asset_prices_series = self._simulate_hedge_asset_prices(main_asset_ticks_df['mid'])
                if hedge_asset_prices_series.empty: return {}
            else:
                logger.warning(f"Dados para o ativo de hedge '{self.hedge_asset_symbol}' não encontrados no market_context.")
                return {}
        else: # Usar dados fornecidos
            if 'mid' not in hedge_asset_ticks_df.columns or 'timestamp' not in hedge_asset_ticks_df.columns:
                 logger.warning(f"Colunas 'mid' ou 'timestamp' ausentes nos ticks de {self.hedge_asset_symbol}")
                 return {}
            hedge_asset_ticks_df['timestamp'] = pd.to_datetime(hedge_asset_ticks_df['timestamp'], utc=True)
            hedge_asset_prices_series = hedge_asset_ticks_df.set_index('timestamp').sort_index()['mid']


        # Alinhar as séries de preços pelo timestamp (índice)
        main_asset_prices = main_asset_ticks_df['mid']
        aligned_main, aligned_hedge = main_asset_prices.align(hedge_asset_prices_series, join='inner', copy=False) # copy=False para performance

        if len(aligned_main) < self.parameters['lookback_period_coint']:
            logger.debug(f"Dados alinhados insuficientes para StatArb: {len(aligned_main)}/{self.parameters['lookback_period_coint']}")
            return {}

        # Verificar cointegração periodicamente
        now_utc = datetime.now(timezone.utc)
        if not self.cointegration_is_valid or \
           (self.last_coint_check_time and \
            (now_utc - self.last_coint_check_time) > timedelta(hours=self.parameters['coint_check_frequency_hours'])):
            
            self._perform_cointegration_check(aligned_main.values, aligned_hedge.values) # Passar arrays numpy
            self.last_coint_check_time = now_utc

        if not self.cointegration_is_valid:
            return {'cointegration_valid': False} # Retornar que a cointegração não é válida

        # Calcular spread atual
        current_spread = aligned_main.iloc[-1] - (self.hedge_ratio * aligned_hedge.iloc[-1])

        # Atualizar histórico do spread (usando pd.Series)
        # O spread_history_series deve conter os spreads calculados com o hedge_ratio ATUAL.
        # Se o hedge_ratio muda, o histórico do spread precisa ser recalculado.
        # Para simplificar, vamos assumir que o hedge_ratio é estável entre checagens de coint.
        # ou que o histórico é apenas dos spreads *desde a última checagem de coint*.
        # O original adicionava ao self.spread_history (lista). Com pd.Series:
        new_spread_entry = pd.Series([current_spread], index=[aligned_main.index[-1]])
        self.spread_history_series = pd.concat([self.spread_history_series, new_spread_entry])
        
        # Manter o tamanho do histórico do spread
        if len(self.spread_history_series) > self.parameters['lookback_period_coint'] * 1.2: # Um pouco mais que o lookback
            self.spread_history_series = self.spread_history_series.iloc[-self.parameters['lookback_period_coint']:]


        # Calcular estatísticas do spread (média e desvio padrão móveis)
        zscore_val = 0.0 # Renomeado
        if len(self.spread_history_series) >= self.parameters['spread_rolling_window']:
            rolling_spreads = self.spread_history_series.iloc[-self.parameters['spread_rolling_window']:]
            self.spread_mean = rolling_spreads.mean()
            self.spread_std_dev = rolling_spreads.std()

            if self.spread_std_dev > 1e-9: # Evitar divisão por zero
                zscore_val = (current_spread - self.spread_mean) / self.spread_std_dev
            
            # Calcular half-life do spread (velocidade de reversão à média)
            self.spread_half_life = self._calculate_ornstein_uhlenbeck_half_life(self.spread_history_series.values) # Passar array numpy
        else:
            # Se não houver dados suficientes para rolling window, usar o que tem ou defaults
            if len(self.spread_history_series) > 1:
                self.spread_mean = self.spread_history_series.mean()
                self.spread_std_dev = self.spread_history_series.std()
                if self.spread_std_dev > 1e-9:
                    zscore_val = (current_spread - self.spread_mean) / self.spread_std_dev
            self.spread_half_life = self.parameters['max_spread_half_life'] # Default para half-life máximo


        indicators = {
            'current_spread_value': current_spread, # Renomeado
            'spread_mean_rolling': self.spread_mean, # Renomeado
            'spread_std_dev_rolling': self.spread_std_dev, # Renomeado
            'current_zscore': zscore_val, # Renomeado
            'calculated_half_life': self.spread_half_life, # Renomeado
            'current_hedge_ratio': self.hedge_ratio, # Renomeado
            'cointegration_valid': self.cointegration_is_valid,
            f'{self.main_asset_symbol}_price': aligned_main.iloc[-1],
            f'{self.hedge_asset_symbol}_price': aligned_hedge.iloc[-1],
            'spread_history_length': len(self.spread_history_series) # Renomeado
        }
        return indicators

    def _simulate_hedge_asset_prices(self, main_asset_prices_series: pd.Series) -> pd.Series: # Renomeado e tipagem
        """Simula dados do ativo de hedge (ex: DXY) com correlação negativa ao ativo principal."""
        if main_asset_prices_series.empty:
            return pd.Series(dtype=float)

        # DXY tem correlação negativa com EURUSD
        # Simular com correlação negativa + ruído
        dxy_base_price = 100.0  # Base do índice DXY (exemplo)
        target_correlation = -0.85 # Correlação alvo

        # Usar retornos logarítmicos para simulação mais estável
        main_log_returns = np.log(main_asset_prices_series / main_asset_prices_series.shift(1)).dropna()
        if main_log_returns.empty: # Se não puder calcular retornos (ex: poucos dados)
            # Retornar uma série constante ou com pequena variação
            return pd.Series(dxy_base_price + np.random.normal(0, 0.01, len(main_asset_prices_series)), index=main_asset_prices_series.index)


        # Gerar ruído e combinar para DXY returns
        noise = np.random.normal(0, main_log_returns.std() * 0.5, len(main_log_returns)) # Ruído com metade da std dev do principal
        hedge_log_returns = target_correlation * main_log_returns + np.sqrt(1 - target_correlation**2) * noise

        # Reconstruir preços do DXY
        # Começar com o primeiro preço do DXY alinhado inversamente com o primeiro preço do EURUSD (aproximação)
        # Ou usar uma base fixa e aplicar retornos.
        sim_hedge_prices_list: List[float] = [dxy_base_price] # Renomeado
        for log_ret in hedge_log_returns:
            sim_hedge_prices_list.append(sim_hedge_prices_list[-1] * np.exp(log_ret))
        
        # Criar a série com o mesmo índice dos retornos (que é um a menos que os preços originais)
        # e depois realinhar ou preencher o primeiro valor.
        if len(sim_hedge_prices_list) > len(main_log_returns.index): # Se incluiu o base_price
            sim_hedge_prices_series = pd.Series(sim_hedge_prices_list[1:], index=main_log_returns.index)
            # Adicionar o primeiro preço
            first_timestamp = main_asset_prices_series.index[0]
            sim_hedge_prices_series = pd.concat([pd.Series([dxy_base_price], index=[first_timestamp]), sim_hedge_prices_series])
        else: # Se o número de preços simulados já corresponde
             sim_hedge_prices_series = pd.Series(sim_hedge_prices_list, index=main_asset_prices_series.index[:len(sim_hedge_prices_list)])


        return sim_hedge_prices_series.reindex(main_asset_prices_series.index, method='ffill').fillna(method='bfill')


    def _perform_cointegration_check(self, main_series_arr: np.ndarray, hedge_series_arr: np.ndarray): # Renomeado
        """Verifica cointegração entre as séries de preços e atualiza hedge_ratio."""
        if len(main_series_arr) < self.parameters['lookback_period_coint'] or \
           len(hedge_series_arr) < self.parameters['lookback_period_coint']:
            logger.debug("Dados insuficientes para checagem de cointegração.")
            self.cointegration_is_valid = False
            return

        try:
            # Usar apenas o lookback_period para o teste
            s1 = main_series_arr[-self.parameters['lookback_period_coint']:]
            s2 = hedge_series_arr[-self.parameters['lookback_period_coint']:]

            # Teste de cointegração Engle-Granger
            # H0: Não há cointegração. Se p-value baixo, rejeita H0.
            score, pvalue, crit_values = coint(s1, s2, trend='c', autolag='AIC') # trend='c' para constante no modelo de erro

            if pvalue < self.parameters['cointegration_pvalue_threshold']:
                # Calcular hedge ratio via regressão OLS: main_asset = intercept + hedge_ratio * hedge_asset
                X_reg = hedge_series_arr.reshape(-1, 1) # Reshape para sklearn
                y_reg = main_series_arr

                model = LinearRegression()
                model.fit(X_reg, y_reg)

                self.hedge_ratio = model.coef_[0]
                # Intercepto (pode ser usado para definir o 'nível justo' do spread)
                # spread_intercept = model.intercept_
                self.cointegration_is_valid = True
                logger.info(f"Cointegração VÁLIDA detectada. P-valor: {pvalue:.4f}, Hedge Ratio: {self.hedge_ratio:.4f}")
                # Resetar/recalcular o spread_history_series com o novo hedge_ratio
                recalculated_spreads = pd.Series(main_series_arr - self.hedge_ratio * hedge_series_arr, index=pd.to_datetime(np.arange(len(main_series_arr)), unit='s')) # Dummy index se não tiver timestamps
                self.spread_history_series = recalculated_spreads.iloc[-self.parameters['lookback_period_coint']:]

            else:
                self.cointegration_is_valid = False
                logger.warning(f"Cointegração INVÁLIDA. P-valor: {pvalue:.4f} (Limite: {self.parameters['cointegration_pvalue_threshold']})")
                self.hedge_ratio = 1.0 # Resetar para default ou manter o último válido?

        except Exception as e_coint: # Renomeado
            logger.exception("Erro ao verificar cointegração:") # Usar logger.exception
            self.cointegration_is_valid = False


    def _calculate_ornstein_uhlenbeck_half_life(self, spread_values_arr: np.ndarray) -> float: # Renomeado
        """Calcula o half-life do spread usando o modelo de Ornstein-Uhlenbeck."""
        if len(spread_values_arr) < 20: # Mínimo de dados para regressão
            return self.parameters['max_spread_half_life']

        try:
            # Regressão: d(Spread) = theta * Spread_lag_1 * dt + dW
            # Simplificado: Spread_t - Spread_{t-1} = (lambda * Spread_{t-1} - mu) + epsilon
            # Ou mais comum: Spread_t - Spread_{t-1} = gamma * Spread_{t-1} + c + error_t
            # Half-life = -ln(2) / gamma  (se gamma é o coeficiente da regressão de dS em S_lag)

            spread_lagged = spread_values_arr[:-1]
            delta_spread = np.diff(spread_values_arr) # Spread_t - Spread_{t-1}

            # Adicionar constante para a regressão (drift)
            X_hl = sm.add_constant(spread_lagged) # Renomeado X para X_hl (statsmodels)
            model_hl = sm.OLS(delta_spread, X_hl).fit() # Renomeado model

            gamma_coeff = model_hl.params[1] # Coeficiente de spread_lagged (não da constante) # Renomeado

            if gamma_coeff >= 0: # Se positivo, não é mean-reverting
                return self.parameters['max_spread_half_life']

            half_life_val = -np.log(2) / gamma_coeff # Renomeado

            return np.clip(half_life_val,
                          self.parameters['min_spread_half_life'],
                          self.parameters['max_spread_half_life'])

        except Exception as e_hl: # Renomeado
            logger.debug(f"Erro ao calcular half-life do spread: {e_hl}. Usando default.")
            return self.parameters['max_spread_half_life']


    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinais de entrada/saída para a arbitragem estatística."""
        indicators = self.indicators # self.indicators é atualizado por BaseStrategy.process_tick

        if not indicators or not indicators.get('cointegration_valid', False):
            return None

        zscore_current = indicators.get('current_zscore', 0.0) # Renomeado
        half_life_current = indicators.get('calculated_half_life', self.parameters['max_spread_half_life']) # Renomeado

        # Verificar se half-life está dentro dos limites aceitáveis
        if not (self.parameters['min_spread_half_life'] <= half_life_current <= self.parameters['max_spread_half_life']):
            logger.debug(f"Half-life ({half_life_current:.1f}) fora dos limites. Nenhum sinal gerado.")
            return None

        entry_thresh = self.parameters['entry_zscore_threshold'] # Renomeado
        signal_trade_type: Optional[str] = None # Renomeado

        # Sinais baseados no Z-Score do spread
        if abs(zscore_current) >= entry_thresh:
            if zscore_current >= entry_thresh:
                # Spread está alto: Vender o spread (Vender EURUSD, Comprar DXY)
                signal_trade_type = 'sell' # Vender o ativo principal (EURUSD)
            elif zscore_current <= -entry_thresh:
                # Spread está baixo: Comprar o spread (Comprar EURUSD, Vender DXY)
                signal_trade_type = 'buy' # Comprar o ativo principal (EURUSD)

        if signal_trade_type:
            # No StatArb, o "sinal" é para o par, mas a execução é em um dos ativos (o outro é hedge)
            # Aqui, o sinal é para o self.main_asset_symbol
            return self._create_stat_arb_signal(signal_trade_type, indicators, market_context)
        return None


    def _create_stat_arb_signal(self, signal_side: str, indicators: Dict[str, Any], # Renomeado e tipado
                                market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para o ativo principal da arbitragem."""
        # O preço de entrada é o preço atual do ativo principal
        main_asset_price = indicators.get(f'{self.main_asset_symbol}_price', 0.0)
        if main_asset_price == 0.0: # Se o preço não estiver disponível
            logger.error(f"Preço do ativo principal {self.main_asset_symbol} não disponível para criar sinal.")
            # Retornar um sinal inválido ou levantar um erro? Por enquanto, logar e continuar com preço 0.
            # Isso será pego pela validação do sinal.

        # Stop Loss e Take Profit são baseados no Z-Score do *spread*
        spread_current_val = indicators['current_spread_value'] # Renomeado
        spread_m = indicators['spread_mean_rolling'] # Renomeado
        spread_s = indicators['spread_std_dev_rolling'] # Renomeado
        hedge_r = indicators['current_hedge_ratio'] # Renomeado
        hedge_asset_px = indicators.get(f'{self.hedge_asset_symbol}_price', 0.0) # Renomeado

        # Calcular SL e TP para o *spread* e depois converter para preço do ativo principal
        # TP é quando o Z-Score do spread se aproxima de zero (ou do exit_zscore_threshold)
        target_z_exit = self.parameters['exit_zscore_threshold']
        stop_z = self.parameters['stop_loss_zscore_threshold'] # Renomeado

        # Preço de Take Profit para o ativo principal
        # Se comprando o spread (EURUSD Long, DXY Short): spread_atual < media_spread. Queremos que spread suba para media_spread.
        #   spread_tp = media_spread - (std_dev_spread * target_z_exit) # Se Z negativo, queremos que Z vá para -exit_zscore
        # Se vendendo o spread (EURUSD Short, DXY Long): spread_atual > media_spread. Queremos que spread caia para media_spread.
        #   spread_tp = media_spread + (std_dev_spread * target_z_exit) # Se Z positivo, queremos que Z vá para +exit_zscore

        if signal_side == 'buy': # Comprando spread (EURUSD Long) -> Z-score era negativo
            spread_target_tp = spread_m - (spread_s * target_z_exit) # Spread alvo para TP
            spread_target_sl = spread_m - (spread_s * stop_z)       # Spread alvo para SL (Z fica mais negativo)
        else: # Vendendo spread (EURUSD Short) -> Z-score era positivo
            spread_target_tp = spread_m + (spread_s * target_z_exit)
            spread_target_sl = spread_m + (spread_s * stop_z)


        # Converter spread_target para preço do ativo principal: P_main = Spread_target + HR * P_hedge
        # Assumindo que P_hedge não muda drasticamente até o TP/SL. Isso é uma simplificação.
        # Uma abordagem mais complexa envolveria monitorar o spread e sair quando ele atingir o alvo.
        take_profit_price = spread_target_tp + (hedge_r * hedge_asset_px) if hedge_asset_px > 0 else main_asset_price * (1.01 if signal_side == 'buy' else 0.99) # Fallback TP
        stop_loss_price = spread_target_sl + (hedge_r * hedge_asset_px) if hedge_asset_px > 0 else main_asset_price * (0.98 if signal_side == 'buy' else 1.02) # Fallback SL

        # Garantir que SL/TP sejam lógicos
        if signal_side == 'buy':
            if take_profit_price <= main_asset_price: take_profit_price = main_asset_price * 1.005 # Ajuste mínimo
            if stop_loss_price >= main_asset_price: stop_loss_price = main_asset_price * 0.995
        else: # sell
            if take_profit_price >= main_asset_price: take_profit_price = main_asset_price * 0.995
            if stop_loss_price <= main_asset_price: stop_loss_price = main_asset_price * 1.005


        confidence_val = self._calculate_stat_arb_confidence(indicators) # Renomeado

        return Signal(
            timestamp=datetime.now(timezone.utc), # Usar UTC
            strategy_name=self.name,
            side=signal_side, # 'buy' ou 'sell' para o ativo principal
            confidence=confidence_val,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            entry_price=main_asset_price, # Entrada a mercado no ativo principal
            # position_size será calculado pelo RiskManager
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


    def _calculate_stat_arb_confidence(self, indicators: Dict[str, Any]) -> float: # Renomeado
        """Calcula a confiança do sinal de arbitragem estatística."""
        confidence = 0.55 # Base confidence
        zscore_abs_val = abs(indicators.get('current_zscore', 0.0)) # Renomeado
        half_life_val = indicators.get('calculated_half_life', self.parameters['max_spread_half_life']) # Renomeado

        # Z-Score extremo aumenta confiança
        if zscore_abs_val > self.parameters['entry_zscore_threshold'] * 1.25: # Ex: > 2.5 se entry é 2.0
            confidence += 0.15
        elif zscore_abs_val > self.parameters['entry_zscore_threshold']:
            confidence += 0.05

        # Half-life ideal (reversão rápida à média)
        if half_life_val < self.parameters['min_spread_half_life'] * 2: # Ex: < 10 se min é 5
            confidence += 0.15
        elif half_life_val < (self.parameters['min_spread_half_life'] + self.parameters['max_spread_half_life']) / 2 : # Abaixo da média do range
            confidence += 0.05

        # Histórico de spread suficiente (robustedez das estatísticas do spread)
        if indicators.get('spread_history_length', 0) >= self.parameters['spread_rolling_window'] * 0.8: # Pelo menos 80% da janela
            confidence += 0.05
        
        # Cointegração forte (se p-value fosse guardado e acessível)
        # if indicators.get('cointegration_pvalue', 1.0) < 0.01:
        #     confidence += 0.10

        return round(np.clip(confidence, 0.5, 0.95), 4) # Limitar e arredondar


    async def calculate_exit_conditions(self, position_obj: Position, # Renomeado
                                       current_market_context: Dict[str, Any]) -> Optional[ExitSignal]: # Renomeado
        """Condições de saída para a posição de arbitragem."""
        # Recalcular indicadores atuais para o par (especialmente o Z-Score do spread)
        # Isso requer que o market_context tenha dados para AMBOS os ativos.
        # Para simplificar, vamos assumir que self.indicators foi atualizado pelo último tick
        # e contém o Z-Score atual.
        
        current_indicators = self.indicators # Usar os indicadores mais recentes da estratégia
        if not current_indicators or not current_indicators.get('cointegration_valid', False):
            logger.warning(f"Saindo da posição {position_obj.id} (StatArb): Cointegração tornou-se inválida.")
            return ExitSignal(position_id=position_obj.id, reason="Cointegração quebrada")

        current_zscore_val = current_indicators.get('current_zscore', 0.0) # Renomeado
        exit_z_thresh = self.parameters['exit_zscore_threshold'] # Renomeado
        stop_z_thresh = self.parameters['stop_loss_zscore_threshold'] # Renomeado

        # Saída principal: Z-Score do spread retornou próximo de zero (ou do threshold de saída)
        if position_obj.side.lower() == 'buy': # Comprando o spread (EURUSD Long)
            if current_zscore_val >= -exit_z_thresh: # Spread subiu de volta (Z-score menos negativo)
                return ExitSignal(position_id=position_obj.id, reason=f"Z-Score do spread normalizado para {current_zscore_val:.2f} (alvo > {-exit_z_thresh})")
            # Stop se Z-Score piorar muito
            elif current_zscore_val <= -stop_z_thresh:
                 return ExitSignal(position_id=position_obj.id, reason=f"Stop Loss por Z-Score do spread: {current_zscore_val:.2f} (limite < {-stop_z_thresh})")

        elif position_obj.side.lower() == 'sell': # Vendendo o spread (EURUSD Short)
            if current_zscore_val <= exit_z_thresh: # Spread caiu de volta (Z-score menos positivo)
                return ExitSignal(position_id=position_obj.id, reason=f"Z-Score do spread normalizado para {current_zscore_val:.2f} (alvo < {exit_z_thresh})")
            # Stop se Z-Score piorar muito
            elif current_zscore_val >= stop_z_thresh:
                return ExitSignal(position_id=position_obj.id, reason=f"Stop Loss por Z-Score do spread: {current_zscore_val:.2f} (limite > {stop_z_thresh})")


        # Saída por tempo máximo de holding (relacionado ao half-life)
        # position_obj.open_time já é datetime
        time_held_seconds = (datetime.now(timezone.utc) - position_obj.open_time).total_seconds() # Usar UTC
        # Converter half-life (que pode ser em número de períodos/ticks) para segundos
        # Isso é complexo se half-life não for em unidades de tempo fixas.
        # Assumindo que max_holding_duration_periods é em número de "ticks" ou "barras" da frequência de decisão.
        # Se a estratégia opera tick a tick, e half-life é N ticks:
        # max_duration_related_to_half_life = current_indicators.get('calculated_half_life', 20) * 2.5 # Ex: 2.5x o half-life
        # Se max_holding_duration_periods for em segundos:
        # if time_held_seconds > self.parameters['max_holding_duration_periods']:
        #     return ExitSignal(position_id=position_obj.id, reason=f"Tempo máximo de holding ({time_held_seconds/3600:.1f}h) excedido.")


        # Lógica de rebalanceamento do hedge (mais avançada, não implementada aqui)
        # if abs(current_indicators.get('current_hedge_ratio', 1.0) - position_obj.metadata.get('entry_hedge_ratio', 1.0)) > self.parameters['rebalance_hedge_threshold_pct']:
        #     logger.info(f"Hedge ratio para {position_obj.id} desviou. Considerar rebalanceamento.")

        return None # Nenhuma condição de saída atingida