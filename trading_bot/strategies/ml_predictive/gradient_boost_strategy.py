# strategies/ml_predictive/gradient_boost_strategy.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta, timezone # Adicionado timezone
import joblib
from sklearn.ensemble import GradientBoostingClassifier # Usado para o modelo
from sklearn.preprocessing import StandardScaler
import talib # Para cálculo de indicadores técnicos
from pathlib import Path # Para manipulação de caminhos

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para caminhos de modelo, etc.
# Importar TickData se for usado explicitamente nos tipos de market_context
from api.ticktrader_ws import TickData

logger = setup_logger("gradient_boost_strategy_logger") # Nome do logger específico

class GradientBoostStrategy(BaseStrategy):
    """
    Estratégia de Machine Learning usando Gradient Boosting para prever movimentos de preço.
    Utiliza uma combinação de indicadores técnicos, microestrutura de mercado,
    padrões de preço e contexto de mercado como features.
    """

    def __init__(self):
        super().__init__("GradientBoostPredictive") # Nome da estratégia
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.RANGE, MarketRegime.HIGH_VOLATILITY] # ML pode se adaptar
        self.min_time_between_signals_sec = 300  # 5 minutos

        # Modelo, scaler e features
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = StandardScaler() # Inicializar sempre
        self.feature_names: List[str] = []
        self.is_trained: bool = False

        # Caminhos para salvar/carregar modelo (usando CONFIG)
        self.model_base_path = Path(CONFIG.MODELS_PATH) / self.name
        self.model_base_path.mkdir(parents=True, exist_ok=True) # Garantir que o diretório exista
        self.model_file_path = self.model_base_path / f"{self.name}_model.pkl"
        self.scaler_file_path = self.model_base_path / f"{self.name}_scaler.pkl"
        self.features_file_path = self.model_base_path / f"{self.name}_features.json" # Salvar features como JSON

        # Buffers (opcional, para análise de predições)
        # self.prediction_history: List[Dict[str, Any]] = []
        # self.feature_importance: Dict[str, float] = {}

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros do Modelo e Predição
            'prediction_horizon_ticks': 5,  # Prever N ticks à frente (ou barras) # Renomeado
            'min_prediction_probability': 0.65, # Probabilidade mínima para considerar o sinal # Renomeado
            # 'use_ensemble_learning': True, # Renomeado (se for usar múltiplos modelos) - não implementado aqui
            'retrain_interval_hours': 24 * 7, # Retreinar semanalmente # Renomeado

            # Configuração de Features
            'use_technical_features': True, # Renomeado
            'use_microstructure_features': True, # Renomeado
            'use_price_pattern_features': True, # Renomeado
            'feature_lookback_window': 100, # Janela de lookback para cálculo de features # Renomeado

            # Gestão de Risco específica da Estratégia ML
            'confidence_to_risk_multiplier': 1.5, # Multiplicador de risco baseado na confiança da predição
            'base_atr_multiplier_sl': 2.0, # ATR SL base (ajustado pela confiança)
            'target_risk_reward_ratio': 2.0,

            # Filtros
            # 'min_feature_importance_shap': 0.02, # Se usar SHAP para importância # Renomeado
            # 'max_feature_correlation': 0.95, # Para remover features correlacionadas no pré-processamento
            'min_samples_for_training': 5000, # Mínimo de amostras para treinar/retreinar # Renomeado
        }

    async def initialize_strategy(self): # Renomeado de initialize
        """Inicializa a estratégia, tentando carregar um modelo treinado."""
        await super().initialize_strategy() # Chama o initialize da BaseStrategy
        try:
            if self.model_file_path.exists() and self.scaler_file_path.exists() and self.features_file_path.exists():
                self.model = joblib.load(self.model_file_path)
                self.scaler = joblib.load(self.scaler_file_path)
                with open(self.features_file_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.is_trained = True
                self.logger.info(f"Modelo ML '{self.name}' e scaler carregados com sucesso de {self.model_base_path}.")
            else:
                self.logger.info(f"Nenhum modelo ML '{self.name}' treinado encontrado. O modelo precisará ser treinado.")
                self.is_trained = False
        except Exception as e_load: # Renomeado
            self.logger.exception(f"Erro ao carregar modelo ML '{self.name}':") # Usar logger.exception
            self.is_trained = False


    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula todas as features necessárias para a predição do modelo ML."""
        self.current_indicators = {} # Resetar indicadores a cada tick
        try:
            recent_ticks_list = market_context.get('recent_ticks', []) # Lista de TickData
            if len(recent_ticks_list) < self.parameters['feature_lookback_window']:
                self.logger.debug(f"Dados insuficientes ({len(recent_ticks_list)}) para calcular features ML (necessário: {self.parameters['feature_lookback_window']}).")
                return

            # Converter lista de TickData para DataFrame para facilitar cálculo de features
            # Assegurar que as colunas esperadas (mid, ask, bid, bid_volume, ask_volume, spread, timestamp) existam.
            # Se recent_ticks_list for de objetos TickData:
            ticks_for_features_df = pd.DataFrame([t.to_dict() for t in recent_ticks_list])
            if 'timestamp' in ticks_for_features_df.columns: # Garantir timestamp como índice e UTC
                 ticks_for_features_df['timestamp'] = pd.to_datetime(ticks_for_features_df['timestamp'], utc=True)
                 ticks_for_features_df.set_index('timestamp', inplace=True)


            # Garantir que as colunas de preço existam e sejam float
            price_cols = ['mid', 'high', 'low', 'close', 'ask', 'bid'] # 'high', 'low', 'close' podem vir de 'ask','bid','mid'
            for p_col in price_cols:
                if p_col not in ticks_for_features_df.columns:
                    if p_col == 'close' and 'mid' in ticks_for_features_df.columns: ticks_for_features_df['close'] = ticks_for_features_df['mid']
                    elif p_col == 'high' and 'ask' in ticks_for_features_df.columns: ticks_for_features_df['high'] = ticks_for_features_df['ask']
                    elif p_col == 'low' and 'bid' in ticks_for_features_df.columns: ticks_for_features_df['low'] = ticks_for_features_df['bid']
                    else:
                        self.logger.warning(f"Coluna de preço '{p_col}' ausente nos ticks para features ML.")
                        # Preencher com um valor (ex: mid) ou retornar se crítico
                        if 'mid' in ticks_for_features_df.columns: ticks_for_features_df[p_col] = ticks_for_features_df['mid']
                        else: return

            # Garantir volume (pode ser estimado)
            if 'volume' not in ticks_for_features_df.columns:
                ticks_for_features_df['volume'] = (ticks_for_features_df.get('bid_volume', 0.0) + ticks_for_features_df.get('ask_volume', 0.0)) / 2.0
                ticks_for_features_df['volume'] = ticks_for_features_df['volume'].replace(0.0, 1.0) # Evitar volume zero

            # Garantir spread
            if 'spread' not in ticks_for_features_df.columns and 'ask' in ticks_for_features_df.columns and 'bid' in ticks_for_features_df.columns:
                 ticks_for_features_df['spread'] = ticks_for_features_df['ask'] - ticks_for_features_df['bid']


            # --- Cálculo das Features ---
            # Usar a última janela de dados para as features atuais
            # Os métodos _calculate_*_features devem retornar um dicionário com a última feature calculada.
            features_dict: Dict[str, Any] = {} # Renomeado

            if self.parameters['use_technical_features']:
                features_dict.update(self._calculate_technical_features_last(ticks_for_features_df))
            if self.parameters['use_microstructure_features']:
                features_dict.update(self._calculate_microstructure_features_last(ticks_for_features_df, recent_ticks_list))
            if self.parameters['use_price_pattern_features']:
                features_dict.update(self._calculate_price_pattern_features_last(ticks_for_features_df))

            # Features de Contexto (não dependem de série temporal longa, mas do market_context atual)
            features_dict.update(self._calculate_contextual_features(market_context)) # Renomeado

            # Armazenar features calculadas
            self.current_indicators.update(features_dict)

            # Fazer predição se o modelo estiver treinado e tivermos features
            if self.is_trained and self.model and self.scaler and self.feature_names:
                if not all(feat_name in features_dict for feat_name in self.feature_names):
                    # Logar quais features estão faltando
                    missing_f = [fn for fn in self.feature_names if fn not in features_dict]
                    self.logger.warning(f"Algumas features esperadas pelo modelo ML não foram calculadas (faltando: {missing_f}). Predição não será feita.")
                    self.current_indicators.update({'ml_prediction_class': 0, 'ml_prediction_probability': 0.5, 'ml_prediction_confidence': 0.0})
                else:
                    prediction_output = self._make_model_prediction(features_dict) # Renomeado
                    self.current_indicators.update({
                        'ml_prediction_class': prediction_output['class'], # Ex: -1 (sell), 0 (hold), 1 (buy)
                        'ml_prediction_probability': prediction_output['probability'], # Probabilidade da classe predita
                        'ml_prediction_confidence': prediction_output['confidence'] # Confiança ajustada
                    })
            else: # Modelo não treinado ou features não calculadas
                self.current_indicators.update({'ml_prediction_class': 0, 'ml_prediction_probability': 0.5, 'ml_prediction_confidence': 0.0})

            # Adicionar preço e spread atuais para referência e logs
            if not ticks_for_features_df.empty:
                self.current_indicators['current_price_mid'] = ticks_for_features_df['mid'].iloc[-1]
                self.current_indicators['current_spread_val'] = ticks_for_features_df['spread'].iloc[-1] if 'spread' in ticks_for_features_df.columns else 0.0


        except Exception as e_calc: # Renomeado
            self.logger.exception("Erro ao calcular features/indicadores para MLStrategy:")
            self.current_indicators.update({'ml_prediction_class': 0, 'ml_prediction_probability': 0.5, 'ml_prediction_confidence': 0.0, 'error_calculating_features': True})


    def _prepare_feature_series(self, df: pd.DataFrame, col_name: str) -> np.ndarray:
        """Prepara uma série de DataFrame para TA-Lib, lidando com NaNs."""
        series = df[col_name].replace([np.inf, -np.inf], np.nan).astype(float)
        # TA-Lib geralmente lida com NaNs no início, mas não no meio.
        # Se houver NaNs no meio, pode ser necessário preencher ou a janela de lookback deve ser suficiente.
        return series.values # TA-Lib espera arrays numpy


    def _calculate_technical_features_last(self, df: pd.DataFrame) -> Dict[str, Any]: # Renomeado e df
        """Calcula a última entrada de features técnicas usando TA-Lib."""
        features_tech: Dict[str, Any] = {} # Renomeado
        # Garantir que df tenha dados suficientes para o maior período
        if len(df) < 50: return features_tech # Ex: max lookback para ADX é ~27, BBands 20

        close = self._prepare_feature_series(df, 'close')
        high = self._prepare_feature_series(df, 'high')
        low = self._prepare_feature_series(df, 'low')
        volume = self._prepare_feature_series(df, 'volume')

        try:
            features_tech['rsi_14'] = talib.RSI(close, 14)[-1]
            features_tech['rsi_7'] = talib.RSI(close, 7)[-1] # Adicionado RSI mais curto
            features_tech['mom_10'] = talib.MOM(close, 10)[-1]
            features_tech['roc_10'] = talib.ROC(close, 10)[-1] # Renomeado de roc_5
            features_tech['atr_14_norm'] = talib.NATR(high, low, close, 14)[-1] # NATR é normalizado
            features_tech['adx_14'] = talib.ADX(high, low, close, 14)[-1]
            features_tech['cci_14'] = talib.CCI(high, low, close, 14)[-1]

            macd, macdsignal, macdhist = talib.MACD(close, 12, 26, 9)
            features_tech['macd_value'] = macd[-1] # Renomeado
            features_tech['macd_signal_line'] = macdsignal[-1] # Renomeado
            features_tech['macd_histogram'] = macdhist[-1] # Renomeado

            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
            features_tech['bb_percent_b'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-9) # Renomeado
            features_tech['bb_width_ratio'] = (bb_upper[-1] - bb_lower[-1]) / (bb_middle[-1] + 1e-9) # Renomeado

            slowk, slowd = talib.STOCH(high, low, close, 5, 3, 0, 3, 0) # Parâmetros padrão
            features_tech['stoch_k_slow'] = slowk[-1] # Renomeado
            features_tech['stoch_d_slow'] = slowd[-1] # Renomeado

            # Volume-based
            features_tech['obv_val'] = talib.OBV(close, volume)[-1] if len(volume) == len(close) else np.nan # Renomeado
            # ADOSC pode precisar de mais dados ou dar NaN
            # features_tech['adosc_val'] = talib.ADOSC(high, low, close, volume, 3, 10)[-1] if len(volume) == len(close) else np.nan

            # Retornar apenas valores não-NaN
            return {k: (round(v, 5) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v) for k, v in features_tech.items() if not (isinstance(v, (float, np.floating)) and np.isnan(v))}

        except Exception as e_tech: # Renomeado
            self.logger.error(f"Erro ao calcular features técnicas: {e_tech}")
            return {}


    def _calculate_microstructure_features_last(self, df: pd.DataFrame, raw_ticks_list: List[TickData]) -> Dict[str, Any]: # Renomeado e df, raw_ticks_list
        """Calcula a última entrada de features de microestrutura."""
        features_micro: Dict[str, Any] = {} # Renomeado
        if len(df) < 20: return features_micro # Precisa de alguns dados para médias, etc.

        # Spread
        spreads_arr = df['spread'].values[-20:] # Renomeado
        features_micro['spread_current_val'] = spreads_arr[-1] # Renomeado
        features_micro['spread_mean_20'] = np.mean(spreads_arr)
        features_micro['spread_std_20'] = np.std(spreads_arr)
        features_micro['spread_to_mean_ratio'] = spreads_arr[-1] / (features_micro['spread_mean_20'] + 1e-9) # Renomeado

        # Bid-Ask Imbalance (baseado nos volumes do top-of-book dos ticks)
        if raw_ticks_list and len(raw_ticks_list) >= 20:
            last_20_ticks = raw_ticks_list[-20:]
            bid_vols = np.array([t.bid_volume for t in last_20_ticks if hasattr(t, 'bid_volume')])
            ask_vols = np.array([t.ask_volume for t in last_20_ticks if hasattr(t, 'ask_volume')])
            if len(bid_vols) > 0 and len(ask_vols) > 0:
                avg_bid_vol = np.mean(bid_vols)
                avg_ask_vol = np.mean(ask_vols)
                total_vol_top = avg_bid_vol + avg_ask_vol
                features_micro['bid_ask_imbalance_ratio_20'] = (avg_bid_vol - avg_ask_vol) / (total_vol_top + 1e-9) # Renomeado
        
        # Autocorrelação de retornos de ticks (eficiência de preço)
        if len(df) >= 2:
            mid_prices_micro = df['mid'].values # Renomeado
            returns_micro = np.diff(mid_prices_micro) # Usar retornos simples para ticks
            if len(returns_micro) > 1 and np.std(returns_micro[:-1]) > 1e-9 and np.std(returns_micro[1:]) > 1e-9 :
                features_micro['tick_return_autocorr_1'] = np.corrcoef(returns_micro[:-1], returns_micro[1:])[0, 1] # Renomeado
            else:
                features_micro['tick_return_autocorr_1'] = 0.0

        return {k: (round(v, 5) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v) for k, v in features_micro.items() if not (isinstance(v, (float, np.floating)) and np.isnan(v))}


    def _calculate_price_pattern_features_last(self, df: pd.DataFrame) -> Dict[str, Any]: # Renomeado e df
        """Calcula a última entrada de features de padrões de preço."""
        features_pattern: Dict[str, Any] = {} # Renomeado
        if len(df) < 20: return features_pattern # Para S/R e padrões de candles

        # Preparar OHLC se não existirem (para padrões de candle)
        # Se 'df' já for OHLC, usar diretamente. Se for ticks, precisa agregar.
        # Assumindo que 'df' tem colunas 'open', 'high', 'low', 'close' ou podem ser derivadas.
        # Para simplificação, se não tiver 'open', usar 'close' do anterior.
        op = df['mid'].shift(1).fillna(df['mid']).values # Open aproximado
        hi = df['high'].values
        lo = df['low'].values
        cl = df['close'].values

        # Candle patterns (usar TA-Lib) - olhar últimos N candles
        candle_lookback = 10 # Olhar últimos 10 "períodos" (ticks ou barras)
        if len(cl) >= candle_lookback:
            op_lb, hi_lb, lo_lb, cl_lb = op[-candle_lookback:], hi[-candle_lookback:], lo[-candle_lookback:], cl[-candle_lookback:]
            features_pattern['pattern_hammer'] = talib.CDLHAMMER(op_lb, hi_lb, lo_lb, cl_lb)[-1]
            features_pattern['pattern_doji'] = talib.CDLDOJI(op_lb, hi_lb, lo_lb, cl_lb)[-1]
            features_pattern['pattern_engulfing'] = talib.CDLENGULFING(op_lb, hi_lb, lo_lb, cl_lb)[-1]
            # Adicionar mais padrões conforme necessidade

        # Distância para Suporte/Resistência simples (últimos 20 períodos)
        recent_highs_sr = hi[-20:] # Renomeado
        recent_lows_sr = lo[-20:] # Renomeado
        current_close_sr = cl[-1] # Renomeado
        max_recent_high = np.max(recent_highs_sr)
        min_recent_low = np.min(recent_lows_sr)

        features_pattern['dist_to_resistance_pct'] = (max_recent_high - current_close_sr) / (current_close_sr + 1e-9) if current_close_sr > 0 else 0.0 # Renomeado
        features_pattern['dist_to_support_pct'] = (current_close_sr - min_recent_low) / (current_close_sr + 1e-9) if current_close_sr > 0 else 0.0 # Renomeado

        # Inclinação da tendência (regressão linear sobre os últimos N períodos)
        trend_window_slope = 20 # Renomeado
        if len(cl) >= trend_window_slope:
            y_slope = cl[-trend_window_slope:] # Renomeado
            x_slope = np.arange(len(y_slope)) # Renomeado
            try:
                slope_val, intercept_val = np.polyfit(x_slope, y_slope, 1) # Renomeado
                features_pattern['trend_slope_20'] = slope_val
                # R-squared
                y_pred_slope = intercept_val + slope_val * x_slope # Renomeado
                ss_res_slope = np.sum((y_slope - y_pred_slope) ** 2) # Renomeado
                ss_tot_slope = np.sum((y_slope - np.mean(y_slope)) ** 2) # Renomeado
                features_pattern['trend_r2_20'] = 1.0 - (ss_res_slope / (ss_tot_slope + 1e-9)) if ss_tot_slope > 0 else 0.0
            except (np.linalg.LinAlgError, ValueError) as e_poly: # Renomeado
                self.logger.debug(f"Erro em polyfit para trend_slope_20: {e_poly}")
                features_pattern['trend_slope_20'] = 0.0
                features_pattern['trend_r2_20'] = 0.0

        return {k: (round(v, 5) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v) for k, v in features_pattern.items() if not (isinstance(v, (float, np.floating)) and np.isnan(v))}


    def _calculate_contextual_features(self, market_context: Dict[str, Any]) -> Dict[str, Any]: # Renomeado
        """Calcula features de contexto (regime, hora, dia)."""
        features_ctx: Dict[str, Any] = {} # Renomeado

        # Regime de mercado (one-hot encode)
        current_regime_ctx = market_context.get('regime', MarketRegime.RANGE) # Renomeado
        for regime_val_ctx in MarketRegime.__dict__.values(): # Renomeado
            if isinstance(regime_val_ctx, str): # Acessar os valores string
                features_ctx[f'regime_{regime_val_ctx}'] = 1 if current_regime_ctx == regime_val_ctx else 0

        # Hora do dia (codificação cíclica)
        # Usar o timestamp do tick atual do market_context
        current_ts_ctx = market_context.get('timestamp', datetime.now(timezone.utc)) # Renomeado
        if isinstance(current_ts_ctx, (int, float)): # Se for unix timestamp
            current_ts_ctx = datetime.fromtimestamp(current_ts_ctx, tz=timezone.utc)
        elif current_ts_ctx.tzinfo is None: # Se for naive datetime
            current_ts_ctx = current_ts_ctx.replace(tzinfo=timezone.utc)

        hour_of_day = current_ts_ctx.hour # Renomeado
        features_ctx['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0)
        features_ctx['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0)

        # Dia da semana (one-hot encode ou cíclico)
        day_of_week = current_ts_ctx.weekday() # Segunda=0, Domingo=6 # Renomeado
        features_ctx['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        features_ctx['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        # features_ctx[f'weekday_{day_of_week}'] = 1 # Exemplo de one-hot

        # Sessão de trading (one-hot encode)
        current_session_ctx = market_context.get('session', 'Transition').lower() # Renomeado
        for session_val_ctx in ['asia', 'london', 'newyork', 'overlap', 'transition']: # Renomeado
            features_ctx[f'session_{session_val_ctx}'] = 1 if current_session_ctx == session_val_ctx else 0

        return features_ctx

    # _calculate_fractal_dimension foi removido por ser complexo e potencialmente instável
    # Se necessário, pode ser re-adicionado com uma implementação robusta.

    def _make_model_prediction(self, current_features_dict: Dict[str, Any]) -> Dict[str, Any]: # Renomeado
        """Faz predição usando o modelo ML carregado e as features atuais."""
        if not self.is_trained or not self.model or not self.scaler or not self.feature_names:
            self.logger.debug("Modelo ML não treinado ou componentes ausentes. Predição padrão retornada.")
            return {'class': 0, 'probability': 0.5, 'confidence': 0.0, 'probabilities': [0.33, 0.34, 0.33]} # Ex: Hold

        try:
            # Preparar features na ordem correta e como array 2D
            # Garantir que todas as features esperadas estejam presentes, preenchendo com 0 ou média se faltar.
            feature_vector_list = [current_features_dict.get(name, 0.0) for name in self.feature_names] # Renomeado
            X_input_array = np.array([feature_vector_list]) # Renomeado

            # Normalizar (o scaler espera dados 2D)
            X_scaled_array = self.scaler.transform(X_input_array) # Renomeado

            # Predição
            predicted_class = self.model.predict(X_scaled_array)[0] # Classe predita (-1, 0, 1)
            class_probabilities = self.model.predict_proba(X_scaled_array)[0] # Probabilidades para cada classe

            # A ordem das probabilidades corresponde a self.model.classes_
            # Se classes_ for [-1, 0, 1], então class_probabilities[0] é P(classe=-1), etc.
            # Encontrar a probabilidade da classe predita
            predicted_class_index = np.where(self.model.classes_ == predicted_class)[0][0]
            probability_of_predicted_class = class_probabilities[predicted_class_index]

            # Calcular confiança: pode ser a própria probabilidade, ou ajustada.
            # Ex: (Probabilidade - 1/NumClasses) / (1 - 1/NumClasses) -> normaliza para 0-1 acima do acaso
            num_classes = len(self.model.classes_)
            confidence_score = (probability_of_predicted_class - (1.0/num_classes)) / (1.0 - (1.0/num_classes)) if num_classes > 1 else probability_of_predicted_class
            confidence_score = max(0.0, confidence_score) # Garantir não negativo

            return {
                'class': int(predicted_class), # Garantir que é int
                'probability': float(probability_of_predicted_class), # Garantir float
                'confidence': float(confidence_score), # Garantir float
                'all_probabilities': class_probabilities.tolist() # Lista de floats
            }

        except Exception as e_pred: # Renomeado
            self.logger.exception("Erro durante a predição do modelo ML:") # Usar logger.exception
            return {'class': 0, 'probability': 0.5, 'confidence': 0.0, 'all_probabilities': [0.33,0.33,0.34] if hasattr(self.model, 'classes_') and len(self.model.classes_)==3 else [0.5,0.5]}


    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera um sinal de trading (Signal) baseado na predição do modelo ML."""
        indic = self.current_indicators # Usar indicadores já calculados
        if not indic or not self.is_trained:
            return None # Sem indicadores ou modelo não treinado

        # Obter predição do modelo
        ml_class_pred = indic.get('ml_prediction_class', 0) # Renomeado
        ml_prob_pred = indic.get('ml_prediction_probability', 0.5) # Renomeado
        # ml_confidence_pred = indic.get('ml_prediction_confidence', 0.0)

        # Verificar se a probabilidade da predição atinge o limiar mínimo
        if ml_prob_pred < self.parameters['min_prediction_probability']:
            self.logger.debug(f"Predição ML ignorada: Probabilidade ({ml_prob_pred:.2f}) < Limiar ({self.parameters['min_prediction_probability']:.2f})")
            return None

        if ml_class_pred == 0: # Predição de "Hold"
            return None

        signal_side_ml = 'buy' if ml_class_pred == 1 else 'sell' # Renomeado

        # (Opcional) Confirmação por importância de features - lógica original mantida, mas pode ser complexa
        # if hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'):
        #     important_feature_names = self._get_top_n_important_features(top_n=5) # Renomeado
        #     confirmation_score_val = self._calculate_feature_confirmation_score(
        #         important_feature_names, indic, signal_side_ml
        #     )
        #     min_confirmation_score = self.parameters.get('min_feature_confirmation_score', 0.5)
        #     if confirmation_score_val < min_confirmation_score:
        #         self.logger.debug(f"Sinal ML para {signal_side_ml} não confirmado por features importantes (Score: {confirmation_score_val:.2f} < {min_confirmation_score:.2f}).")
        #         return None


        return self._create_signal_from_ml_prediction(signal_side_ml, indic, market_context) # Renomeado

    # _get_top_n_important_features e _calculate_feature_confirmation_score podem ser complexos e
    # dependem de como a importância das features é definida (ex: SHAP, feature_importances_).
    # Mantidos como placeholders se forem reimplementados.


    def _create_signal_from_ml_prediction(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                         market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal a partir da predição ML."""
        current_price_val = indicators_dict.get('current_price_mid', market_context['tick'].mid if market_context.get('tick') else 0.0) # Renomeado
        if current_price_val == 0.0:
            self.logger.error("Preço atual zerado ao criar sinal ML. Sinal inválido.")
            # Retornar um sinal que falhará na validação ou None
            return Signal(timestamp=datetime.now(timezone.utc), strategy_name=self.name, symbol=CONFIG.SYMBOL, side=signal_side, confidence=0.0, stop_loss=0.0, take_profit=0.0)


        # ATR para stops (deve estar em pips ou ser convertido)
        atr_pips_val = indicators_dict.get('atr_pips', 10.0) # Pegar ATR em pips se calculado, senão default
        if atr_pips_val == 0.0: atr_pips_val = 10.0 # Garantir que ATR não seja zero para SL/TP
        pip_size_val = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01

        # Ajustar SL/TP pela confiança da predição
        ml_confidence_val = indicators_dict.get('ml_prediction_confidence', 0.0) # Renomeado
        # Multiplicador de risco: confiança mais alta pode permitir risco ligeiramente maior ou SL mais justo.
        # A lógica original usava `confidence_multiplier` para /SL e *TP, o que é estranho.
        # Geralmente, maior confiança -> SL pode ser um pouco mais apertado OU tamanho da posição maior.
        # Aqui, vamos ajustar a distância do SL: maior confiança, SL pode ser um pouco mais apertado (maior divisor).
        # Se confiança = 0, risk_adj_factor = 1. Se confiança = 1, risk_adj_factor = 1 + X
        risk_adjustment_factor = 1.0 + (ml_confidence_val * (self.parameters['confidence_to_risk_multiplier'] - 1.0))

        sl_distance_pips = (atr_pips_val * self.parameters['base_atr_multiplier_sl']) / risk_adjustment_factor
        tp_distance_pips = sl_distance_pips * self.parameters['target_risk_reward_ratio']

        stop_loss_val: float # Adicionada tipagem
        take_profit_val: float # Adicionada tipagem

        if signal_side == 'buy':
            stop_loss_val = current_price_val - (sl_distance_pips * pip_size_val)
            take_profit_val = current_price_val + (tp_distance_pips * pip_size_val)
        else: # sell
            stop_loss_val = current_price_val + (sl_distance_pips * pip_size_val)
            take_profit_val = current_price_val - (tp_distance_pips * pip_size_val)

        # Confiança final do sinal (pode ser a probabilidade do modelo ou a confiança ajustada)
        final_signal_confidence = float(indicators_dict.get('ml_prediction_probability', 0.5)) # Usar probabilidade como confiança

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=round(np.clip(final_signal_confidence, 0.5, 1.0), 4), # Garantir mínimo de 0.5
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"ML Pred: {signal_side.upper()}, Prob: {indicators_dict.get('ml_prediction_probability', 0.0):.2%}, Conf: {ml_confidence_val:.2%}",
            metadata={
                'ml_class': indicators_dict.get('ml_prediction_class'),
                'ml_probability': indicators_dict.get('ml_prediction_probability'),
                'ml_confidence_score': ml_confidence_val,
                'atr_pips_at_signal': atr_pips_val,
                'sl_pips_calculated': sl_distance_pips
                # 'important_features_values': {k: indicators_dict.get(k) for k in self._get_top_n_important_features(3)} # Exemplo
            }
        )


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Avalia condições de saída para uma posição aberta baseada em ML."""
        # Indicadores (incluindo predição ML) devem ser recalculados antes desta chamada
        # pela BaseStrategy.on_tick ou um método similar.
        indic = self.current_indicators
        if not indic or not self.is_trained: # Se não há indicadores ou modelo não treinado
            return None

        # Obter predição ML atual
        current_ml_class_pred = indic.get('ml_prediction_class', 0) # Renomeado
        current_ml_prob = indic.get('ml_prediction_probability', 0.5)

        # 1. Sair se a predição do modelo reverter fortemente
        if open_position.side.lower() == 'buy' and current_ml_class_pred == -1: # Predição virou SELL
            if current_ml_prob >= self.parameters['min_prediction_probability'] * 0.9: # Um pouco menos estrito para sair
                return ExitSignal(position_id_to_close=open_position.id, reason="Predição ML reverteu para VENDA com confiança.")
        elif open_position.side.lower() == 'sell' and current_ml_class_pred == 1: # Predição virou BUY
            if current_ml_prob >= self.parameters['min_prediction_probability'] * 0.9:
                return ExitSignal(position_id_to_close=open_position.id, reason="Predição ML reverteu para COMPRA com confiança.")

        # 2. Sair se a confiança na direção da posição cair muito
        # Se a classe predita ainda for a mesma da posição, mas a probabilidade caiu.
        required_prob_for_hold = self.parameters['min_prediction_probability'] * 0.7 # Ex: 70% do limiar de entrada
        if (open_position.side.lower() == 'buy' and current_ml_class_pred == 1 and current_ml_prob < required_prob_for_hold) or \
           (open_position.side.lower() == 'sell' and current_ml_class_pred == -1 and current_ml_prob < required_prob_for_hold):
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Confiança ML na direção da posição ({open_position.side}) caiu para {current_ml_prob:.2%}")


        # 3. (Opcional) Verificar deterioração de features importantes (lógica complexa, mantida como no original)
        # if hasattr(self.model, 'feature_importances_'):
        #     important_features_exit = self._get_top_n_important_features(top_n=3) # Renomeado
        #     deterioration_score_val = self._calculate_feature_deterioration_score( # Renomeado
        #         important_features_exit, indic, open_position
        #     )
        #     if deterioration_score_val > self.parameters.get('max_feature_deterioration_exit', 0.7):
        #         return ExitSignal(position_id_to_close=open_position.id, reason="Features importantes deterioraram significativamente.")

        return None # Nenhuma condição de saída ML atingida

    # _calculate_feature_deterioration_score: Implementação original mantida como placeholder.
    # Requer que 'entry_features' sejam armazenadas no metadata da posição.
    # def _calculate_feature_deterioration_score(...)

    async def train_model_periodically(self, historical_ticks_df: pd.DataFrame): # Renomeado
        """Treina ou retreina o modelo ML com novos dados históricos."""
        self.logger.info(f"Iniciando treinamento/retreinamento do modelo ML '{self.name}'...")
        if historical_ticks_df.empty or len(historical_ticks_df) < self.parameters['min_samples_for_training']:
            self.logger.warning(f"Dados históricos insuficientes ({len(historical_ticks_df)}) para treinar modelo. Mínimo: {self.parameters['min_samples_for_training']}.")
            return

        try:
            # 1. Engenharia de Features
            #    Certificar que historical_ticks_df tenha colunas 'mid', 'high', 'low', 'close', 'volume', 'timestamp'
            #    A função _calculate_features da classe MarketRegime foi usada como exemplo,
            #    mas esta estratégia deve ter sua própria função de cálculo de features para X.
            
            # Exemplo de como preparar X (features) e y (labels)
            # Esta parte é a mais crítica e específica do problema de ML.
            features_for_training_df = pd.DataFrame() # Dataframe para todas as features
            # Popular com as mesmas features usadas em calculate_indicators
            # ... (lógica para iterar sobre historical_ticks_df, calcular features em janelas e criar X) ...

            # Criar Labels (y)
            # Exemplo: y[i] = 1 se preço N ticks após X[i] subiu, -1 se caiu, 0 se manteve.
            price_series_for_labels = historical_ticks_df['mid'].values # Usar mid para labels
            labels_list = [] # Renomeado
            horizon = self.parameters['prediction_horizon_ticks']
            # Threshold para definir subida/descida (ex: em pips ou %)
            # pip_size_label = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01
            # move_threshold_price = 2 * pip_size_label # Ex: 2 pips

            for i in range(len(price_series_for_labels) - horizon):
                # features_at_i = ... (calcular features para o tick i)
                # X_list.append(features_at_i)
                
                price_now = price_series_for_labels[i]
                price_future = price_series_for_labels[i + horizon]
                price_diff = price_future - price_now

                # if price_diff > move_threshold_price: labels_list.append(1)  # Buy
                # elif price_diff < -move_threshold_price: labels_list.append(-1) # Sell
                # else: labels_list.append(0) # Hold
            # Esta parte de geração de X,y precisa ser feita *antes* de chamar train_model_periodically,
            # ou esta função precisa de uma sub-função robusta para isso.
            # Por ora, vou assumir que X_train_df e y_train_series são pré-processados e passados.
            # O código original chamava calculate_indicators em um loop, o que é ineficiente para treino.

            # ===== Placeholder para Geração de X e y (DEVE SER IMPLEMENTADO CORRETAMENTE) =====
            if not hasattr(self, '_generate_training_features_and_labels'):
                 self.logger.error("Função _generate_training_features_and_labels não implementada. Treinamento ML cancelado.")
                 return
            
            X_train_df, y_train_series = self._generate_training_features_and_labels(historical_ticks_df)
            if X_train_df.empty or y_train_series.empty:
                self.logger.error("Geração de features/labels para treino ML falhou ou resultou em dados vazios.")
                return
            # ===================================================================================


            # Usar as features que foram definidas durante o primeiro cálculo ou carregamento
            if not self.feature_names: # Se feature_names ainda não foi definido
                self.feature_names = X_train_df.columns.tolist()
            else: # Garantir que X_train_df tenha as colunas corretas e na ordem certa
                X_train_df = X_train_df[self.feature_names]


            X_train_arr = X_train_df.values # Renomeado
            y_train_arr = y_train_series.values # Renomeado


            # (Opcional) Balancear classes se necessário (ex: usando SMOTE ou undersampling)
            # X_balanced, y_balanced = self._balance_training_classes(X_train_arr, y_train_arr)

            # Normalizar features
            X_scaled_train = self.scaler.fit_transform(X_train_arr) # fit_transform no treino

            # Treinar o modelo
            self.model = GradientBoostingClassifier(
                n_estimators=self.parameters.get('gb_n_estimators', 100),
                learning_rate=self.parameters.get('gb_learning_rate', 0.1),
                max_depth=self.parameters.get('gb_max_depth', 5),
                min_samples_split=self.parameters.get('gb_min_samples_split', 20),
                min_samples_leaf=self.parameters.get('gb_min_samples_leaf', 10),
                subsample=self.parameters.get('gb_subsample', 0.8),
                random_state=42 # Para reprodutibilidade
            )
            self.model.fit(X_scaled_train, y_train_arr) # Usar y_train_arr
            self.is_trained = True

            # Salvar modelo, scaler e features
            joblib.dump(self.model, self.model_file_path)
            joblib.dump(self.scaler, self.scaler_file_path)
            with open(self.features_file_path, 'w') as f:
                json.dump(self.feature_names, f)

            self.logger.info(f"Modelo ML '{self.name}' treinado/retreinado com {len(X_train_arr)} amostras e salvo.")
            # Opcional: Logar acurácia no treino ou outras métricas
            # train_accuracy = self.model.score(X_scaled_train, y_train_arr)
            # self.logger.info(f"Acurácia do modelo no conjunto de treino: {train_accuracy:.2%}")

        except Exception as e_train: # Renomeado
            self.logger.exception(f"Erro durante o treinamento/retreinamento do modelo ML '{self.name}':")
            self.is_trained = False # Marcar como não treinado em caso de erro


    def _generate_training_features_and_labels(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Placeholder: Esta função DEVE ser implementada para gerar features (X) e labels (y)
        a partir dos dados históricos para o treinamento do modelo ML.
        É uma parte crucial e específica do design da sua estratégia de ML.
        """
        self.logger.warning("_generate_training_features_and_labels é um placeholder e precisa ser implementado!")
        # Exemplo muito simplificado:
        # features_df = self._calculate_features_for_dataframe(historical_data) # Um método que calcula todas as features em todo o DF
        # labels = self._create_labels_for_training(historical_data['mid'], self.parameters['prediction_horizon_ticks'])
        # Alinhar features e labels, remover NaNs
        # combined = pd.concat([features_df, labels.rename('label')], axis=1).dropna()
        # return combined.drop('label', axis=1), combined['label']
        return pd.DataFrame(), pd.Series(dtype=int)


    # _balance_training_classes foi removido, pois a implementação depende da biblioteca (ex: imblearn)
    # e da estratégia de balanceamento escolhida.