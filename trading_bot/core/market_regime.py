# core/market_regime.py
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any # Adicionado Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime, timedelta, timezone # Adicionado timezone
import joblib
from pathlib import Path # Adicionado para manipulação de caminhos

from config.settings import REGIME_CONFIG, CONFIG # Adicionado CONFIG para caminhos
from utils.logger import setup_logger

logger = setup_logger("market_regime")

class MarketRegime: # Mantido como classe com atributos string
    """Enum para regimes de mercado"""
    TREND = "trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLUME = "low_volume" # Ou LOW_LIQUIDITY
    UNDEFINED = "undefined" # Adicionado para estado inicial ou incerto


class MarketRegimeDetector:
    """Detector híbrido de regime de mercado (ML + Regras)"""

    def __init__(self, model_path: str = CONFIG.MODELS_PATH): # Usar CONFIG para caminho
        self.model_path = Path(model_path) # Converter para Path
        self.model_path.mkdir(parents=True, exist_ok=True) # Garantir que o diretório exista

        self.rf_model: Optional[RandomForestClassifier] = None # Tipagem
        self.scaler: Optional[StandardScaler] = None # Tipagem
        self.is_trained: bool = False
        self.feature_names: List[str] = []
        self.last_regime: str = MarketRegime.UNDEFINED # Inicializar com UNDEFINED
        self.regime_history: List[Dict[str, Any]] = [] # Tipagem

        # Tentar carregar modelo na inicialização
        # asyncio.create_task(self.load_model()) # Não é ideal chamar async de __init__ síncrono.
        # O carregamento deve ser chamado explicitamente por um método async ou na inicialização do Orchestrator.

    async def train(self, historical_data: pd.DataFrame):
        """Treina o modelo com dados históricos"""
        if historical_data.empty or len(historical_data) < REGIME_CONFIG.TREND_WINDOW + 50: # Adicionar verificação de tamanho
            logger.warning(f"Dados históricos insuficientes para treinar o detector de regime (necessário > {REGIME_CONFIG.TREND_WINDOW + 50} barras, recebido {len(historical_data)}).")
            self.is_trained = False
            return

        try:
            logger.info("Treinando detector de regime...")

            # Calcular features
            # Assegurar que 'historical_data' tenha as colunas 'open', 'high', 'low', 'close', 'volume'
            # e um índice de datetime ou uma coluna 'timestamp'.
            # A função _calculate_features espera um DataFrame com colunas específicas.
            # Se historical_data for uma lista de TickData, precisa ser convertida.
            # Assumindo que historical_data já é um DataFrame OHLCV com 'timestamp'.
            if not isinstance(historical_data.index, pd.DatetimeIndex) and 'timestamp' in historical_data.columns:
                historical_data = historical_data.set_index(pd.to_datetime(historical_data['timestamp'], utc=True))
            elif not isinstance(historical_data.index, pd.DatetimeIndex):
                logger.error("DataFrame histórico para treino de regime não possui índice de datetime ou coluna 'timestamp'.")
                self.is_trained = False
                return


            features_df = self._calculate_features(historical_data.copy()) # Usar cópia para evitar SettingWithCopyWarning
            if features_df.empty:
                logger.error("Cálculo de features para treino de regime resultou em DataFrame vazio.")
                self.is_trained = False
                return


            # Rotular regimes usando regras
            labels = self._label_regimes(features_df)
            if len(labels) != len(features_df):
                logger.error(f"Disparidade de tamanho entre features ({len(features_df)}) e labels ({len(labels)}) para treino de regime.")
                self.is_trained = False
                return

            # Remover NaNs após cálculo de features e antes de treinar
            # (features_df já é tratado para NaNs em _calculate_features)
            # X = features_df.dropna().values
            # y = labels[features_df.index.isin(features_df.dropna().index)] # Alinhar labels com features não-NaN

            X = features_df.values # Assumindo que _calculate_features já trata NaNs
            y = labels


            if X.shape[0] < 10: # Mínimo de amostras para treinar
                logger.warning(f"Amostras insuficientes ({X.shape[0]}) para treinar o modelo de regime após processamento.")
                self.is_trained = False
                return

            # Inicializar modelo e scaler aqui para garantir que sejam frescos para o treino
            self.rf_model = RandomForestClassifier(
                n_estimators=REGIME_CONFIG.RF_N_ESTIMATORS if hasattr(REGIME_CONFIG, 'RF_N_ESTIMATORS') else 100,
                max_depth=REGIME_CONFIG.RF_MAX_DEPTH if hasattr(REGIME_CONFIG, 'RF_MAX_DEPTH') else 10,
                min_samples_split=REGIME_CONFIG.RF_MIN_SAMPLES_SPLIT if hasattr(REGIME_CONFIG, 'RF_MIN_SAMPLES_SPLIT') else 2,
                min_samples_leaf=REGIME_CONFIG.RF_MIN_SAMPLES_LEAF if hasattr(REGIME_CONFIG, 'RF_MIN_SAMPLES_LEAF') else 1,
                class_weight='balanced', # Bom para classes desbalanceadas
                random_state=42,
                n_jobs=-1 # Usar todos os cores disponíveis
            )
            self.scaler = StandardScaler()

            # Normalizar features
            X_scaled = self.scaler.fit_transform(X)

            # Treinar modelo
            self.rf_model.fit(X_scaled, y)

            self.feature_names = features_df.columns.tolist()
            self.is_trained = True

            if hasattr(self.rf_model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({ # Renomeado
                    'feature': self.feature_names,
                    'importance': self.rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"Top 5 features de regime mais importantes:\n{feature_importance_df.head().to_string()}")

            await self._save_model()

        except Exception as e:
            logger.exception("Erro ao treinar detector de regime:")
            self.is_trained = False # Marcar como não treinado em caso de erro
            # Não relançar aqui para permitir que o bot continue com detecção baseada em regras se o treino falhar.


    async def detect_regime(self, ticks_df: pd.DataFrame) -> Tuple[str, float]: # Renomeado para clareza
        """Detecta regime atual do mercado com base em um DataFrame de ticks/OHLC recentes."""
        if ticks_df.empty or len(ticks_df) < REGIME_CONFIG.TREND_WINDOW: # Usar um lookback mínimo razoável
            logger.warning(f"Dados de ticks insuficientes ({len(ticks_df)}) para detecção de regime. Retornando último conhecido ou UNDEFINED.")
            return self.last_regime, 0.1 # Baixa confiança


        try:
            # Calcular features para os ticks_df fornecidos
            # _calculate_features espera colunas 'open', 'high', 'low', 'close', 'volume'
            # Se ticks_df for de ticks, precisa ser agregado para OHLCV primeiro,
            # ou _calculate_features precisa ser adaptado.
            # Assumindo que ticks_df é um DataFrame OHLCV com 'timestamp' como índice ou coluna.

            if not isinstance(ticks_df.index, pd.DatetimeIndex) and 'timestamp' in ticks_df.columns:
                ticks_df = ticks_df.set_index(pd.to_datetime(ticks_df['timestamp'], utc=True))
            elif not isinstance(ticks_df.index, pd.DatetimeIndex):
                logger.error("DataFrame de ticks para detecção de regime não possui índice de datetime ou coluna 'timestamp'.")
                return self.last_regime, 0.1

            # Garantir que as colunas esperadas por TA-Lib existam
            # (TA-Lib geralmente espera 'open', 'high', 'low', 'close', 'volume')
            # Se 'ticks_df' vier de TickData, pode ter 'mid' em vez de 'close', etc.
            # Vamos mapear 'mid' para 'close' se 'close' não existir.
            data_for_features = ticks_df.copy()
            if 'mid' in data_for_features.columns and 'close' not in data_for_features.columns:
                data_for_features['close'] = data_for_features['mid']
            if 'bid' in data_for_features.columns and 'low' not in data_for_features.columns: # Aproximação
                data_for_features['low'] = data_for_features['bid']
            if 'ask' in data_for_features.columns and 'high' not in data_for_features.columns: # Aproximação
                data_for_features['high'] = data_for_features['ask']
            if 'bid_volume' in data_for_features.columns and 'ask_volume' in data_for_features.columns and 'volume' not in data_for_features.columns:
                 data_for_features['volume'] = (data_for_features['bid_volume'] + data_for_features['ask_volume']) / 2
            elif 'volume' not in data_for_features.columns:
                 data_for_features['volume'] = 1.0 # Placeholder se volume não estiver disponível


            current_features_df = self._calculate_features(data_for_features) # Renomeado
            if current_features_df.empty:
                logger.warning("Cálculo de features para detecção de regime resultou em DataFrame vazio. Usando regras de fallback.")
                # Fallback para detecção apenas por regras usando a última linha de `data_for_features` se possível
                # Isso é simplificado; idealmente, _detect_by_rules também precisaria de um DataFrame de features.
                # Por ora, retornamos o último regime conhecido ou indefinido.
                return self.last_regime, 0.1


            # Detectar por regras (usando a última linha das features calculadas)
            rule_regime = self._detect_by_rules(current_features_df)
            rule_confidence = 0.7 # Confiança base das regras (ajustada)


            if self.is_trained and self.rf_model and self.scaler and self.feature_names:
                # Preparar features para o modelo ML (última linha)
                # Garantir que as colunas estejam na mesma ordem do treino
                last_feature_vector_df = current_features_df.iloc[-1:][self.feature_names]
                X_live = last_feature_vector_df.values
                X_live_scaled = self.scaler.transform(X_live) # Usar o scaler treinado

                ml_proba_array = self.rf_model.predict_proba(X_live_scaled)[0] # Renomeado
                ml_predicted_regime = self.rf_model.classes_[np.argmax(ml_proba_array)] # Renomeado
                ml_regime_confidence = np.max(ml_proba_array) # Renomeado

                # Ensemble simples (poderia ser mais sofisticado)
                if ml_predicted_regime == rule_regime:
                    final_regime = ml_predicted_regime
                    final_confidence = np.mean([ml_regime_confidence, rule_confidence]) * 1.1 # Boost por concordância
                elif ml_regime_confidence > rule_confidence + 0.15 : # ML tem confiança significativamente maior
                    final_regime = ml_predicted_regime
                    final_confidence = ml_regime_confidence * 0.9 # Pequena penalidade por discordância
                else: # Regra tem mais confiança ou são próximas
                    final_regime = rule_regime
                    final_confidence = rule_confidence * 0.9 # Pequena penalidade por discordância
                final_confidence = min(final_confidence, 1.0) # Limitar a 1.0
            else:
                final_regime = rule_regime
                final_confidence = rule_confidence

            self.regime_history.append({
                'timestamp': datetime.now(timezone.utc), # Usar UTC
                'regime': final_regime,
                'confidence': round(final_confidence, 4), # Arredondar
                'rule_regime': rule_regime,
                'ml_regime': ml_predicted_regime if self.is_trained else None,
                'ml_confidence': round(ml_regime_confidence, 4) if self.is_trained else None
            })

            self.regime_history = self.regime_history[-1000:] # Manter histórico
            self.last_regime = final_regime

            return final_regime, final_confidence

        except Exception as e:
            logger.exception("Erro ao detectar regime:")
            return self.last_regime or MarketRegime.UNDEFINED, 0.1 # Baixa confiança em caso de erro


    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para detecção de regime.
           'data' deve ser um DataFrame OHLCV com índice DatetimeIndex ou coluna 'timestamp'.
        """
        # Garantir que o índice seja DatetimeIndex se 'timestamp' não for coluna
        if not isinstance(data.index, pd.DatetimeIndex) and 'timestamp' in data.columns:
             features_df = pd.DataFrame(index=pd.to_datetime(data['timestamp'], utc=True))
        elif isinstance(data.index, pd.DatetimeIndex):
             features_df = pd.DataFrame(index=data.index)
        else:
            logger.error("Dados para _calculate_features não têm índice de tempo nem coluna 'timestamp'.")
            return pd.DataFrame() # Retornar DataFrame vazio

        # Assegurar que as colunas OHLCV existam com nomes padrão que TA-Lib espera
        # Se os nomes das colunas no 'data' forem diferentes (ex: 'mid', 'bid_vol'), eles precisam ser mapeados.
        # Para simplificar, assumiremos que 'data' já tem 'open', 'high', 'low', 'close', 'volume'
        # ou que foram criados/mapeados antes de chamar esta função (como feito em detect_regime)

        # Usar .values para passar arrays numpy para TA-Lib, pois TA-Lib pode não lidar bem com NaNs no início de Series.
        # No entanto, TA-Lib geralmente lida com isso retornando NaNs, que são tratados depois.
        # O principal é que os DataFrames/Series passados para TA-Lib não sejam totalmente NaN.

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Coluna '{col}' esperada por TA-Lib não encontrada em dados para _calculate_features. Usando 'close' ou 1.0 como fallback.")
                if col == 'volume':
                    data[col] = 1.0 # Volume placeholder
                else:
                    data[col] = data['close'] # Usar 'close' para OHL se ausentes


        open_p = data['open'].astype(float).values
        high_p = data['high'].astype(float).values
        low_p = data['low'].astype(float).values
        close_p = data['close'].astype(float).values
        volume_p = data['volume'].astype(float).values


        # Features de volatilidade
        features_df['atr'] = talib.ATR(high_p, low_p, close_p, timeperiod=REGIME_CONFIG.VOLATILITY_WINDOW if hasattr(REGIME_CONFIG,'VOLATILITY_WINDOW') else 14) # Usar config
        features_df['atr_pct'] = (features_df['atr'] / close_p) * 100 # Cuidado com close_p sendo zero
        features_df['volatility_20'] = pd.Series(close_p).pct_change().rolling(20).std() * np.sqrt(252) # Anualizado para retornos diários
        features_df['volatility_50'] = pd.Series(close_p).pct_change().rolling(50).std() * np.sqrt(252)


        # Features de tendência
        features_df['adx'] = talib.ADX(high_p, low_p, close_p, timeperiod=14)
        features_df['rsi_14'] = talib.RSI(close_p, timeperiod=14) # Renomeado para clareza
        features_df['ema_8'] = talib.EMA(close_p, timeperiod=8)
        features_df['ema_21'] = talib.EMA(close_p, timeperiod=21)
        features_df['ema_50'] = talib.EMA(close_p, timeperiod=50)
        # Usar pd.Series para diff em arrays numpy que vêm do talib
        features_df['ema_slope_8'] = pd.Series(features_df['ema_8'].values).diff(5) / 5
        features_df['ema_slope_21'] = pd.Series(features_df['ema_21'].values).diff(5) / 5


        window = REGIME_CONFIG.TREND_WINDOW
        # Garantir que o rolling não produza todos NaNs se a série for curta
        if len(close_p) >= window:
            features_df['lin_reg_slope'] = pd.Series(close_p).rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window and not np.isnan(x).any() else np.nan, raw=True
            )
            features_df['lin_reg_r2'] = pd.Series(close_p).rolling(window).apply(
                lambda x: self._calculate_r2(x) if len(x) == window and not np.isnan(x).any() else np.nan, raw=True
            )
        else:
            features_df['lin_reg_slope'] = np.nan
            features_df['lin_reg_r2'] = np.nan


        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2.0, nbdevdn=2.0) # nbdev como float
        features_df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9) # Adicionar epsilon para evitar divisão por zero
        features_df['bb_pct_b'] = (close_p - bb_lower) / ((bb_upper - bb_lower) + 1e-9) # Renomeado para %B, adicionar epsilon


        # Features de volume e liquidez
        if len(volume_p) >= 20:
             features_df['volume_ma_20'] = pd.Series(volume_p).rolling(20).mean() # Renomeado
             features_df['volume_ratio_vs_ma20'] = volume_p / (features_df['volume_ma_20'] + 1e-9) # Renomeado
        else:
            features_df['volume_ma_20'] = np.nan
            features_df['volume_ratio_vs_ma20'] = np.nan

        # VWAP (precisa de acesso ao volume original do tick, não agregado)
        # Se 'data' for OHLC, o VWAP calculado aqui é sobre o volume da barra.
        # Para um VWAP de sessão/dia, o cálculo deve ser cumulativo.
        # O VWAP original (close*volume).cumsum() / volume.cumsum() é um VWAP cumulativo.
        # Se a intenção é um VWAP de janela móvel:
        # features_df['vwap_20'] = (pd.Series(close_p * volume_p).rolling(20).sum() / pd.Series(volume_p).rolling(20).sum()).fillna(method='ffill')
        # Mantendo o cumulativo, mas precisa ser resetado em algum ponto (ex: diário)
        cumulative_price_volume = np.cumsum(close_p * volume_p)
        cumulative_volume = np.cumsum(volume_p)
        features_df['vwap_cumulative'] = cumulative_price_volume / (cumulative_volume + 1e-9) # Renomeado
        features_df['price_to_vwap_cumulative_ratio'] = close_p / (features_df['vwap_cumulative'] + 1e-9) # Renomeado


        if 'bid' in data.columns and 'ask' in data.columns: # Se dados de bid/ask estão disponíveis
            features_df['spread_abs'] = data['ask'].astype(float) - data['bid'].astype(float) # Renomeado
            features_df['spread_bps'] = (features_df['spread_abs'] / data['mid'].astype(float)) * 10000 # Renomeado
            if len(features_df['spread_abs']) >= 100:
                 features_df['spread_ma_100'] = features_df['spread_abs'].rolling(100).mean()
                 features_df['spread_std_100'] = features_df['spread_abs'].rolling(100).std()
            else:
                features_df['spread_ma_100'] = np.nan
                features_df['spread_std_100'] = np.nan


        features_df['momentum_10'] = talib.MOM(close_p, timeperiod=10)
        features_df['momentum_30'] = talib.MOM(close_p, timeperiod=30)
        features_df['roc_10'] = talib.ROC(close_p, timeperiod=10)


        # Features cíclicas (hora do dia UTC)
        # O índice de features_df já deve ser DatetimeIndex UTC
        features_df['hour_utc'] = features_df.index.hour # Renomeado
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_utc'] / 24.0)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_utc'] / 24.0)


        # Tratamento de NaNs
        # O ffill preenche NaNs com o último valor válido.
        # O fillna(0) subsequente preenche quaisquer NaNs restantes (geralmente no início) com 0.
        # Isso pode não ser ideal para todas as features (ex: ratios podem se tornar 0 indevidamente).
        # Uma abordagem melhor pode ser dropar linhas com NaNs APÓS todos os cálculos de features,
        # especialmente antes de passar para um modelo ML.
        # features_df = features_df.fillna(method='ffill') # Preencher para frente
        # features_df = features_df.fillna(0)             # Preencher NaNs restantes (início) com 0
        # Melhor: dropar linhas que ainda têm NaNs após ffill, ou usar uma janela de aquecimento maior.
        # Por agora, mantendo a lógica original mas ciente de suas implicações.
        # Para features que são ratios, dividir por (denominador + epsilon) é melhor que fillna(0) posterior.
        features_df = features_df.replace([np.inf, -np.inf], np.nan) # Substituir inf por NaN
        features_df = features_df.ffill().bfill() # Preencher para frente e depois para trás
        # Se ainda houver NaNs (caso toda a coluna seja NaN), preencher com 0, mas isso é um mau sinal.
        features_df = features_df.fillna(0.0)


        return features_df


    def _calculate_r2(self, y: np.ndarray) -> float:
        """Calcula R² da regressão linear. y deve ser um array 1D."""
        if len(y) < 2 or pd.Series(y).isnull().all(): # Adicionado check de nulos
            return 0.0

        x = np.arange(len(y))
        # Remover NaNs de y e alinhar x
        finite_mask = np.isfinite(y)
        y_finite = y[finite_mask]
        x_finite = x[finite_mask]

        if len(y_finite) < 2: # Não é possível calcular R² com menos de 2 pontos
            return 0.0

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_finite, y_finite) # Usar stats.linregress
            return r_value**2
        except ValueError: # Se todos os y_finite forem iguais, por exemplo
            return 0.0


    def _detect_by_rules(self, features_df: pd.DataFrame) -> str: # Renomeado features para features_df
        """Detecta regime usando regras fixas (usa a última linha de features_df)."""
        if features_df.empty:
            return MarketRegime.UNDEFINED

        last_row = features_df.iloc[-1]

        # Verificar alta volatilidade (prioridade)
        # Usar REGIME_CONFIG para limiares
        atr_pct_threshold = self._get_percentile(features_df['atr_pct'].dropna(), REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE) \
            if not features_df['atr_pct'].dropna().empty else REGIME_CONFIG.VOLATILITY_SPREAD_DELTA # Fallback
        spread_threshold = REGIME_CONFIG.VOLATILITY_SPREAD_DELTA # Supondo que isso seja em bps

        if (last_row.get('atr_pct', 0.0) > atr_pct_threshold or
            last_row.get('spread_bps', 0.0) > spread_threshold): # Comparar spread_bps
            return MarketRegime.HIGH_VOLATILITY

        # Verificar tendência forte
        if (last_row.get('adx', 0.0) > REGIME_CONFIG.TREND_ADX_MIN and
            last_row.get('lin_reg_r2', 0.0) > REGIME_CONFIG.TREND_R2_MIN): # Usar lin_reg_r2
            return MarketRegime.TREND

        # Verificar baixo volume
        # volume_ratio_vs_ma20 pode ser NaN se não houver dados suficientes
        if last_row.get('volume_ratio_vs_ma20', 1.0) < 0.5:
            return MarketRegime.LOW_VOLUME

        # Verificar range/lateral
        # bb_pct_b pode ser NaN
        if (REGIME_CONFIG.RANGE_BB_LOW <= last_row.get('bb_pct_b', 0.5) <= REGIME_CONFIG.RANGE_BB_HIGH and
            last_row.get('atr_pct', atr_pct_threshold + 1) < (features_df['atr_pct'].median() if not features_df['atr_pct'].dropna().empty else atr_pct_threshold)):
            return MarketRegime.RANGE


        return MarketRegime.RANGE # Padrão se nenhuma outra condição for atendida


    def _label_regimes(self, features_df: pd.DataFrame) -> np.ndarray: # Renomeado features para features_df
        """Rotula regimes para treino supervisionado"""
        labels = []
        if features_df.empty:
            return np.array(labels)

        # Calcular limiares dinâmicos uma vez para toda a série
        atr_pct_q_volatility = self._get_percentile(features_df['atr_pct'].dropna(), REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE) \
            if not features_df['atr_pct'].dropna().empty else float('inf')
        median_atr_pct = features_df['atr_pct'].median() \
            if not features_df['atr_pct'].dropna().empty else 0.0


        for i in range(len(features_df)):
            row = features_df.iloc[i]
            # Aplicar regras para rotular (similar a _detect_by_rules mas para cada linha)
            if row.get('atr_pct', 0.0) > atr_pct_q_volatility:
                labels.append(MarketRegime.HIGH_VOLATILITY)
            elif row.get('adx', 0.0) > REGIME_CONFIG.TREND_ADX_MIN and row.get('lin_reg_r2', 0.0) > REGIME_CONFIG.TREND_R2_MIN:
                labels.append(MarketRegime.TREND)
            elif row.get('volume_ratio_vs_ma20', 1.0) < 0.5:
                labels.append(MarketRegime.LOW_VOLUME)
            elif (REGIME_CONFIG.RANGE_BB_LOW <= row.get('bb_pct_b', 0.5) <= REGIME_CONFIG.RANGE_BB_HIGH and
                  row.get('atr_pct', median_atr_pct + 1) < median_atr_pct):
                labels.append(MarketRegime.RANGE)
            else:
                labels.append(MarketRegime.RANGE) # Default label

        return np.array(labels)


    def _get_percentile(self, series: pd.Series, percentile: int) -> float:
        """Calcula percentil de uma série, tratando NaNs e séries vazias."""
        cleaned_series = series.dropna()
        if cleaned_series.empty:
            # logger.warning(f"Série vazia para cálculo de percentil {percentile}. Retornando 0.")
            return 0.0 # Ou um valor default apropriado, ou levantar erro
        try:
            return np.percentile(cleaned_series, percentile)
        except IndexError: # Pode acontecer se cleaned_series for muito pequena após dropna
            # logger.warning(f"IndexError ao calcular percentil {percentile} da série. Retornando mediana ou 0.")
            return cleaned_series.median() if not cleaned_series.empty else 0.0


    async def _save_model(self):
        """Salva modelo treinado e scaler."""
        if not self.is_trained or not self.rf_model or not self.scaler:
            logger.warning("Tentativa de salvar modelo de regime não treinado ou incompleto.")
            return
        try:
            # Usar nomes de arquivo da CONFIG ou definir aqui
            model_file = self.model_path / "regime_detector_rf.pkl"
            scaler_file = self.model_path / "regime_detector_scaler.pkl"
            features_file = self.model_path / "regime_detector_features.json" # Salvar nomes das features

            joblib.dump(self.rf_model, model_file)
            joblib.dump(self.scaler, scaler_file)
            with open(features_file, 'w') as f:
                json.dump(self.feature_names, f)

            logger.info(f"Modelo de regime salvo em: {self.model_path}")
        except Exception as e:
            logger.exception("Erro ao salvar modelo de regime:")


    async def load_model(self):
        """Carrega modelo previamente treinado e scaler."""
        try:
            model_file = self.model_path / "regime_detector_rf.pkl"
            scaler_file = self.model_path / "regime_detector_scaler.pkl"
            features_file = self.model_path / "regime_detector_features.json"

            if not model_file.exists() or not scaler_file.exists() or not features_file.exists():
                logger.warning(f"Arquivos de modelo de regime não encontrados em {self.model_path}. O modelo não será carregado.")
                self.is_trained = False
                return

            self.rf_model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)

            self.is_trained = True
            logger.info(f"Modelo de regime carregado de: {self.model_path}")
        except Exception as e:
            logger.exception("Não foi possível carregar modelo de regime:")
            self.is_trained = False # Garantir que is_trained seja False


    def get_regime_distribution(self, hours: int = 24) -> Dict[str, float]:
        """Retorna distribuição de regimes nas últimas N horas"""
        if not self.regime_history:
            return {MarketRegime.UNDEFINED: 1.0}

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours) # Usar UTC
        # Filtrar histórico recente
        recent_period_history = [r for r in self.regime_history if r['timestamp'] >= cutoff_time]

        if not recent_period_history:
            # Se não houver histórico recente, retornar o último regime conhecido
            last_known = self.regime_history[-1]['regime'] if self.regime_history else MarketRegime.UNDEFINED
            return {last_known: 1.0}


        regime_counts: Dict[str, int] = defaultdict(int) # Usar defaultdict
        for r_event in recent_period_history: # Renomeado r para r_event
            regime_counts[r_event['regime']] += 1

        total_recent_events = len(recent_period_history)
        return {k: round(v / total_recent_events, 4) for k, v in regime_counts.items()}