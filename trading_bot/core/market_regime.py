# core/market_regime.py
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime, timedelta, timezone 
import joblib
from pathlib import Path 
from scipy import stats # Adicionado import

from config.settings import REGIME_CONFIG, CONFIG 
from utils.logger import setup_logger

logger = setup_logger("market_regime")

class MarketRegime: 
    """Enum para regimes de mercado"""
    TREND = "trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLUME = "low_volume" 
    UNDEFINED = "undefined" 


class MarketRegimeDetector:
    """Detector hibrido de regime de mercado (ML + Regras)"""

    def __init__(self, model_path: str = CONFIG.MODELS_PATH): 
        self.model_path = Path(model_path) 
        self.model_path.mkdir(parents=True, exist_ok=True) 

        self.rf_model: Optional[RandomForestClassifier] = None 
        self.scaler: Optional[StandardScaler] = None 
        self.is_trained: bool = False
        self.feature_names: List[str] = []
        self.last_regime: str = MarketRegime.UNDEFINED 
        self.regime_history: List[Dict[str, Any]] = [] 

    async def train(self, historical_data: pd.DataFrame):
        """Treina o modelo com dados historicos"""
        if historical_data.empty or len(historical_data) < REGIME_CONFIG.TREND_WINDOW + 50: 
            logger.warning(f"Dados historicos insuficientes para treinar o detector de regime (necessario > {REGIME_CONFIG.TREND_WINDOW + 50} barras, recebido {len(historical_data)}).")
            self.is_trained = False
            return

        try:
            logger.info("Treinando detector de regime...")

            if not isinstance(historical_data.index, pd.DatetimeIndex) and 'timestamp' in historical_data.columns:
                historical_data = historical_data.set_index(pd.to_datetime(historical_data['timestamp'], utc=True))
            elif not isinstance(historical_data.index, pd.DatetimeIndex):
                logger.error("DataFrame historico para treino de regime nao possui indice de datetime ou coluna 'timestamp'.")
                self.is_trained = False
                return


            features_df = self._calculate_features(historical_data.copy()) 
            if features_df.empty:
                logger.error("Calculo de features para treino de regime resultou em DataFrame vazio.")
                self.is_trained = False
                return


            labels = self._label_regimes(features_df)
            if len(labels) != len(features_df):
                logger.error(f"Disparidade de tamanho entre features ({len(features_df)}) e labels ({len(labels)}) para treino de regime.")
                self.is_trained = False
                return

            X = features_df.values 
            y = labels

            if X.shape[0] < 10: 
                logger.warning(f"Amostras insuficientes ({X.shape[0]}) para treinar o modelo de regime apos processamento.")
                self.is_trained = False
                return

            self.rf_model = RandomForestClassifier(
                n_estimators=REGIME_CONFIG.RF_N_ESTIMATORS if hasattr(REGIME_CONFIG, 'RF_N_ESTIMATORS') else 100,
                max_depth=REGIME_CONFIG.RF_MAX_DEPTH if hasattr(REGIME_CONFIG, 'RF_MAX_DEPTH') else 10,
                min_samples_split=REGIME_CONFIG.RF_MIN_SAMPLES_SPLIT if hasattr(REGIME_CONFIG, 'RF_MIN_SAMPLES_SPLIT') else 2,
                min_samples_leaf=REGIME_CONFIG.RF_MIN_SAMPLES_LEAF if hasattr(REGIME_CONFIG, 'RF_MIN_SAMPLES_LEAF') else 1,
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1 
            )
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.rf_model.fit(X_scaled, y)

            self.feature_names = features_df.columns.tolist()
            self.is_trained = True

            if hasattr(self.rf_model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({ 
                    'feature': self.feature_names,
                    'importance': self.rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"Top 5 features de regime mais importantes:\n{feature_importance_df.head().to_string()}")

            await self._save_model()

        except Exception as e:
            logger.exception("Erro ao treinar detector de regime:")
            self.is_trained = False 


    async def detect_regime(self, ticks_df: pd.DataFrame) -> Tuple[str, float]: 
        """Detecta regime atual do mercado com base em um DataFrame de ticks/OHLC recentes."""
        if ticks_df.empty or len(ticks_df) < REGIME_CONFIG.TREND_WINDOW: 
            logger.warning(f"Dados de ticks insuficientes ({len(ticks_df)}) para deteccao de regime. Retornando ultimo conhecido ou UNDEFINED.")
            return self.last_regime, 0.1 


        try:
            if not isinstance(ticks_df.index, pd.DatetimeIndex) and 'timestamp' in ticks_df.columns:
                ticks_df = ticks_df.set_index(pd.to_datetime(ticks_df['timestamp'], utc=True))
            elif not isinstance(ticks_df.index, pd.DatetimeIndex):
                logger.error("DataFrame de ticks para deteccao de regime nao possui indice de datetime ou coluna 'timestamp'.")
                return self.last_regime, 0.1

            data_for_features = ticks_df.copy()
            if 'mid' in data_for_features.columns and 'close' not in data_for_features.columns:
                data_for_features['close'] = data_for_features['mid']
            if 'bid' in data_for_features.columns and 'low' not in data_for_features.columns: 
                data_for_features['low'] = data_for_features['bid']
            if 'ask' in data_for_features.columns and 'high' not in data_for_features.columns: 
                data_for_features['high'] = data_for_features['ask']
            if 'bid_volume' in data_for_features.columns and 'ask_volume' in data_for_features.columns and 'volume' not in data_for_features.columns:
                 data_for_features['volume'] = (data_for_features['bid_volume'] + data_for_features['ask_volume']) / 2
            elif 'volume' not in data_for_features.columns:
                 data_for_features['volume'] = 1.0 


            current_features_df = self._calculate_features(data_for_features) 
            if current_features_df.empty:
                logger.warning("Calculo de features para deteccao de regime resultou em DataFrame vazio. Usando regras de fallback.")
                return self.last_regime, 0.1


            rule_regime = self._detect_by_rules(current_features_df)
            rule_confidence = 0.7 

            ml_predicted_regime = MarketRegime.UNDEFINED 
            ml_regime_confidence = 0.0 

            if self.is_trained and self.rf_model and self.scaler and self.feature_names:
                last_feature_vector_df = current_features_df.iloc[-1:][self.feature_names]
                X_live = last_feature_vector_df.values
                X_live_scaled = self.scaler.transform(X_live) 

                ml_proba_array = self.rf_model.predict_proba(X_live_scaled)[0] 
                ml_predicted_regime = self.rf_model.classes_[np.argmax(ml_proba_array)] 
                ml_regime_confidence = np.max(ml_proba_array) 

                if ml_predicted_regime == rule_regime:
                    final_regime = ml_predicted_regime
                    final_confidence = np.mean([ml_regime_confidence, rule_confidence]) * 1.1 
                elif ml_regime_confidence > rule_confidence + 0.15 : 
                    final_regime = ml_predicted_regime
                    final_confidence = ml_regime_confidence * 0.9 
                else: 
                    final_regime = rule_regime
                    final_confidence = rule_confidence * 0.9 
                final_confidence = min(final_confidence, 1.0) 
            else:
                final_regime = rule_regime
                final_confidence = rule_confidence

            self.regime_history.append({
                'timestamp': datetime.now(timezone.utc), 
                'regime': final_regime,
                'confidence': round(final_confidence, 4), 
                'rule_regime': rule_regime,
                'ml_regime': ml_predicted_regime if self.is_trained else None,
                'ml_confidence': round(ml_regime_confidence, 4) if self.is_trained else None
            })

            self.regime_history = self.regime_history[-1000:] 
            self.last_regime = final_regime

            return final_regime, final_confidence

        except Exception as e:
            logger.exception("Erro ao detectar regime:")
            return self.last_regime or MarketRegime.UNDEFINED, 0.1 


    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para deteccao de regime."""
        if not isinstance(data.index, pd.DatetimeIndex) and 'timestamp' in data.columns:
             features_df = pd.DataFrame(index=pd.to_datetime(data['timestamp'], utc=True))
        elif isinstance(data.index, pd.DatetimeIndex):
             features_df = pd.DataFrame(index=data.index)
        else:
            logger.error("Dados para _calculate_features nao tem indice de tempo nem coluna 'timestamp'.")
            return pd.DataFrame() 

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Coluna '{col}' esperada por TA-Lib nao encontrada em dados para _calculate_features. Usando 'close' ou 1.0 como fallback.")
                if col == 'volume':
                    data[col] = 1.0 
                elif 'close' in data.columns: 
                    data[col] = data['close'] 
                elif 'mid' in data.columns: 
                    data[col] = data['mid']
                    if col == 'close': data['close'] = data['mid'] 
                else: 
                    logger.error(f"Colunas 'close' ou 'mid' necessarias para fallback de OHL em _calculate_features, mas nao encontradas.")
                    return pd.DataFrame()


        open_p = data['open'].astype(float).values
        high_p = data['high'].astype(float).values
        low_p = data['low'].astype(float).values
        close_p = data['close'].astype(float).values
        volume_p = data['volume'].astype(float).values


        features_df['atr'] = talib.ATR(high_p, low_p, close_p, timeperiod=REGIME_CONFIG.VOLATILITY_WINDOW if hasattr(REGIME_CONFIG,'VOLATILITY_WINDOW') else 14) 
        features_df['atr_pct'] = (features_df['atr'] / (close_p + 1e-9)) * 100 
        features_df['volatility_20'] = pd.Series(close_p).pct_change().rolling(20).std() * np.sqrt(252) 
        features_df['volatility_50'] = pd.Series(close_p).pct_change().rolling(50).std() * np.sqrt(252)


        features_df['adx'] = talib.ADX(high_p, low_p, close_p, timeperiod=14)
        features_df['rsi_14'] = talib.RSI(close_p, timeperiod=14) 
        features_df['ema_8'] = talib.EMA(close_p, timeperiod=8)
        features_df['ema_21'] = talib.EMA(close_p, timeperiod=21)
        features_df['ema_50'] = talib.EMA(close_p, timeperiod=50)
        features_df['ema_slope_8'] = pd.Series(features_df['ema_8'].values).diff(5) / 5
        features_df['ema_slope_21'] = pd.Series(features_df['ema_21'].values).diff(5) / 5


        window = REGIME_CONFIG.TREND_WINDOW
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


        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2.0, nbdevdn=2.0) 
        features_df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9) 
        features_df['bb_pct_b'] = (close_p - bb_lower) / ((bb_upper - bb_lower) + 1e-9) 


        if len(volume_p) >= 20:
             features_df['volume_ma_20'] = pd.Series(volume_p).rolling(20).mean() 
             features_df['volume_ratio_vs_ma20'] = volume_p / (features_df['volume_ma_20'] + 1e-9) 
        else:
            features_df['volume_ma_20'] = np.nan
            features_df['volume_ratio_vs_ma20'] = np.nan

        cumulative_price_volume = np.cumsum(close_p * volume_p)
        cumulative_volume = np.cumsum(volume_p)
        features_df['vwap_cumulative'] = cumulative_price_volume / (cumulative_volume + 1e-9) 
        features_df['price_to_vwap_cumulative_ratio'] = close_p / (features_df['vwap_cumulative'] + 1e-9) 


        if 'bid' in data.columns and 'ask' in data.columns and 'mid' in data.columns: 
            features_df['spread_abs'] = data['ask'].astype(float) - data['bid'].astype(float) 
            features_df['spread_bps'] = (features_df['spread_abs'] / (data['mid'].astype(float) + 1e-9)) * 10000 
            if len(features_df['spread_abs']) >= 100:
                 features_df['spread_ma_100'] = features_df['spread_abs'].rolling(100).mean()
                 features_df['spread_std_100'] = features_df['spread_abs'].rolling(100).std()
            else:
                features_df['spread_ma_100'] = np.nan
                features_df['spread_std_100'] = np.nan


        features_df['momentum_10'] = talib.MOM(close_p, timeperiod=10)
        features_df['momentum_30'] = talib.MOM(close_p, timeperiod=30)
        features_df['roc_10'] = talib.ROC(close_p, timeperiod=10)


        features_df['hour_utc'] = features_df.index.hour 
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_utc'] / 24.0)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_utc'] / 24.0)


        features_df = features_df.replace([np.inf, -np.inf], np.nan) 
        features_df = features_df.ffill().bfill() 
        features_df = features_df.fillna(0.0)

        return features_df


    def _calculate_r2(self, y: np.ndarray) -> float:
        """Calcula R2 da regressao linear. y deve ser um array 1D."""
        if len(y) < 2 or pd.Series(y).isnull().all(): 
            return 0.0

        x = np.arange(len(y))
        finite_mask = np.isfinite(y)
        y_finite = y[finite_mask]
        x_finite = x[finite_mask]

        if len(y_finite) < 2: 
            return 0.0

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_finite, y_finite) 
            return r_value**2
        except ValueError: 
            return 0.0


    def _detect_by_rules(self, features_df: pd.DataFrame) -> str: 
        """Detecta regime usando regras fixas (usa a ultima linha de features_df)."""
        if features_df.empty:
            return MarketRegime.UNDEFINED

        last_row = features_df.iloc[-1]

        atr_pct_val = last_row.get('atr_pct', 0.0)
        spread_bps_val = last_row.get('spread_bps', 0.0)
        adx_val = last_row.get('adx', 0.0)
        lin_reg_r2_val = last_row.get('lin_reg_r2', 0.0)
        volume_ratio_val = last_row.get('volume_ratio_vs_ma20', 1.0)
        bb_pct_b_val = last_row.get('bb_pct_b', 0.5)


        atr_pct_q_volatility = self._get_percentile(features_df['atr_pct'].dropna(), REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE) \
            if not features_df['atr_pct'].dropna().empty else REGIME_CONFIG.VOLATILITY_SPREAD_DELTA 
        spread_threshold = REGIME_CONFIG.VOLATILITY_SPREAD_DELTA 

        if (atr_pct_val > atr_pct_q_volatility or spread_bps_val > spread_threshold): 
            return MarketRegime.HIGH_VOLATILITY

        if (adx_val > REGIME_CONFIG.TREND_ADX_MIN and lin_reg_r2_val > REGIME_CONFIG.TREND_R2_MIN): 
            return MarketRegime.TREND

        if volume_ratio_val < 0.5:
            return MarketRegime.LOW_VOLUME

        median_atr_pct_for_range = features_df['atr_pct'].median() if not features_df['atr_pct'].dropna().empty else atr_pct_q_volatility
        if (REGIME_CONFIG.RANGE_BB_LOW <= bb_pct_b_val <= REGIME_CONFIG.RANGE_BB_HIGH and
            atr_pct_val < median_atr_pct_for_range):
            return MarketRegime.RANGE

        return MarketRegime.RANGE 


    def _label_regimes(self, features_df: pd.DataFrame) -> np.ndarray: 
        """Rotula regimes para treino supervisionado"""
        labels = []
        if features_df.empty:
            return np.array(labels)

        atr_pct_q_volatility = self._get_percentile(features_df['atr_pct'].dropna(), REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE) \
            if not features_df['atr_pct'].dropna().empty else float('inf')
        median_atr_pct = features_df['atr_pct'].median() \
            if not features_df['atr_pct'].dropna().empty else 0.0


        for i in range(len(features_df)):
            row = features_df.iloc[i]
            atr_pct_val = row.get('atr_pct', 0.0)
            adx_val = row.get('adx', 0.0)
            lin_reg_r2_val = row.get('lin_reg_r2', 0.0)
            volume_ratio_val = row.get('volume_ratio_vs_ma20', 1.0)
            bb_pct_b_val = row.get('bb_pct_b', 0.5)

            if atr_pct_val > atr_pct_q_volatility:
                labels.append(MarketRegime.HIGH_VOLATILITY)
            elif adx_val > REGIME_CONFIG.TREND_ADX_MIN and lin_reg_r2_val > REGIME_CONFIG.TREND_R2_MIN:
                labels.append(MarketRegime.TREND)
            elif volume_ratio_val < 0.5:
                labels.append(MarketRegime.LOW_VOLUME)
            elif (REGIME_CONFIG.RANGE_BB_LOW <= bb_pct_b_val <= REGIME_CONFIG.RANGE_BB_HIGH and
                  atr_pct_val < median_atr_pct):
                labels.append(MarketRegime.RANGE)
            else:
                labels.append(MarketRegime.RANGE) 

        return np.array(labels)


    def _get_percentile(self, series: pd.Series, percentile: int) -> float:
        """Calcula percentil de uma serie, tratando NaNs e series vazias."""
        cleaned_series = series.dropna()
        if cleaned_series.empty:
            return 0.0 
        try:
            return np.percentile(cleaned_series, percentile)
        except IndexError: 
            return cleaned_series.median() if not cleaned_series.empty else 0.0


    async def _save_model(self):
        """Salva modelo treinado e scaler."""
        if not self.is_trained or not self.rf_model or not self.scaler:
            logger.warning("Tentativa de salvar modelo de regime nao treinado ou incompleto.")
            return
        try:
            model_file = self.model_path / "regime_detector_rf.pkl"
            scaler_file = self.model_path / "regime_detector_scaler.pkl"
            features_file = self.model_path / "regime_detector_features.json" 

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
                logger.warning(f"Arquivos de modelo de regime nao encontrados em {self.model_path}. O modelo nao sera carregado.")
                self.is_trained = False
                return

            self.rf_model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)

            self.is_trained = True
            logger.info(f"Modelo de regime carregado de: {self.model_path}")
        except Exception as e:
            logger.exception("Nao foi possivel carregar modelo de regime:")
            self.is_trained = False 


    def get_regime_distribution(self, hours: int = 24) -> Dict[str, float]:
        """Retorna distribuicao de regimes nas ultimas N horas"""
        if not self.regime_history:
            return {MarketRegime.UNDEFINED: 1.0}

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours) 
        recent_period_history = [r for r in self.regime_history if r['timestamp'] >= cutoff_time]

        if not recent_period_history:
            last_known = self.regime_history[-1]['regime'] if self.regime_history else MarketRegime.UNDEFINED
            return {last_known: 1.0}

        regime_counts: Dict[str, int] = defaultdict(int) 
        for r_event in recent_period_history: 
            regime_counts[r_event['regime']] += 1

        total_recent_events = len(recent_period_history)
        return {k: round(v / total_recent_events, 4) for k, v in regime_counts.items()}