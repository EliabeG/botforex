# core/market_regime.py
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime, timedelta
import joblib

from config.settings import REGIME_CONFIG
from utils.logger import setup_logger

logger = setup_logger("market_regime")

class MarketRegime:
    """Enum para regimes de mercado"""
    TREND = "trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLUME = "low_volume"

class MarketRegimeDetector:
    """Detector híbrido de regime de mercado (ML + Regras)"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.last_regime = None
        self.regime_history = []
        
    async def train(self, historical_data: pd.DataFrame):
        """Treina o modelo com dados históricos"""
        try:
            logger.info("Treinando detector de regime...")
            
            # Calcular features
            features_df = self._calculate_features(historical_data)
            
            # Rotular regimes usando regras
            labels = self._label_regimes(features_df)
            
            # Preparar dados
            X = features_df.values
            y = labels
            
            # Normalizar features
            X_scaled = self.scaler.fit_transform(X)
            
            # Treinar modelo
            self.rf_model.fit(X_scaled, y)
            
            # Salvar nomes das features
            self.feature_names = features_df.columns.tolist()
            
            self.is_trained = True
            
            # Avaliar importância das features
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Top 5 features mais importantes:\n{feature_importance.head()}")
            
            # Salvar modelo
            await self._save_model()
            
        except Exception as e:
            logger.error(f"Erro ao treinar detector de regime: {e}")
            raise
    
    async def detect_regime(self, ticks: pd.DataFrame) -> Tuple[str, float]:
        """Detecta regime atual do mercado"""
        try:
            # Calcular features
            features = self._calculate_features(ticks)
            
            # Detectar por regras
            rule_regime = self._detect_by_rules(features)
            rule_confidence = 0.8  # Confiança base das regras
            
            # Se modelo treinado, usar ensemble
            if self.is_trained and len(features) > 0:
                # Preparar features
                X = features[self.feature_names].iloc[-1:].values
                X_scaled = self.scaler.transform(X)
                
                # Predição do modelo
                ml_proba = self.rf_model.predict_proba(X_scaled)[0]
                ml_regime = self.rf_model.classes_[np.argmax(ml_proba)]
                ml_confidence = np.max(ml_proba)
                
                # Soft voting entre regras e ML
                if ml_regime == rule_regime:
                    # Concordância aumenta confiança
                    final_regime = ml_regime
                    final_confidence = (ml_confidence + rule_confidence) / 2
                else:
                    # Usar o de maior confiança
                    if ml_confidence > rule_confidence:
                        final_regime = ml_regime
                        final_confidence = ml_confidence * 0.9  # Penalizar discordância
                    else:
                        final_regime = rule_regime
                        final_confidence = rule_confidence * 0.9
            else:
                # Usar apenas regras
                final_regime = rule_regime
                final_confidence = rule_confidence
            
            # Atualizar histórico
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': final_regime,
                'confidence': final_confidence
            })
            
            # Manter apenas últimas 1000 detecções
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            self.last_regime = final_regime
            
            return final_regime, final_confidence
            
        except Exception as e:
            logger.error(f"Erro ao detectar regime: {e}")
            # Retornar regime anterior ou padrão
            return self.last_regime or MarketRegime.RANGE, 0.5
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para detecção de regime"""
        features = pd.DataFrame(index=data.index)
        
        # Preços
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Features de volatilidade
        features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        features['atr_pct'] = features['atr'] / close * 100
        features['volatility_20'] = pd.Series(close).pct_change().rolling(20).std() * np.sqrt(252)
        features['volatility_50'] = pd.Series(close).pct_change().rolling(50).std() * np.sqrt(252)
        
        # Features de tendência
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        features['rsi'] = talib.RSI(close, timeperiod=14)
        features['ema_8'] = talib.EMA(close, timeperiod=8)
        features['ema_21'] = talib.EMA(close, timeperiod=21)
        features['ema_50'] = talib.EMA(close, timeperiod=50)
        features['ema_slope_8'] = pd.Series(features['ema_8']).diff(5) / 5
        features['ema_slope_21'] = pd.Series(features['ema_21']).diff(5) / 5
        
        # Regressão linear para tendência
        window = REGIME_CONFIG.TREND_WINDOW
        features['lin_reg_slope'] = pd.Series(close).rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        features['lin_reg_r2'] = pd.Series(close).rolling(window).apply(
            lambda x: self._calculate_r2(x) if len(x) == window else np.nan
        )
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        features['bb_pct'] = (close - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Features de volume e liquidez
        features['volume_ma'] = pd.Series(volume).rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma']
        features['vwap'] = (close * volume).cumsum() / volume.cumsum()
        features['price_vwap_ratio'] = close / features['vwap']
        
        # Features de microestrutura (se disponível)
        if 'bid' in data.columns and 'ask' in data.columns:
            features['spread'] = data['ask'] - data['bid']
            features['spread_pct'] = features['spread'] / data['mid'] * 10000  # em bps
            features['spread_ma'] = features['spread'].rolling(100).mean()
            features['spread_std'] = features['spread'].rolling(100).std()
        
        # Features de momentum
        features['momentum_10'] = talib.MOM(close, timeperiod=10)
        features['momentum_30'] = talib.MOM(close, timeperiod=30)
        features['roc_10'] = talib.ROC(close, timeperiod=10)
        
        # Features cíclicas (hora do dia)
        if 'timestamp' in data.columns:
            features['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Remover NaN
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_r2(self, y: np.ndarray) -> float:
        """Calcula R² da regressão linear"""
        if len(y) < 2:
            return 0
        
        x = np.arange(len(y))
        # Regressão linear
        p = np.polyfit(x, y, 1)
        y_pred = np.polyval(p, x)
        
        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _detect_by_rules(self, features: pd.DataFrame) -> str:
        """Detecta regime usando regras fixas"""
        if len(features) == 0:
            return MarketRegime.RANGE
        
        last_row = features.iloc[-1]
        
        # Verificar alta volatilidade (prioridade)
        if (last_row['atr_pct'] > self._get_percentile(features['atr_pct'], REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE) or
            last_row.get('spread', 0) > REGIME_CONFIG.VOLATILITY_SPREAD_DELTA):
            return MarketRegime.HIGH_VOLATILITY
        
        # Verificar tendência forte
        if (last_row['adx'] > REGIME_CONFIG.TREND_ADX_MIN and
            last_row['lin_reg_r2'] > REGIME_CONFIG.TREND_R2_MIN):
            return MarketRegime.TREND
        
        # Verificar baixo volume
        if last_row.get('volume_ratio', 1) < 0.5:  # Volume < 50% da média
            return MarketRegime.LOW_VOLUME
        
        # Verificar range/lateral
        if (REGIME_CONFIG.RANGE_BB_LOW <= last_row['bb_pct'] <= REGIME_CONFIG.RANGE_BB_HIGH and
            last_row['atr_pct'] < features['atr_pct'].median()):
            return MarketRegime.RANGE
        
        # Padrão
        return MarketRegime.RANGE
    
    def _label_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Rotula regimes para treino supervisionado"""
        labels = []
        
        for i in range(len(features)):
            row = features.iloc[i]
            
            # Aplicar regras para rotular
            if row['atr_pct'] > self._get_percentile(features['atr_pct'], REGIME_CONFIG.VOLATILITY_ATR_PERCENTILE):
                labels.append(MarketRegime.HIGH_VOLATILITY)
            elif row['adx'] > REGIME_CONFIG.TREND_ADX_MIN and row['lin_reg_r2'] > REGIME_CONFIG.TREND_R2_MIN:
                labels.append(MarketRegime.TREND)
            elif row.get('volume_ratio', 1) < 0.5:
                labels.append(MarketRegime.LOW_VOLUME)
            else:
                labels.append(MarketRegime.RANGE)
        
        return np.array(labels)
    
    def _get_percentile(self, series: pd.Series, percentile: int) -> float:
        """Calcula percentil de uma série"""
        return np.percentile(series.dropna(), percentile)
    
    async def _save_model(self):
        """Salva modelo treinado"""
        try:
            joblib.dump(self.rf_model, 'models/regime_detector_rf.pkl')
            joblib.dump(self.scaler, 'models/regime_detector_scaler.pkl')
            logger.info("Modelo de regime salvo")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    async def load_model(self):
        """Carrega modelo previamente treinado"""
        try:
            self.rf_model = joblib.load('models/regime_detector_rf.pkl')
            self.scaler = joblib.load('models/regime_detector_scaler.pkl')
            self.is_trained = True
            logger.info("Modelo de regime carregado")
        except Exception as e:
            logger.warning(f"Não foi possível carregar modelo: {e}")
    
    def get_regime_distribution(self, hours: int = 24) -> Dict[str, float]:
        """Retorna distribuição de regimes nas últimas N horas"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.regime_history if r['timestamp'] > cutoff]
        
        if not recent:
            return {}
        
        regime_counts = {}
        for r in recent:
            regime = r['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(recent)
        return {k: v/total for k, v in regime_counts.items()}