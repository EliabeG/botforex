# strategies/ml_predictive/gradient_boost_strategy.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import talib

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("gradient_boost_strategy")

class GradientBoostStrategy(BaseStrategy):
    """
    Estratégia de ML usando Gradient Boosting
    
    Features:
    - Indicadores técnicos
    - Microestrutura de mercado
    - Padrões de preço
    - Regime de mercado
    """
    
    def __init__(self):
        super().__init__("GradientBoostSHAP")
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.RANGE]
        self.min_time_between_signals = 300  # 5 minutos
        
        # Modelo
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Buffers
        self.prediction_history = []
        self.feature_importance = {}
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Model parameters
            'prediction_horizon': 5,          # Prever próximos 5 ticks
            'min_probability': 0.65,          # Probabilidade mínima
            'use_ensemble': True,             # Usar ensemble de modelos
            'retrain_hours': 24,              # Retreinar a cada 24h
            
            # Features config
            'use_technical': True,
            'use_microstructure': True,
            'use_patterns': True,
            'feature_window': 100,            # Janela para features
            
            # Risk
            'confidence_multiplier': 1.5,     # Multiplicador baseado em confiança
            'atr_multiplier_sl': 2.0,
            'risk_reward_ratio': 2.0,
            
            # Filters
            'min_feature_importance': 0.02,   # Importância mínima
            'max_correlation': 0.95,          # Correlação máxima entre features
            'min_samples': 1000,              # Samples mínimos para treinar
        }
    
    async def initialize(self):
        """Inicializa estratégia e carrega modelo"""
        await super().initialize()
        
        # Tentar carregar modelo existente
        try:
            self.model = joblib.load(f'models/{self.name}_model.pkl')
            self.scaler = joblib.load(f'models/{self.name}_scaler.pkl')
            self.feature_names = joblib.load(f'models/{self.name}_features.pkl')
            self.is_trained = True
            logger.info("Modelo ML carregado com sucesso")
        except:
            logger.info("Modelo não encontrado, será treinado com dados históricos")
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula features para o modelo"""
        try:
            ticks = market_context.get('recent_ticks', [])
            
            if len(ticks) < self.parameters['feature_window']:
                return {}
            
            # Extrair dados básicos
            closes = np.array([t.mid for t in ticks])
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            volumes = np.array([(t.bid_volume + t.ask_volume) / 2 for t in ticks])
            spreads = np.array([t.spread for t in ticks])
            
            features = {}
            
            # Features técnicas
            if self.parameters['use_technical']:
                features.update(self._calculate_technical_features(closes, highs, lows, volumes))
            
            # Features de microestrutura
            if self.parameters['use_microstructure']:
                features.update(self._calculate_microstructure_features(ticks, spreads))
            
            # Features de padrões
            if self.parameters['use_patterns']:
                features.update(self._calculate_pattern_features(closes, highs, lows))
            
            # Features de contexto
            features.update(self._calculate_context_features(market_context))
            
            # Preparar para predição
            if self.is_trained:
                prediction = self._make_prediction(features)
                features['ml_prediction'] = prediction['class']
                features['ml_probability'] = prediction['probability']
                features['ml_confidence'] = prediction['confidence']
            else:
                features['ml_prediction'] = 0
                features['ml_probability'] = 0.5
                features['ml_confidence'] = 0
            
            features['current_price'] = closes[-1]
            features['spread'] = spreads[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao calcular features: {e}")
            return {}
    
    def _calculate_technical_features(self, closes: np.ndarray, highs: np.ndarray,
                                    lows: np.ndarray, volumes: np.ndarray) -> Dict:
        """Calcula features técnicas"""
        features = {}
        
        # Médias móveis
        features['sma_5'] = talib.SMA(closes, 5)[-1]
        features['sma_20'] = talib.SMA(closes, 20)[-1]
        features['ema_9'] = talib.EMA(closes, 9)[-1]
        
        # Momentum
        features['rsi_14'] = talib.RSI(closes, 14)[-1]
        features['rsi_5'] = talib.RSI(closes, 5)[-1]
        features['mom_10'] = talib.MOM(closes, 10)[-1]
        features['roc_5'] = talib.ROC(closes, 5)[-1]
        
        # Volatilidade
        features['atr_14'] = talib.ATR(highs, lows, closes, 14)[-1]
        features['atr_5'] = talib.ATR(highs, lows, closes, 5)[-1]
        features['natr_14'] = talib.NATR(highs, lows, closes, 14)[-1]
        
        # Volume
        features['adosc'] = talib.ADOSC(highs, lows, closes, volumes, 3, 10)[-1]
        features['obv'] = talib.OBV(closes, volumes)[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(closes, 12, 26, 9)
        features['macd'] = macd[-1]
        features['macd_signal'] = signal[-1]
        features['macd_hist'] = hist[-1]
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(closes, 20, 2, 2)
        features['bb_upper'] = upper[-1]
        features['bb_middle'] = middle[-1]
        features['bb_lower'] = lower[-1]
        features['bb_width'] = upper[-1] - lower[-1]
        features['bb_pct'] = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] > lower[-1] else 0.5
        
        # Stochastic
        slowk, slowd = talib.STOCH(highs, lows, closes, 5, 3, 0, 3, 0)
        features['stoch_k'] = slowk[-1]
        features['stoch_d'] = slowd[-1]
        
        # CCI
        features['cci_14'] = talib.CCI(highs, lows, closes, 14)[-1]
        
        # ADX
        features['adx_14'] = talib.ADX(highs, lows, closes, 14)[-1]
        features['plus_di'] = talib.PLUS_DI(highs, lows, closes, 14)[-1]
        features['minus_di'] = talib.MINUS_DI(highs, lows, closes, 14)[-1]
        
        return features
    
    def _calculate_microstructure_features(self, ticks: List, spreads: np.ndarray) -> Dict:
        """Calcula features de microestrutura"""
        features = {}
        
        # Spread statistics
        features['spread_mean'] = np.mean(spreads[-20:])
        features['spread_std'] = np.std(spreads[-20:])
        features['spread_current'] = spreads[-1]
        features['spread_ratio'] = spreads[-1] / features['spread_mean'] if features['spread_mean'] > 0 else 1
        
        # Bid-Ask imbalance
        bid_volumes = [t.bid_volume for t in ticks[-20:]]
        ask_volumes = [t.ask_volume for t in ticks[-20:]]
        
        features['ba_imbalance'] = (np.mean(bid_volumes) - np.mean(ask_volumes)) / (np.mean(bid_volumes) + np.mean(ask_volumes))
        features['volume_ratio'] = np.mean(bid_volumes) / np.mean(ask_volumes) if np.mean(ask_volumes) > 0 else 1
        
        # Price efficiency
        prices = [t.mid for t in ticks[-50:]]
        returns = np.diff(prices)
        if len(returns) > 1:
            features['autocorr_1'] = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            features['autocorr_1'] = 0
        
        # Tick direction
        tick_directions = []
        for i in range(1, min(20, len(ticks))):
            if ticks[-i].mid > ticks[-i-1].mid:
                tick_directions.append(1)
            elif ticks[-i].mid < ticks[-i-1].mid:
                tick_directions.append(-1)
            else:
                tick_directions.append(0)
        
        features['tick_direction_sum'] = sum(tick_directions)
        features['tick_direction_ratio'] = sum(1 for d in tick_directions if d > 0) / len(tick_directions) if tick_directions else 0.5
        
        return features
    
    def _calculate_pattern_features(self, closes: np.ndarray, highs: np.ndarray,
                                   lows: np.ndarray) -> Dict:
        """Calcula features de padrões"""
        features = {}
        
        # Candle patterns
        features['hammer'] = talib.CDLHAMMER(highs[-10:], lows[-10:], closes[-10:], closes[-10:])[-1]
        features['doji'] = talib.CDLDOJI(highs[-10:], lows[-10:], closes[-10:], closes[-10:])[-1]
        features['engulfing'] = talib.CDLENGULFING(highs[-10:], lows[-10:], closes[-10:], closes[-10:])[-1]
        
        # Support/Resistance
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        features['resistance_distance'] = (max(recent_highs) - closes[-1]) / closes[-1]
        features['support_distance'] = (closes[-1] - min(recent_lows)) / closes[-1]
        
        # Trend strength
        if len(closes) >= 20:
            x = np.arange(20)
            slope, _ = np.polyfit(x, closes[-20:], 1)
            features['trend_slope'] = slope
            
            # R-squared
            y_pred = np.polyval([slope, _], x)
            ss_res = np.sum((closes[-20:] - y_pred) ** 2)
            ss_tot = np.sum((closes[-20:] - np.mean(closes[-20:])) ** 2)
            features['trend_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            features['trend_slope'] = 0
            features['trend_r2'] = 0
        
        # Fractal dimension (simplicado)
        if len(closes) >= 50:
            features['fractal_dim'] = self._calculate_fractal_dimension(closes[-50:])
        else:
            features['fractal_dim'] = 1.5
        
        return features
    
    def _calculate_context_features(self, market_context: Dict) -> Dict:
        """Calcula features de contexto"""
        features = {}
        
        # Regime de mercado
        regime = market_context.get('regime', MarketRegime.RANGE)
        features['regime_trend'] = 1 if regime == MarketRegime.TREND else 0
        features['regime_range'] = 1 if regime == MarketRegime.RANGE else 0
        features['regime_volatile'] = 1 if regime == MarketRegime.HIGH_VOLATILITY else 0
        
        # Hora do dia (ciclical encoding)
        hour = datetime.now().hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Dia da semana
        weekday = datetime.now().weekday()
        features['weekday'] = weekday
        features['is_monday'] = 1 if weekday == 0 else 0
        features['is_friday'] = 1 if weekday == 4 else 0
        
        # Sessão
        if 7 <= hour < 16:
            features['session_london'] = 1
            features['session_ny'] = 0
            features['session_asia'] = 0
        elif 13 <= hour < 22:
            features['session_london'] = 0
            features['session_ny'] = 1
            features['session_asia'] = 0
        else:
            features['session_london'] = 0
            features['session_ny'] = 0
            features['session_asia'] = 1
        
        return features
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calcula dimensão fractal simplificada"""
        n = len(prices)
        if n < 2:
            return 1.5
        
        # Hurst exponent simplificado
        lags = range(2, min(20, n // 2))
        tau = []
        
        for lag in lags:
            pp = prices[lag:] - prices[:-lag]
            tau.append(np.sqrt(np.mean(pp ** 2)))
        
        if len(tau) > 1:
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = reg[0]
            return 2 - hurst
        
        return 1.5
    
    def _make_prediction(self, features: Dict) -> Dict:
        """Faz predição usando o modelo"""
        try:
            # Preparar features na ordem correta
            X = np.array([[features.get(name, 0) for name in self.feature_names]])
            
            # Normalizar
            X_scaled = self.scaler.transform(X)
            
            # Predição
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Calcular confiança
            confidence = max(probabilities) - 0.5  # 0.5 a 1.0 -> 0 a 0.5
            
            return {
                'class': prediction,
                'probability': max(probabilities),
                'confidence': confidence * 2,  # Normalizar para 0-1
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'class': 0,
                'probability': 0.5,
                'confidence': 0,
                'probabilities': [0.5, 0.5]
            }
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal baseado em predição ML"""
        indicators = self.indicators
        
        if not indicators or not self.is_trained:
            return None
        
        # Verificar predição
        if indicators['ml_probability'] < self.parameters['min_probability']:
            return None
        
        prediction = int(indicators['ml_prediction'])
        
        # -1 = sell, 0 = hold, 1 = buy
        if prediction == 0:
            return None
        
        signal_type = 'buy' if prediction == 1 else 'sell'
        
        # Verificar feature importance (se disponível)
        if hasattr(self.model, 'feature_importances_'):
            important_features = self._get_important_features(indicators)
            
            # Verificar se features importantes confirmam
            confirmation_score = self._calculate_confirmation_score(
                important_features,
                indicators,
                signal_type
            )
            
            if confirmation_score < 0.5:
                return None
        
        return self._create_ml_signal(signal_type, indicators, market_context)
    
    def _get_important_features(self, indicators: Dict) -> List[str]:
        """Retorna features mais importantes"""
        if not hasattr(self.model, 'feature_importances_'):
            return self.feature_names[:10]  # Top 10 default
        
        importances = self.model.feature_importances_
        
        # Ordenar por importância
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Filtrar por threshold
        important = [
            name for name, imp in feature_importance
            if imp >= self.parameters['min_feature_importance']
        ]
        
        return important[:10]  # Top 10
    
    def _calculate_confirmation_score(self, important_features: List[str],
                                    indicators: Dict, signal_type: str) -> float:
        """Calcula score de confirmação baseado em features importantes"""
        score = 0
        count = 0
        
        for feature in important_features:
            if feature not in indicators:
                continue
            
            value = indicators[feature]
            
            # Regras simplificadas de confirmação
            if 'rsi' in feature:
                if signal_type == 'buy' and value < 30:
                    score += 1
                elif signal_type == 'sell' and value > 70:
                    score += 1
                count += 1
                
            elif 'macd' in feature:
                if signal_type == 'buy' and value > 0:
                    score += 1
                elif signal_type == 'sell' and value < 0:
                    score += 1
                count += 1
                
            elif 'trend_slope' in feature:
                if signal_type == 'buy' and value > 0:
                    score += 1
                elif signal_type == 'sell' and value < 0:
                    score += 1
                count += 1
        
        return score / count if count > 0 else 0.5
    
    def _create_ml_signal(self, signal_type: str, indicators: Dict,
                         market_context: Dict) -> Signal:
        """Cria sinal ML"""
        price = indicators['current_price']
        
        # ATR para stops (ou calcular se não disponível)
        atr = indicators.get('atr_14', 0.0010)
        
        # Stops baseados em confiança
        confidence_mult = 1 + (indicators['ml_confidence'] * self.parameters['confidence_multiplier'])
        
        if signal_type == 'buy':
            stop_loss = price - (atr * self.parameters['atr_multiplier_sl'] / confidence_mult)
            take_profit = price + (atr * self.parameters['atr_multiplier_sl'] * self.parameters['risk_reward_ratio'])
        else:
            stop_loss = price + (atr * self.parameters['atr_multiplier_sl'] / confidence_mult)
            take_profit = price - (atr * self.parameters['atr_multiplier_sl'] * self.parameters['risk_reward_ratio'])
        
        # Confiança final
        confidence = min(
            0.5 + (indicators['ml_confidence'] * 0.3) + (indicators['ml_probability'] - 0.5),
            0.95
        )
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"ML Prediction {signal_type.upper()} - Prob: {indicators['ml_probability']:.2%}",
            metadata={
                'ml_probability': indicators['ml_probability'],
                'ml_confidence': indicators['ml_confidence'],
                'important_features': self._get_important_features(indicators)[:5]
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída ML"""
        indicators = self.indicators
        
        if not indicators or not self.is_trained:
            return None
        
        # Verificar se predição mudou
        prediction = int(indicators['ml_prediction'])
        
        if position.side == 'buy' and prediction == -1:  # Virou sell
            return ExitSignal(
                position_id=position.id,
                reason="ML prediction reversed to SELL",
                exit_price=current_price
            )
        elif position.side == 'sell' and prediction == 1:  # Virou buy
            return ExitSignal(
                position_id=position.id,
                reason="ML prediction reversed to BUY",
                exit_price=current_price
            )
        
        # Saída se probabilidade cair muito
        if indicators['ml_probability'] < 0.55:  # Abaixo do threshold
            return ExitSignal(
                position_id=position.id,
                reason=f"ML probability dropped to {indicators['ml_probability']:.2%}",
                exit_price=current_price
            )
        
        # Verificar deterioração das features importantes
        important_features = self._get_important_features(indicators)
        deterioration_score = self._calculate_feature_deterioration(
            important_features,
            indicators,
            position
        )
        
        if deterioration_score > 0.7:
            return ExitSignal(
                position_id=position.id,
                reason="Important features deteriorating",
                exit_price=current_price
            )
        
        return None
    
    def _calculate_feature_deterioration(self, features: List[str],
                                       indicators: Dict, position: Position) -> float:
        """Calcula deterioração das features desde entrada"""
        if 'entry_features' not in position.metadata:
            return 0
        
        entry_features = position.metadata['entry_features']
        deterioration = 0
        count = 0
        
        for feature in features:
            if feature not in indicators or feature not in entry_features:
                continue
            
            current = indicators[feature]
            entry = entry_features[feature]
            
            # Calcular mudança relativa
            if entry != 0:
                change = (current - entry) / abs(entry)
            else:
                change = 0
            
            # Verificar se mudança é desfavorável
            if position.side == 'buy':
                if 'rsi' in feature and current > 70:  # RSI alto demais
                    deterioration += 1
                elif 'trend_slope' in feature and change < -0.5:  # Tendência reverteu
                    deterioration += 1
                elif 'macd' in feature and current < 0 and entry > 0:  # MACD cruzou
                    deterioration += 1
            else:  # sell
                if 'rsi' in feature and current < 30:  # RSI baixo demais
                    deterioration += 1
                elif 'trend_slope' in feature and change > 0.5:
                    deterioration += 1
                elif 'macd' in feature and current > 0 and entry < 0:
                    deterioration += 1
            
            count += 1
        
        return deterioration / count if count > 0 else 0
    
    async def train_model(self, historical_ticks: List, labels: List):
        """Treina o modelo com dados históricos"""
        logger.info("Iniciando treinamento do modelo ML...")
        
        try:
            # Preparar features
            X = []
            y = []
            
            for i in range(self.parameters['feature_window'], len(historical_ticks) - self.parameters['prediction_horizon']):
                # Calcular features para janela
                window_ticks = historical_ticks[i - self.parameters['feature_window']:i]
                
                # Simular market context
                market_context = {
                    'recent_ticks': window_ticks,
                    'regime': MarketRegime.TREND  # Simplificado
                }
                
                features = await self.calculate_indicators(market_context)
                
                if features:
                    # Criar vetor de features
                    feature_vector = [features.get(name, 0) for name in self.feature_names]
                    X.append(feature_vector)
                    
                    # Label baseado no movimento futuro
                    future_price = historical_ticks[i + self.parameters['prediction_horizon']].mid
                    current_price = historical_ticks[i].mid
                    
                    if future_price > current_price * 1.0001:  # 1 pip up
                        y.append(1)  # Buy
                    elif future_price < current_price * 0.9999:  # 1 pip down
                        y.append(-1)  # Sell
                    else:
                        y.append(0)  # Hold
            
            # Converter para arrays
            X = np.array(X)
            y = np.array(y)
            
            # Balancear classes
            X_balanced, y_balanced = self._balance_classes(X, y)
            
            # Normalizar
            X_scaled = self.scaler.fit_transform(X_balanced)
            
            # Treinar modelo
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
            
            self.model.fit(X_scaled, y_balanced)
            
            # Salvar modelo
            joblib.dump(self.model, f'models/{self.name}_model.pkl')
            joblib.dump(self.scaler, f'models/{self.name}_scaler.pkl')
            joblib.dump(self.feature_names, f'models/{self.name}_features.pkl')
            
            self.is_trained = True
            
            logger.info(f"Modelo treinado com {len(X_balanced)} amostras")
            logger.info(f"Acurácia no treino: {self.model.score(X_scaled, y_balanced):.2%}")
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
    
    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Balanceia classes para treino"""
        # Contar classes
        unique, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        
        # Subamostrar classes majoritárias
        balanced_indices = []
        
        for class_label in unique:
            class_indices = np.where(y == class_label)[0]
            sampled = np.random.choice(class_indices, min_count, replace=False)
            balanced_indices.extend(sampled)
        
        # Embaralhar
        np.random.shuffle(balanced_indices)
        
        return X[balanced_indices], y[balanced_indices]