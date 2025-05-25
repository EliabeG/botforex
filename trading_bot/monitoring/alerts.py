# monitoring/alerts.py
"""Sistema de alertas e notificações"""
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

from utils.logger import setup_logger

logger = setup_logger("alerts")

class AlertSeverity(Enum):
    """Níveis de severidade de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Canais de notificação"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"

class Alert:
    """Estrutura de alerta"""
    def __init__(self, title: str, message: str, severity: AlertSeverity,
                 category: str, metadata: Optional[Dict] = None):
        self.id = f"{datetime.now().timestamp()}_{category}"
        self.timestamp = datetime.now()
        self.title = title
        self.message = message
        self.severity = severity
        self.category = category
        self.metadata = metadata or {}
        self.sent = False
        self.sent_channels = []

class AlertManager:
    """Gerenciador de alertas e notificações"""
    
    def __init__(self):
        self.alerts_history = []
        self.alert_rules = {}
        self.channels = {}
        self.rate_limits = {}
        self.callbacks = []
        
        # Configurações padrão
        self.max_alerts_per_hour = 50
        self.alert_cooldown = 300  # 5 minutos entre alertas similares
        
        # Categorias de alerta
        self.categories = {
            'connection': 'Problemas de Conexão',
            'execution': 'Execução de Ordens',
            'risk': 'Gestão de Risco',
            'performance': 'Performance',
            'system': 'Sistema',
            'market': 'Condições de Mercado',
            'strategy': 'Estratégias'
        }
    
    def configure_email(self, smtp_host: str, smtp_port: int,
                       username: str, password: str,
                       from_email: str, to_emails: List[str]):
        """Configura canal de email"""
        self.channels[AlertChannel.EMAIL] = {
            'host': smtp_host,
            'port': smtp_port,
            'username': username,
            'password': password,
            'from': from_email,
            'to': to_emails
        }
        logger.info("Canal de email configurado")
    
    def configure_webhook(self, url: str, headers: Optional[Dict] = None):
        """Configura webhook genérico"""
        self.channels[AlertChannel.WEBHOOK] = {
            'url': url,
            'headers': headers or {}
        }
        logger.info(f"Webhook configurado: {url}")
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configura notificações Telegram"""
        self.channels[AlertChannel.TELEGRAM] = {
            'bot_token': bot_token,
            'chat_id': chat_id,
            'url': f"https://api.telegram.org/bot{bot_token}/sendMessage"
        }
        logger.info("Canal Telegram configurado")
    
    def configure_slack(self, webhook_url: str):
        """Configura notificações Slack"""
        self.channels[AlertChannel.SLACK] = {
            'webhook_url': webhook_url
        }
        logger.info("Canal Slack configurado")
    
    def configure_discord(self, webhook_url: str):
        """Configura notificações Discord"""
        self.channels[AlertChannel.DISCORD] = {
            'webhook_url': webhook_url
        }
        logger.info("Canal Discord configurado")
    
    async def send_alert(self, title: str, message: str, 
                        severity: AlertSeverity = AlertSeverity.INFO,
                        category: str = 'system',
                        metadata: Optional[Dict] = None,
                        channels: Optional[List[AlertChannel]] = None):
        """Envia alerta através dos canais configurados"""
        
        # Verificar rate limiting
        if not self._check_rate_limit(category):
            logger.warning(f"Rate limit atingido para categoria {category}")
            return
        
        # Criar alerta
        alert = Alert(title, message, severity, category, metadata)
        
        # Adicionar ao histórico
        self.alerts_history.append(alert)
        if len(self.alerts_history) > 1000:
            self.alerts_history = self.alerts_history[-1000:]
        
        # Determinar canais
        if channels is None:
            # Usar canais baseados na severidade
            if severity == AlertSeverity.CRITICAL:
                channels = list(self.channels.keys())
            elif severity == AlertSeverity.ERROR:
                channels = [AlertChannel.EMAIL, AlertChannel.TELEGRAM]
            else:
                channels = [AlertChannel.WEBHOOK]
        
        # Enviar por cada canal
        tasks = []
        for channel in channels:
            if channel in self.channels:
                tasks.append(self._send_to_channel(alert, channel))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Marcar canais bem-sucedidos
            for i, result in enumerate(results):
                if result is not True:
                    logger.error(f"Erro ao enviar alerta por {channels[i]}: {result}")
                else:
                    alert.sent_channels.append(channels[i])
            
            alert.sent = len(alert.sent_channels) > 0
        
        # Executar callbacks
        for callback in self.callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Erro em callback de alerta: {e}")
        
        logger.info(f"Alerta enviado: {title} [{severity.value}] via {alert.sent_channels}")
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Envia alerta para canal específico"""
        try:
            if channel == AlertChannel.EMAIL:
                return await self._send_email(alert)
            elif channel == AlertChannel.WEBHOOK:
                return await self._send_webhook(alert)
            elif channel == AlertChannel.TELEGRAM:
                return await self._send_telegram(alert)
            elif channel == AlertChannel.SLACK:
                return await self._send_slack(alert)
            elif channel == AlertChannel.DISCORD:
                return await self._send_discord(alert)
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao enviar por {channel}: {e}")
            return False
    
    async def _send_email(self, alert: Alert) -> bool:
        """Envia alerta por email"""
        config = self.channels.get(AlertChannel.EMAIL)
        if not config:
            return False
        
        try:
            # Criar mensagem
            msg = MIMEMultipart()
            msg['From'] = config['from']
            msg['To'] = ', '.join(config['to'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Corpo do email
            body = f"""
Trading Bot Alert

Severity: {alert.severity.value.upper()}
Category: {self.categories.get(alert.category, alert.category)}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC

{alert.message}

"""
            if alert.metadata:
                body += "Details:\n"
                for key, value in alert.metadata.items():
                    body += f"  {key}: {value}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Enviar email (executar em thread para não bloquear)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email_sync, config, msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
            return False
    
    def _send_email_sync(self, config: Dict, msg: MIMEMultipart):
        """Envia email de forma síncrona"""
        with smtplib.SMTP(config['host'], config['port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
    
    async def _send_webhook(self, alert: Alert) -> bool:
        """Envia alerta via webhook"""
        config = self.channels.get(AlertChannel.WEBHOOK)
        if not config:
            return False
        
        try:
            payload = {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'category': alert.category,
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=payload,
                    headers=config['headers']
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
            return False
    
    async def _send_telegram(self, alert: Alert) -> bool:
        """Envia alerta via Telegram"""
        config = self.channels.get(AlertChannel.TELEGRAM)
        if not config:
            return False
        
        try:
            # Formatar mensagem
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            text = f"{severity_emoji[alert.severity]} *{alert.title}*\n\n"
            text += f"_{alert.message}_\n\n"
            
            if alert.metadata:
                text += "```\n"
                for key, value in alert.metadata.items():
                    text += f"{key}: {value}\n"
                text += "```"
            
            payload = {
                'chat_id': config['chat_id'],
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['url'], json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")
            return False
    
    async def _send_slack(self, alert: Alert) -> bool:
        """Envia alerta via Slack"""
        config = self.channels.get(AlertChannel.SLACK)
        if not config:
            return False
        
        try:
            # Formatar para Slack
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.ERROR: "#f44336",
                AlertSeverity.CRITICAL: "#d32f2f"
            }
            
            payload = {
                'attachments': [{
                    'color': color[alert.severity],
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {
                            'title': key,
                            'value': str(value),
                            'short': True
                        }
                        for key, value in alert.metadata.items()
                    ] if alert.metadata else [],
                    'footer': f"Trading Bot | {alert.category}",
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['webhook_url'],
                    json=payload
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Erro ao enviar Slack: {e}")
            return False
    
    async def _send_discord(self, alert: Alert) -> bool:
        """Envia alerta via Discord"""
        config = self.channels.get(AlertChannel.DISCORD)
        if not config:
            return False
        
        try:
            # Formatar para Discord
            color = {
                AlertSeverity.INFO: 0x00ff00,
                AlertSeverity.WARNING: 0xffa500,
                AlertSeverity.ERROR: 0xff0000,
                AlertSeverity.CRITICAL: 0x8b0000
            }
            
            embed = {
                'title': alert.title,
                'description': alert.message,
                'color': color[alert.severity],
                'timestamp': alert.timestamp.isoformat(),
                'footer': {
                    'text': f"Trading Bot | {alert.category}"
                }
            }
            
            if alert.metadata:
                embed['fields'] = [
                    {
                        'name': key,
                        'value': str(value),
                        'inline': True
                    }
                    for key, value in alert.metadata.items()
                ]
            
            payload = {'embeds': [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['webhook_url'],
                    json=payload
                ) as response:
                    return response.status == 204
                    
        except Exception as e:
            logger.error(f"Erro ao enviar Discord: {e}")
            return False
    
    def _check_rate_limit(self, category: str) -> bool:
        """Verifica rate limiting por categoria"""
        now = datetime.now()
        
        # Limpar rate limits antigos
        self.rate_limits = {
            k: v for k, v in self.rate_limits.items()
            if now - v < timedelta(hours=1)
        }
        
        # Verificar limite global
        if len(self.rate_limits) >= self.max_alerts_per_hour:
            return False
        
        # Verificar cooldown por categoria
        last_alert_key = f"{category}_last"
        if last_alert_key in self.rate_limits:
            if now - self.rate_limits[last_alert_key] < timedelta(seconds=self.alert_cooldown):
                return False
        
        # Atualizar rate limits
        self.rate_limits[f"{category}_{now.timestamp()}"] = now
        self.rate_limits[last_alert_key] = now
        
        return True
    
    def register_callback(self, callback: Callable):
        """Registra callback para alertas"""
        self.callbacks.append(callback)
    
    def get_alert_history(self, 
                         category: Optional[str] = None,
                         severity: Optional[AlertSeverity] = None,
                         limit: int = 100) -> List[Alert]:
        """Retorna histórico de alertas"""
        alerts = self.alerts_history
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    def get_alert_statistics(self) -> Dict:
        """Retorna estatísticas de alertas"""
        stats = {
            'total': len(self.alerts_history),
            'by_severity': {},
            'by_category': {},
            'last_24h': 0,
            'last_hour': 0,
            'channels_configured': list(self.channels.keys())
        }
        
        now = datetime.now()
        
        for alert in self.alerts_history:
            # Por severidade
            severity = alert.severity.value
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            
            # Por categoria
            category = alert.category
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # Últimas 24h
            if now - alert.timestamp < timedelta(hours=24):
                stats['last_24h'] += 1
                
                # Última hora
                if now - alert.timestamp < timedelta(hours=1):
                    stats['last_hour'] += 1
        
        return stats
    
    # === Alertas Predefinidos ===
    
    async def alert_connection_lost(self, service: str, details: str = ""):
        """Alerta de perda de conexão"""
        await self.send_alert(
            title=f"Conexão Perdida: {service}",
            message=f"A conexão com {service} foi perdida. {details}",
            severity=AlertSeverity.ERROR,
            category='connection',
            metadata={'service': service, 'timestamp': datetime.now().isoformat()}
        )
    
    async def alert_high_latency(self, latency_ms: float, threshold_ms: float = 100):
        """Alerta de alta latência"""
        if latency_ms > threshold_ms:
            await self.send_alert(
                title="Alta Latência Detectada",
                message=f"Latência de {latency_ms:.1f}ms detectada (limite: {threshold_ms}ms)",
                severity=AlertSeverity.WARNING,
                category='performance',
                metadata={'latency_ms': latency_ms, 'threshold_ms': threshold_ms}
            )
    
    async def alert_drawdown(self, current_dd: float, max_allowed: float):
        """Alerta de drawdown"""
        severity = AlertSeverity.WARNING
        if current_dd >= max_allowed:
            severity = AlertSeverity.CRITICAL
        elif current_dd >= max_allowed * 0.8:
            severity = AlertSeverity.ERROR
        
        await self.send_alert(
            title=f"Drawdown: {current_dd:.1%}",
            message=f"Drawdown atual de {current_dd:.1%} (limite: {max_allowed:.1%})",
            severity=severity,
            category='risk',
            metadata={'current_drawdown': current_dd, 'max_allowed': max_allowed}
        )
    
    async def alert_circuit_breaker(self, reason: str, details: Dict):
        """Alerta de circuit breaker acionado"""
        await self.send_alert(
            title="🚨 CIRCUIT BREAKER ACIONADO",
            message=f"Trading pausado devido a: {reason}",
            severity=AlertSeverity.CRITICAL,
            category='risk',
            metadata=details
        )
    
    async def alert_strategy_error(self, strategy: str, error: str):
        """Alerta de erro em estratégia"""
        await self.send_alert(
            title=f"Erro na Estratégia: {strategy}",
            message=f"Erro detectado: {error}",
            severity=AlertSeverity.ERROR,
            category='strategy',
            metadata={'strategy': strategy, 'error': error}
        )
    
    async def alert_order_failed(self, order_id: str, reason: str, strategy: str = ""):
        """Alerta de falha em ordem"""
        await self.send_alert(
            title="Falha na Execução de Ordem",
            message=f"Ordem {order_id} falhou: {reason}",
            severity=AlertSeverity.ERROR,
            category='execution',
            metadata={'order_id': order_id, 'reason': reason, 'strategy': strategy}
        )
    
    async def alert_daily_summary(self, metrics: Dict):
        """Envia resumo diário"""
        message = f"""
Resumo Diário de Trading

📊 Performance:
- PnL: ${metrics.get('daily_pnl', 0):.2f} ({metrics.get('daily_pnl_pct', 0):.2%})
- Trades: {metrics.get('total_trades', 0)}
- Win Rate: {metrics.get('win_rate', 0):.1%}

💰 Conta:
- Saldo: ${metrics.get('balance', 0):.2f}
- Drawdown: {metrics.get('drawdown', 0):.1%}

🎯 Estratégias Ativas: {metrics.get('active_strategies', 0)}
⏱️ Uptime: {metrics.get('uptime', 'N/A')}
"""
        
        await self.send_alert(
            title="📈 Resumo Diário",
            message=message,
            severity=AlertSeverity.INFO,
            category='performance',
            metadata=metrics,
            channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM]
        )
    
    async def alert_market_conditions(self, regime: str, volatility: float, spread: float):
        """Alerta sobre condições de mercado"""
        if volatility > 0.02 or spread > 2:  # Alta volatilidade ou spread
            await self.send_alert(
                title="⚠️ Condições de Mercado Adversas",
                message=f"Regime: {regime} | Vol: {volatility:.2%} | Spread: {spread:.1f} pips",
                severity=AlertSeverity.WARNING,
                category='market',
                metadata={
                    'regime': regime,
                    'volatility': volatility,
                    'spread_pips': spread
                }
            )