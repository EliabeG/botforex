# monitoring/alerts.py
"""Sistema de alertas e notificações"""
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable, Any, Set # Adicionado Any, Set
from datetime import datetime, timedelta, timezone # Adicionado timezone
from enum import Enum
import json
import uuid # Para ID de alerta mais único se necessário

from utils.logger import setup_logger
# Importar CONFIG para configurações de canais de alerta
from config.settings import CONFIG

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
    WEBHOOK = "webhook" # Webhook genérico
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    # Adicionar outros canais conforme necessário, ex: PAGERDUTY, SMS

class Alert:
    """Estrutura de alerta"""
    def __init__(self, title: str, message: str, severity: AlertSeverity,
                 category: str, metadata: Optional[Dict[str, Any]] = None): # Usar Any
        self.id: str = f"{category}_{str(uuid.uuid4())[:8]}" # ID mais curto e único
        self.timestamp: datetime = datetime.now(timezone.utc) # Usar UTC
        self.title: str = title
        self.message: str = message
        self.severity: AlertSeverity = severity
        self.category: str = category
        self.metadata: Dict[str, Any] = metadata or {}
        self.sent: bool = False
        self.sent_channels: List[AlertChannel] = []
        self.error_sending: Optional[str] = None # Para registrar erros de envio

class AlertManager:
    """Gerenciador de alertas e notificações"""

    def __init__(self):
        self.alerts_history: List[Alert] = [] # Buffer de histórico de alertas
        self.alert_rules: Dict[str, Any] = {} # Para definir regras de alerta (não usado no código original)
        self.channels: Dict[AlertChannel, Dict[str, Any]] = {} # Configurações dos canais
        self.rate_limits_category_timestamps: Dict[str, datetime] = {} # Para cooldown por categoria
        self.global_alerts_in_last_hour: deque[datetime] = deque() # Para rate limit global

        # Configurações padrão (podem ser carregadas de CONFIG)
        self.max_alerts_per_hour_global: int = getattr(CONFIG, 'ALERT_MAX_GLOBAL_PER_HOUR', 50)
        self.alert_cooldown_seconds_per_category: int = getattr(CONFIG, 'ALERT_COOLDOWN_PER_CATEGORY_S', 300) # 5 minutos


        self.callbacks: List[Callable[[Alert], Any]] = [] # Callbacks que recebem o objeto Alert


        self.categories: Dict[str, str] = { # Descrições para categorias
            'connection': 'Problemas de Conexão',
            'execution': 'Execução de Ordens',
            'risk': 'Gestão de Risco',
            'performance': 'Performance do Bot',
            'system': 'Sistema Interno do Bot',
            'market': 'Condições de Mercado',
            'strategy': 'Estratégias de Trading',
            'data': 'Gerenciamento de Dados',
            'security': 'Eventos de Segurança'
        }
        self._load_channel_configs_from_settings() # Carregar configs ao inicializar

    def _load_channel_configs_from_settings(self):
        """Carrega configurações de canais a partir de CONFIG (settings.py)."""
        # Exemplo para Email (precisa de mais campos em CONFIG)
        if getattr(CONFIG, 'ALERT_EMAIL_HOST', None):
            self.configure_email(
                smtp_host=CONFIG.ALERT_EMAIL_HOST, # type: ignore
                smtp_port=getattr(CONFIG, 'ALERT_EMAIL_PORT', 587),
                username=getattr(CONFIG, 'ALERT_EMAIL_USER', ''),
                password=getattr(CONFIG, 'ALERT_EMAIL_PASS', ''),
                from_email=getattr(CONFIG, 'ALERT_EMAIL_FROM', 'bot@example.com'),
                to_emails=getattr(CONFIG, 'ALERT_EMAIL_TO', '').split(',') if getattr(CONFIG, 'ALERT_EMAIL_TO', None) else []
            )
        # Adicionar carregamento para outros canais (Telegram, Slack, etc.) de forma similar
        if getattr(CONFIG, 'ALERT_TELEGRAM_BOT_TOKEN', None) and getattr(CONFIG, 'ALERT_TELEGRAM_CHAT_ID', None):
            self.configure_telegram(CONFIG.ALERT_TELEGRAM_BOT_TOKEN, CONFIG.ALERT_TELEGRAM_CHAT_ID) # type: ignore

        if getattr(CONFIG, 'ALERT_SLACK_WEBHOOK_URL', None):
            self.configure_slack(CONFIG.ALERT_SLACK_WEBHOOK_URL) # type: ignore

        if getattr(CONFIG, 'ALERT_DISCORD_WEBHOOK_URL', None):
            self.configure_discord(CONFIG.ALERT_DISCORD_WEBHOOK_URL) # type: ignore

        if getattr(CONFIG, 'ALERT_GENERIC_WEBHOOK_URL', None):
            self.configure_webhook(CONFIG.ALERT_GENERIC_WEBHOOK_URL) # type: ignore


    def configure_email(self, smtp_host: str, smtp_port: int,
                       username: str, password: str,
                       from_email: str, to_emails: List[str]):
        """Configura canal de email"""
        if not all([smtp_host, username, password, from_email, to_emails]):
            logger.warning("Configuração de email incompleta. Canal de email desabilitado.")
            return
        self.channels[AlertChannel.EMAIL] = {
            'host': smtp_host,
            'port': smtp_port,
            'username': username,
            'password': password,
            'from': from_email,
            'to': to_emails
        }
        logger.info("Canal de Email para alertas configurado.")


    def configure_webhook(self, url: str, headers: Optional[Dict[str, str]] = None):
        """Configura webhook genérico"""
        if not url:
            logger.warning("URL de Webhook não fornecida. Canal de Webhook genérico desabilitado.")
            return
        self.channels[AlertChannel.WEBHOOK] = {
            'url': url,
            'headers': headers or {'Content-Type': 'application/json'} # Default para JSON
        }
        logger.info(f"Webhook genérico para alertas configurado: {url}")


    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configura notificações Telegram"""
        if not bot_token or not chat_id:
            logger.warning("Token do bot Telegram ou Chat ID não fornecido. Canal Telegram desabilitado.")
            return
        self.channels[AlertChannel.TELEGRAM] = {
            'bot_token': bot_token,
            'chat_id': chat_id,
            'url': f"https://api.telegram.org/bot{bot_token}/sendMessage"
        }
        logger.info("Canal Telegram para alertas configurado.")


    def configure_slack(self, webhook_url: str):
        """Configura notificações Slack"""
        if not webhook_url:
            logger.warning("URL do webhook Slack não fornecida. Canal Slack desabilitado.")
            return
        self.channels[AlertChannel.SLACK] = {
            'webhook_url': webhook_url
        }
        logger.info("Canal Slack para alertas configurado.")


    def configure_discord(self, webhook_url: str):
        """Configura notificações Discord"""
        if not webhook_url:
            logger.warning("URL do webhook Discord não fornecida. Canal Discord desabilitado.")
            return
        self.channels[AlertChannel.DISCORD] = {
            'webhook_url': webhook_url
        }
        logger.info("Canal Discord para alertas configurado.")


    async def send_alert(self, title: str, message: str,
                        severity: AlertSeverity = AlertSeverity.INFO,
                        category: str = 'system', # Usar chaves de self.categories
                        metadata: Optional[Dict[str, Any]] = None, # Usar Any
                        target_channels: Optional[List[AlertChannel]] = None): # Renomeado de channels
        """Envia alerta através dos canais configurados, aplicando rate limiting."""

        if category not in self.categories:
            logger.warning(f"Categoria de alerta desconhecida: '{category}'. Usando 'system'.")
            category = 'system'


        if not self._check_rate_limits(category, severity, title): # Passar mais info para rate limit
            return # Log já feito em _check_rate_limits

        alert = Alert(title, message, severity, category, metadata)
        self.alerts_history.append(alert)
        if len(self.alerts_history) > 2000: # Manter histórico maior
            self.alerts_history = self.alerts_history[-2000:]

        # Determinar canais de destino
        channels_to_send: Set[AlertChannel] = set() # Usar set para evitar duplicatas
        if target_channels: # Se canais específicos forem fornecidos
            channels_to_send.update(target_channels)
        else: # Usar lógica baseada na severidade
            if severity == AlertSeverity.CRITICAL:
                channels_to_send.update(self.channels.keys()) # Todos os canais configurados
            elif severity == AlertSeverity.ERROR:
                channels_to_send.update([ch for ch in [AlertChannel.EMAIL, AlertChannel.TELEGRAM, AlertChannel.SLACK] if ch in self.channels])
            elif severity == AlertSeverity.WARNING:
                channels_to_send.update([ch for ch in [AlertChannel.TELEGRAM, AlertChannel.DISCORD] if ch in self.channels])
            else: # INFO
                channels_to_send.update([ch for ch in [AlertChannel.WEBHOOK] if ch in self.channels]) # Ex: apenas webhook para INFO


        if not channels_to_send:
            logger.info(f"Nenhum canal de alerta configurado ou adequado para o alerta '{title}' com severidade {severity.value}.")
            return


        tasks = [self._send_to_channel(alert, ch) for ch in channels_to_send if ch in self.channels]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result_item in enumerate(results): # Renomeado result para result_item
                sent_channel = list(channels_to_send)[i] # Recuperar canal correspondente
                if isinstance(result_item, Exception) or result_item is not True : # Checar se é exceção ou False
                    alert.error_sending = str(result_item)
                    logger.error(f"Erro ao enviar alerta por {sent_channel.value}: {result_item}")
                else:
                    alert.sent_channels.append(sent_channel)

            if alert.sent_channels: # Se enviado para pelo menos um canal
                alert.sent = True
                logger.info(f"Alerta '{alert.title}' [{alert.severity.value}] enviado via: {[ch.value for ch in alert.sent_channels]}")
            else:
                logger.error(f"Falha ao enviar alerta '{alert.title}' por todos os canais selecionados. Último erro: {alert.error_sending}")


        # Executar callbacks síncronos (se houver) ou agendar corrotinas
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(alert))
                else:
                    callback(alert) # Para callbacks síncronos
            except Exception as e_cb: # Renomeado
                logger.error(f"Erro em callback de alerta: {e_cb}")


    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Envia alerta para canal específico (dispatcher)."""
        sender_map: Dict[AlertChannel, Callable[[Alert], Any]] = { # Usar Any para o retorno da Coroutine
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.WEBHOOK: self._send_webhook,
            AlertChannel.TELEGRAM: self._send_telegram,
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.DISCORD: self._send_discord,
        }
        if channel in sender_map:
            try:
                return await sender_map[channel](alert)
            except Exception as e_send: # Renomeado
                logger.error(f"Exceção ao enviar alerta por {channel.value} para '{alert.title}': {e_send}")
                return False
        logger.warning(f"Tentativa de enviar alerta por canal não implementado: {channel.value}")
        return False


    async def _send_email(self, alert: Alert) -> bool:
        """Envia alerta por email."""
        config = self.channels.get(AlertChannel.EMAIL)
        if not config or not config.get('to'): # Checar se 'to' existe e não está vazio
            logger.debug("Canal de email não configurado ou sem destinatários para alerta.")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = config['from']
            msg['To'] = ', '.join(config['to'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Trading Bot: {alert.title}"

            category_desc = self.categories.get(alert.category, alert.category.capitalize())
            html_body = f"""
            <html><body>
            <h2>Trading Bot Alert: {alert.title}</h2>
            <p><strong>Severity:</strong> <span style="color:{self._get_severity_color(alert.severity)};">{alert.severity.value.upper()}</span></p>
            <p><strong>Category:</strong> {category_desc}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}</p>
            <hr>
            <p>{alert.message.replace(chr(10), "<br>")}</p>
            """
            if alert.metadata:
                html_body += "<h3>Details:</h3><pre>"
                html_body += json.dumps(alert.metadata, indent=2, default=str)
                html_body += "</pre>"
            html_body += "</body></html>"
            msg.attach(MIMEText(html_body, 'html'))


            loop = asyncio.get_event_loop()
            # Usar um executor para a operação bloqueante de envio de email
            await loop.run_in_executor(None, self._send_email_sync, config, msg)
            return True
        except Exception as e:
            logger.exception(f"Erro ao construir ou enviar email para alerta '{alert.title}':")
            return False

    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Retorna cor HTML para a severidade."""
        return {
            AlertSeverity.INFO: "blue",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.ERROR: "red",
            AlertSeverity.CRITICAL: "darkred"
        }.get(severity, "black")


    def _send_email_sync(self, config: Dict[str, Any], msg: MIMEMultipart):
        """Envia email de forma síncrona (para ser usado com run_in_executor)."""
        # Usar STARTTLS se a porta for 587, ou SMTP_SSL se for 465
        # Este exemplo assume STARTTLS.
        smtp_class = smtplib.SMTP_SSL if config['port'] == 465 else smtplib.SMTP
        try:
            with smtp_class(config['host'], config['port'], timeout=10) as server: # Adicionado timeout
                if config['port'] != 465: # Não usar starttls com SMTP_SSL
                    server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
                logger.debug(f"Email enviado para {msg['To']} com assunto: {msg['Subject']}")
        except Exception as e_sync_email: # Renomeado
            # Este erro será logado pelo chamador (_send_email) se run_in_executor levantar exceção
            raise ConnectionError(f"Falha no _send_email_sync: {e_sync_email}") from e_sync_email


    async def _send_webhook(self, alert: Alert) -> bool:
        """Envia alerta via webhook genérico."""
        config = self.channels.get(AlertChannel.WEBHOOK)
        if not config:
            logger.debug("Canal Webhook não configurado para alerta.")
            return False

        try:
            payload = {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'category': alert.category,
                'category_description': self.categories.get(alert.category, alert.category.capitalize()),
                'metadata': alert.metadata
            }
            timeout = aiohttp.ClientTimeout(total=10) # Timeout de 10s para webhook
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    config['url'],
                    json=payload,
                    headers=config.get('headers') # Usar .get()
                ) as response:
                    if 200 <= response.status < 300:
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"Falha ao enviar webhook para {config['url']} (Status: {response.status}): {response_text}")
                        return False
        except Exception as e:
            logger.exception(f"Erro ao enviar webhook para {config.get('url', 'N/A')} (Alerta: '{alert.title}'):")
            return False

    async def _send_telegram(self, alert: Alert) -> bool:
        """Envia alerta via Telegram."""
        config = self.channels.get(AlertChannel.TELEGRAM)
        if not config:
            logger.debug("Canal Telegram não configurado.")
            return False

        try:
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }.get(alert.severity, "📢")


            text = (f"{severity_emoji} *{alert.severity.value.upper()}: {alert.title}*\n\n"
                    f"_{self.categories.get(alert.category, alert.category.capitalize())}_\n\n"
                    f"{alert.message}\n\n")

            if alert.metadata:
                text += "*Detalhes Adicionais:*\n```json\n"
                # Limitar tamanho do metadata para não exceder limites do Telegram
                metadata_str = json.dumps(alert.metadata, indent=2, default=str)
                if len(metadata_str) > 1000: # Limite arbitrário
                    metadata_str = metadata_str[:1000] + "\n... (truncado)"
                text += metadata_str + "\n```\n"
            text += f"`Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}`"


            payload = {
                'chat_id': config['chat_id'],
                'text': text,
                'parse_mode': 'MarkdownV2' # Usar MarkdownV2 para melhor formatação (escapar caracteres especiais)
            }
            # Escapar caracteres para MarkdownV2
            def escape_markdown_v2(text_md: str) -> str: # Renomeado text para text_md
                escape_chars = r'_*[]()~`>#+-=|{}.!'
                return "".join(['\\' + char if char in escape_chars else char for char in text_md])

            payload['text'] = escape_markdown_v2(payload['text'])


            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config['url'], json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"Falha ao enviar Telegram (Status: {response.status}): {response_text}. Payload: {payload}")
                        return False
        except Exception as e:
            logger.exception(f"Erro ao enviar mensagem Telegram para alerta '{alert.title}':")
            return False


    async def _send_slack(self, alert: Alert) -> bool:
        """Envia alerta via Slack."""
        config = self.channels.get(AlertChannel.SLACK)
        if not config:
            logger.debug("Canal Slack não configurado.")
            return False

        try:
            color_hex = { # Cores hexadecimais para Slack
                AlertSeverity.INFO: "#439FE0", # Azul
                AlertSeverity.WARNING: "#FFA500", # Laranja
                AlertSeverity.ERROR: "#D00000", # Vermelho
                AlertSeverity.CRITICAL: "#B00020"  # Vermelho escuro
            }.get(alert.severity, "#808080") # Cinza para default


            fields = []
            if alert.metadata:
                for key, value in alert.metadata.items():
                    # Limitar tamanho do valor do campo
                    value_str = str(value)
                    if len(value_str) > 150: # Limite arbitrário
                        value_str = value_str[:150] + "..."
                    fields.append({'title': key.capitalize(), 'value': value_str, 'short': len(value_str) < 40})


            payload = {
                'attachments': [{
                    'fallback': f"[{alert.severity.value.upper()}] {alert.title} - {alert.message}",
                    'color': color_hex,
                    'author_name': "Trading Bot Alert",
                    # 'author_icon': "URL_PARA_UM_ICONE_DO_BOT", # Opcional
                    'title': f"[{alert.severity.value.upper()}] {alert.title}",
                    'text': alert.message,
                    'fields': fields,
                    'footer': f"{self.categories.get(alert.category, alert.category.capitalize())} | {CONFIG.SYMBOL}",
                    'ts': int(alert.timestamp.timestamp())
                }]
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"Falha ao enviar Slack (Status: {response.status}): {response_text}")
                        return False
        except Exception as e:
            logger.exception(f"Erro ao enviar mensagem Slack para alerta '{alert.title}':")
            return False


    async def _send_discord(self, alert: Alert) -> bool:
        """Envia alerta via Discord."""
        config = self.channels.get(AlertChannel.DISCORD)
        if not config:
            logger.debug("Canal Discord não configurado.")
            return False

        try:
            color_decimal = { # Cores decimais para Discord
                AlertSeverity.INFO: 3447003,    # Azul
                AlertSeverity.WARNING: 16753920, # Laranja
                AlertSeverity.ERROR: 15548997,   # Vermelho
                AlertSeverity.CRITICAL: 10038562 # Vermelho escuro
            }.get(alert.severity, 8421504) # Cinza


            embed = {
                'title': f"[{alert.severity.value.upper()}] {alert.title}",
                'description': alert.message,
                'color': color_decimal,
                'timestamp': alert.timestamp.isoformat(),
                'footer': {
                    'text': f"Trading Bot | Categoria: {self.categories.get(alert.category, alert.category.capitalize())} | {CONFIG.SYMBOL}"
                }
            }

            if alert.metadata:
                embed['fields'] = []
                for key, value in alert.metadata.items():
                    value_str = str(value)
                    if len(value_str) > 1020: # Limite do Discord para valor do campo (1024) com margem
                         value_str = value_str[:1020] + "..."
                    embed['fields'].append({
                        'name': key.capitalize(),
                        'value': value_str,
                        'inline': len(value_str) < 40 # Heurística para inline
                    })


            payload = {'embeds': [embed], 'username': "Trading Bot Alerter"} # Nome customizável do bot no Discord

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    # Discord retorna 204 No Content para sucesso com webhooks
                    if response.status == 204:
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"Falha ao enviar Discord (Status: {response.status}): {response_text}")
                        return False
        except Exception as e:
            logger.exception(f"Erro ao enviar mensagem Discord para alerta '{alert.title}':")
            return False


    def _check_rate_limits(self, category: str, severity: AlertSeverity, title: str) -> bool:
        """Verifica rate limiting global e por categoria/título (para evitar spam de alertas idênticos)."""
        now = datetime.now(timezone.utc)

        # Limpar deque de alertas globais da última hora
        while self.global_alerts_in_last_hour and self.global_alerts_in_last_hour[0] < (now - timedelta(hours=1)):
            self.global_alerts_in_last_hour.popleft()

        # Verificar limite global
        if len(self.global_alerts_in_last_hour) >= self.max_alerts_per_hour_global:
            logger.warning(f"Rate limit GLOBAL de alertas atingido ({self.max_alerts_per_hour_global}/hora). Alerta '{title}' suprimido.")
            return False

        # Verificar cooldown por categoria e título (para alertas repetidos)
        # Usar uma chave combinada para o cooldown de alertas idênticos ou muito similares.
        # Poderia ser (category, title) ou um hash do conteúdo.
        # Para simplificar, vamos usar apenas a categoria para o cooldown principal.
        last_alert_time_for_category = self.rate_limits_category_timestamps.get(category)
        if last_alert_time_for_category:
            if (now - last_alert_time_for_category).total_seconds() < self.alert_cooldown_seconds_per_category:
                # Permitir CRITICALs mesmo em cooldown, mas com log mais enfático.
                if severity != AlertSeverity.CRITICAL:
                    logger.info(f"Cooldown para categoria '{category}' ainda ativo. Alerta '{title}' (severidade {severity.value}) suprimido.")
                    return False
                else:
                    logger.warning(f"Cooldown para categoria '{category}' ativo, mas enviando alerta CRÍTICO '{title}'.")


        # Atualizar timestamps e contadores
        self.global_alerts_in_last_hour.append(now)
        self.rate_limits_category_timestamps[category] = now

        return True


    def register_alert_callback(self, callback: Callable[[Alert], Any]): # Renomeado e tipagem de callback
        """Registra callback para ser executado quando um alerta é processado (após tentativa de envio)."""
        if not callable(callback):
            logger.error(f"Tentativa de registrar callback inválido: {callback}")
            return
        self.callbacks.append(callback)
        logger.info(f"Callback de alerta registrado: {getattr(callback, '__name__', repr(callback))}")


    def get_alert_history(self,
                         category: Optional[str] = None,
                         severity: Optional[AlertSeverity] = None,
                         limit: int = 100) -> List[Alert]:
        """Retorna histórico de objetos Alert, filtrados opcionalmente."""
        # Filtrar em uma nova lista para não modificar o histórico original
        filtered_alerts = list(self.alerts_history) # Começar com uma cópia

        if category:
            filtered_alerts = [a for a in filtered_alerts if a.category == category]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        # Retornar os últimos 'limit' alertas da lista filtrada (os mais recentes)
        return filtered_alerts[-limit:]


    def get_alert_statistics(self) -> Dict[str, Any]: # Usar Any
        """Retorna estatísticas de alertas."""
        stats: Dict[str, Any] = { # Tipagem
            'total_alerts_logged': len(self.alerts_history),
            'alerts_sent_successfully': sum(1 for a in self.alerts_history if a.sent),
            'by_severity': defaultdict(int),
            'by_category': defaultdict(int),
            'alerts_last_24h': 0,
            'alerts_last_hour': 0,
            'active_alert_channels': [ch.value for ch in self.channels.keys()] # Lista de nomes de canais
        }
        now = datetime.now(timezone.utc)

        for alert_obj in self.alerts_history: # Renomeado alert para alert_obj
            stats['by_severity'][alert_obj.severity.value] += 1
            stats['by_category'][alert_obj.category] += 1

            if alert_obj.timestamp > (now - timedelta(hours=24)):
                stats['alerts_last_24h'] += 1
                if alert_obj.timestamp > (now - timedelta(hours=1)):
                    stats['alerts_last_hour'] += 1
        return stats


    # === Alertas Predefinidos (Exemplos Melhorados) ===

    async def system_startup_alert(self):
        """Alerta de inicialização do sistema."""
        await self.send_alert(
            title="Trading Bot Iniciado",
            message=f"O Trading Bot foi iniciado com sucesso no modo '{CONFIG.TRADING_MODE}' para o símbolo {CONFIG.SYMBOL}.",
            severity=AlertSeverity.INFO,
            category='system',
            metadata={'mode': CONFIG.TRADING_MODE, 'symbol': CONFIG.SYMBOL, 'version': getattr(CONFIG, 'BOT_VERSION', 'N/A')}
        )

    async def system_shutdown_alert(self, reason: str = "Desligamento normal"):
        """Alerta de desligamento do sistema."""
        await self.send_alert(
            title="Trading Bot Parando",
            message=f"O Trading Bot está sendo desligado. Razão: {reason}",
            severity=AlertSeverity.INFO,
            category='system',
            metadata={'reason': reason}
        )


    async def connection_status_alert(self, service_name: str, is_connected: bool, details: str = ""): # Renomeado
        """Alerta sobre o status de uma conexão."""
        if not is_connected:
            await self.send_alert(
                title=f"Conexão Perdida: {service_name.upper()}",
                message=f"A conexão com o serviço '{service_name}' foi perdida. {details}",
                severity=AlertSeverity.ERROR, # Pode ser CRITICAL dependendo do serviço
                category='connection',
                metadata={'service': service_name, 'status': 'disconnected', 'details': details}
            )
        else: # Opcional: alerta de reconexão
            await self.send_alert(
                title=f"Conexão Restaurada: {service_name.upper()}",
                message=f"A conexão com o serviço '{service_name}' foi restaurada.",
                severity=AlertSeverity.INFO,
                category='connection',
                metadata={'service': service_name, 'status': 'connected'}
            )


    async def high_latency_alert(self, service_name: str, latency_ms: float, threshold_ms: float): # Renomeado
        """Alerta de alta latência para um serviço."""
        # Este alerta pode ser ruidoso, considerar lógica de cooldown mais estrita ou agregação.
        if latency_ms > threshold_ms:
            await self.send_alert(
                title=f"Alta Latência em {service_name.upper()}",
                message=f"Latência de {latency_ms:.1f}ms detectada para '{service_name}' (Limite: {threshold_ms}ms).",
                severity=AlertSeverity.WARNING,
                category='performance',
                metadata={'service': service_name, 'latency_ms': latency_ms, 'threshold_ms': threshold_ms}
            )

    # Renomeado de alert_drawdown para maior clareza
    async def drawdown_limit_alert(self, current_dd_pct: float, limit_dd_pct: float, level: str = "Warning"): # Usar DD em %
        """Alerta de atingimento de limite de drawdown."""
        sev = AlertSeverity.WARNING
        if level.lower() == "critical": sev = AlertSeverity.CRITICAL
        elif level.lower() == "error": sev = AlertSeverity.ERROR

        await self.send_alert(
            title=f"{level.upper()} Drawdown: {current_dd_pct:.2%}",
            message=f"Drawdown atual de {current_dd_pct:.2%} atingiu o nível de {level.lower()} (Limite configurado para alerta: {limit_dd_pct:.2%}).",
            severity=sev,
            category='risk',
            metadata={'current_drawdown_percent': current_dd_pct, 'limit_percent': limit_dd_pct, 'level': level}
        )


    async def circuit_breaker_tripped_alert(self, reason: str, details: Dict[str, Any]): # Renomeado, Usar Any
        """Alerta de circuit breaker acionado."""
        await self.send_alert(
            title="🚨 CIRCUIT BREAKER ACIONADO 🚨",
            message=f"Trading Interrompido! Razão: {reason}.",
            severity=AlertSeverity.CRITICAL,
            category='risk',
            metadata=details
        )
    
    async def circuit_breaker_reset_alert(self):
        """Alerta de circuit breaker resetado."""
        await self.send_alert(
            title="Circuit Breaker Resetado",
            message="O Circuit Breaker foi resetado. Operações de trading podem ser retomadas conforme as regras.",
            severity=AlertSeverity.INFO,
            category='risk'
        )


    async def strategy_error_alert(self, strategy_name: str, error_message: str, error_details: Optional[Dict[str,Any]] = None): # Renomeado
        """Alerta de erro em estratégia."""
        meta = {'strategy': strategy_name, 'error_message': error_message}
        if error_details: meta.update(error_details)

        await self.send_alert(
            title=f"Erro na Estratégia: {strategy_name}",
            message=f"Um erro ocorreu na estratégia '{strategy_name}': {error_message[:200]}...", # Limitar tamanho da msg de erro
            severity=AlertSeverity.ERROR,
            category='strategy',
            metadata=meta
        )


    async def order_execution_failure_alert(self, client_order_id: str, broker_order_id: Optional[str],
                                          reason: str, strategy_name: Optional[str] = None): # Renomeado
        """Alerta de falha na execução de ordem."""
        await self.send_alert(
            title="Falha na Execução de Ordem",
            message=f"Ordem (ClienteID: {client_order_id}, BrokerID: {broker_order_id or 'N/A'}) falhou. Razão: {reason}",
            severity=AlertSeverity.ERROR,
            category='execution',
            metadata={'client_order_id': client_order_id, 'broker_order_id': broker_order_id, 'reason': reason, 'strategy': strategy_name or "N/A"}
        )


    async def daily_trading_summary_alert(self, metrics: Dict[str, Any]): # Renomeado, Usar Any
        """Envia resumo diário de trading."""
        # Formatar a mensagem de forma mais estruturada
        pnl_val = metrics.get('daily_pnl', 0.0)
        pnl_pct_val = metrics.get('daily_pnl_pct', 0.0) * 100 # Converter para %
        balance_val = metrics.get('balance', 0.0)
        drawdown_val = metrics.get('drawdown', 0.0) * 100 # Converter para %

        message = (
            f"📊 *Performance:* PnL: ${pnl_val:.2f} ({pnl_pct_val:.2f}%) | Trades: {metrics.get('total_trades', 0)} | Win Rate: {metrics.get('win_rate', 0.0)*100:.1f}%\n"
            f"💰 *Conta:* Saldo: ${balance_val:.2f} | Drawdown (dia): {drawdown_val:.2f}%\n"
            f"🎯 *Estratégias Ativas:* {metrics.get('active_strategies_count', 0)}\n" # Renomeado
            f"⏱️ *Uptime (aprox):* {metrics.get('uptime_hours', 'N/A')} horas"
        )

        await self.send_alert(
            title=f"📈 Resumo Diário de Trading - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            message=message,
            severity=AlertSeverity.INFO,
            category='performance',
            metadata=metrics, # Enviar todas as métricas como metadados
            target_channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM] # Exemplo de canais alvo
        )

    async def adverse_market_conditions_alert(self, regime: str, volatility: float, spread_pips: float): # Renomeado
        """Alerta sobre condições de mercado adversas (ex: alta volatilidade, spread largo)."""
        # Definir thresholds em CONFIG ou RiskConfig
        high_vol_threshold = getattr(CONFIG, 'HIGH_VOLATILITY_THRESHOLD_ALERT', 0.02) # Ex: 2%
        high_spread_threshold_pips = getattr(CONFIG, 'HIGH_SPREAD_THRESHOLD_ALERT_PIPS', 3.0) # Ex: 3 pips

        is_adverse = False
        reasons = []
        if volatility > high_vol_threshold:
            is_adverse = True
            reasons.append(f"Volatilidade alta ({volatility:.2%})")
        if spread_pips > high_spread_threshold_pips:
            is_adverse = True
            reasons.append(f"Spread largo ({spread_pips:.1f} pips)")

        if is_adverse:
            await self.send_alert(
                title="⚠️ Condições de Mercado Adversas Detectadas",
                message=f"Regime Atual: {regime}. Motivos: {', '.join(reasons)}.",
                severity=AlertSeverity.WARNING,
                category='market',
                metadata={
                    'current_regime': regime,
                    'volatility_value': volatility,
                    'volatility_threshold': high_vol_threshold,
                    'spread_pips_value': spread_pips,
                    'spread_pips_threshold': high_spread_threshold_pips
                }
            )