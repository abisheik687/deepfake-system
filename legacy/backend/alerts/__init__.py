"""
KAVACH-AI Real-Time Alert System
WebSocket server and notification delivery
NO API KEYS REQUIRED - All processing is local (except optional email/SMS)
"""

import asyncio
from typing import Dict, List, Set, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from loguru import logger

from fastapi import WebSocket, WebSocketDisconnect
from backend.database import SessionLocal, Alert
from backend.detection.threat_intelligence import ThreatLevel, AttackType


class AlertChannel(Enum):
    """Alert delivery channels"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    SYSLOG = "syslog"
    WEBHOOK = "webhook"


@dataclass
class AlertNotification:
    """Alert notification message"""
    alert_id: int
    stream_id: int
    timestamp: datetime
    severity: str
    attack_type: str
    confidence: float
    description: str
    indicators: List[str]
    priority: float


class ConnectionManager:
    """
    WebSocket connection manager.
    
    Manages active WebSocket connections for real-time updates.
    """
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Set[WebSocket] = set()
        logger.info("WebSocket connection manager initialized")
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
        
        if message.get('type') == 'alert':
            logger.info(f"Alert broadcasted to {len(self.active_connections)} clients")
    
    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)


class AlertQueue:
    """
    Priority-based alert queue.
    
    Queues alerts by priority for processing.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize alert queue"""
        self.queue: List[AlertNotification] = []
        self.max_size = max_size
        self.processed_count = 0
        
        logger.info(f"Alert queue initialized (max_size={max_size})")
    
    def enqueue(self, alert: AlertNotification):
        """
        Add alert to queue (sorted by priority).
        
        Args:
            alert: Alert notification
        """
        # Add to queue
        self.queue.append(alert)
        
        # Sort by priority (descending)
        self.queue.sort(key=lambda a: a.priority, reverse=True)
        
        # Limit queue size
        if len(self.queue) > self.max_size:
            dropped = self.queue.pop()  # Remove lowest priority
            logger.warning(f"Alert queue full, dropped alert {dropped.alert_id}")
        
        logger.debug(f"Alert {alert.alert_id} enqueued (priority={alert.priority:.3f})")
    
    def dequeue(self) -> Optional[AlertNotification]:
        """Get highest priority alert from queue"""
        if self.queue:
            alert = self.queue.pop(0)
            self.processed_count += 1
            return alert
        return None
    
    def peek(self) -> Optional[AlertNotification]:
        """View highest priority alert without removing"""
        return self.queue[0] if self.queue else None
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)
    
    def clear(self):
        """Clear all alerts from queue"""
        count = len(self.queue)
        self.queue.clear()
        logger.info(f"Alert queue cleared ({count} alerts removed)")


class NotificationDelivery:
    """
    Multi-channel notification delivery.
    
    Delivers alerts via WebSocket, email, SMS, syslog, etc.
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        enable_email: bool = False,
        enable_sms: bool = False,
        enable_syslog: bool = False
    ):
        """Initialize notification delivery"""
        self.connection_manager = connection_manager
        self.enable_email = enable_email
        self.enable_sms = enable_sms
        self.enable_syslog = enable_syslog
        
        logger.info(
            f"Notification delivery initialized "
            f"(email={enable_email}, sms={enable_sms}, syslog={enable_syslog})"
        )
    
    async def deliver(self, alert: AlertNotification, channels: List[AlertChannel]):
        """
        Deliver alert to specified channels.
        
        Args:
            alert: Alert notification
            channels: Delivery channels
        """
        for channel in channels:
            try:
                if channel == AlertChannel.WEBSOCKET:
                    await self._deliver_websocket(alert)
                elif channel == AlertChannel.EMAIL and self.enable_email:
                    await self._deliver_email(alert)
                elif channel == AlertChannel.SMS and self.enable_sms:
                    await self._deliver_sms(alert)
                elif channel == AlertChannel.SYSLOG and self.enable_syslog:
                    await self._deliver_syslog(alert)
            except Exception as e:
                logger.error(f"Error delivering alert to {channel.value}: {e}")
    
    async def _deliver_websocket(self, alert: AlertNotification):
        """Deliver via WebSocket"""
        message = {
            'type': 'alert',
            'data': asdict(alert)
        }
        await self.connection_manager.broadcast(message)
        logger.info(f"Alert {alert.alert_id} delivered via WebSocket")
    
    async def _deliver_email(self, alert: AlertNotification):
        """Deliver via email"""
        # TODO: Implement email delivery (SMTP)
        # For now, just log
        logger.info(f"Alert {alert.alert_id} would be sent via email (not implemented)")
    
    async def _deliver_sms(self, alert: AlertNotification):
        """Deliver via SMS"""
        # TODO: Implement SMS delivery (Twilio, etc.)
        logger.info(f"Alert {alert.alert_id} would be sent via SMS (not implemented)")
    
    async def _deliver_syslog(self, alert: AlertNotification):
        """Deliver via Syslog/CEF"""
        # CEF format for SIEM integration
        severity_map = {
            'low': 3,
            'medium': 5,
            'high': 8,
            'critical': 10
        }
        
        cef_message = (
            f"CEF:0|KAVACH-AI|Deepfake Detector|1.0|"
            f"alert_{alert.alert_id}|{alert.attack_type}|"
            f"{severity_map.get(alert.severity, 5)}|"
            f"src={alert.stream_id} "
            f"confidence={alert.confidence:.2f} "
            f"priority={alert.priority:.2f} "
            f"msg={alert.description}"
        )
        
        # TODO: Send to syslog server
        logger.info(f"CEF: {cef_message}")


class AlertManager:
    """
    Main alert management system.
    
    Coordinates alert queuing, processing, and delivery.
    """
    
    def __init__(self):
        """Initialize alert manager"""
        self.connection_manager = ConnectionManager()
        self.alert_queue = AlertQueue()
        self.notification_delivery = NotificationDelivery(
            self.connection_manager,
            enable_email=False,  # Disabled by default
            enable_sms=False,
            enable_syslog=True  # Enable syslog/CEF
        )
        
        # Start background alert processor
        self._processing = False
        
        logger.info("Alert manager initialized")
    
    async def process_alert(self, alert_data: dict):
        """
        Process new alert.
        
        Args:
            alert_data: Alert data from detection pipeline
        """
        # Create alert notification
        notification = AlertNotification(
            alert_id=alert_data['id'],
            stream_id=alert_data['stream_id'],
            timestamp=alert_data.get('timestamp', datetime.utcnow()),
            severity=alert_data['severity'],
            attack_type=alert_data.get('attack_type', 'unknown'),
            confidence=alert_data['confidence_score'],
            description=alert_data['description'],
            indicators=alert_data.get('indicators', []),
            priority=alert_data.get('priority', 0.5)
        )
        
        # Add to queue
        self.alert_queue.enqueue(notification)
        
        # Determine delivery channels
        channels = [AlertChannel.WEBSOCKET]
        
        if notification.severity in ['high', 'critical']:
            channels.append(AlertChannel.SYSLOG)
        
        # Deliver immediately for critical alerts
        if notification.severity == 'critical':
            await self.notification_delivery.deliver(notification, channels)
        
        logger.info(
            f"Alert processed: {notification.alert_id} "
            f"({notification.severity}, priority={notification.priority:.3f})"
        )
    
    async def start_processing(self):
        """Start background alert processing"""
        if self._processing:
            logger.warning("Alert processing already running")
            return
        
        self._processing = True
        logger.info("Alert processing started")
        
        while self._processing:
            try:
                # Process alerts from queue
                alert = self.alert_queue.dequeue()
                
                if alert:
                    # Deliver to appropriate channels
                    channels = [AlertChannel.WEBSOCKET]
                    
                    if alert.severity in ['high', 'critical']:
                        channels.append(AlertChannel.SYSLOG)
                    
                    await self.notification_delivery.deliver(alert, channels)
                else:
                    # Queue empty, wait a bit
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(1)
    
    def stop_processing(self):
        """Stop background alert processing"""
        self._processing = False
        logger.info("Alert processing stopped")
    
    async def broadcast_detection(self, detection_data: dict):
        """
        Broadcast detection update to connected clients.
        
        Args:
            detection_data: Detection result data
        """
        message = {
            'type': 'detection',
            'data': detection_data
        }
        await self.connection_manager.broadcast(message)
    
    async def broadcast_stream_update(self, stream_data: dict):
        """
        Broadcast stream status update.
        
        Args:
            stream_data: Stream status data
        """
        message = {
            'type': 'stream_update',
            'data': stream_data
        }
        await self.connection_manager.broadcast(message)


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
