"""Message bus module for decoupled channel-agent communication."""

from ghostbot.bus.events import InboundMessage, OutboundMessage
from ghostbot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
