"""Slash command routing and built-in handlers."""

from ghostbot.command.builtin import register_builtin_commands
from ghostbot.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]
