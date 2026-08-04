"""
Email platform plugin for Hermes Gateway.

Supports both standard IMAP/SMTP auth mode (default) and Microsoft Graph API auth mode.
Select mode via environment variable or secret:
    EMAIL_AUTH_MODE=imap  → EmailAdapter (IMAP/SMTP)
    EMAIL_AUTH_MODE=graph → GraphEmailAdapter (Microsoft Graph API)
"""

import logging
import os

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

from .adapter import register as register_imap
from .graph_adapter import register as register_graph

logger = logging.getLogger(__name__)


def _get_auth_mode() -> str:
    """Return configured EMAIL_AUTH_MODE ('imap' or 'graph')."""
    try:
        mode = _scoped_get_secret("EMAIL_AUTH_MODE", "imap")
    except _UnscopedSecretError:
        mode = os.getenv("EMAIL_AUTH_MODE", "imap")
    return (mode or "imap").strip().lower()


def register(ctx) -> None:
    """Register email platform based on EMAIL_AUTH_MODE selection."""
    auth_mode = _get_auth_mode()
    if auth_mode == "graph":
        logger.info("[Email] Registering Microsoft Graph API adapter (EMAIL_AUTH_MODE=graph)")
        register_graph(ctx)
    else:
        logger.info("[Email] Registering IMAP/SMTP adapter (EMAIL_AUTH_MODE=imap)")
        register_imap(ctx)


__all__ = ["register"]
