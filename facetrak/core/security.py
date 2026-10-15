"""Shared security helpers: auth tokens, secret redaction, error envelope."""

import logging
import os
import secrets
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

ENV_TOKEN = "FACETRAK_API_TOKEN"
ENV_MCP_CONFIRM = "FACETRAK_MCP_REQUIRE_CONFIRM"


def get_api_token(cfg: dict) -> str:
    """Bearer token: environment wins, then config file."""
    return os.environ.get(ENV_TOKEN, "") or cfg.get("api", {}).get("auth_token", "")


def token_configured(cfg: dict) -> bool:
    return bool(get_api_token(cfg))


def verify_token(cfg: dict, presented: str | None) -> bool:
    expected = get_api_token(cfg)
    if not expected:
        return True  # auth disabled: localhost-only binding is the boundary
    if not presented:
        return False
    return secrets.compare_digest(presented, expected)


def redact_url(url: str) -> str:
    """Strip userinfo (credentials) from a URL for logs/API output."""
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except (ValueError, TypeError):
        return "<unparseable-url>"


def error(code: str, message: str, status: int = 400, **extra) -> dict:
    """Unified API error envelope."""
    body: dict = {"error": {"code": code, "message": message}}
    if extra:
        body["error"]["details"] = extra
    body["_status"] = status
    return body
