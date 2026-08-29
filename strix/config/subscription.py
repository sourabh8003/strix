"""Combine the model-subscription backends (ChatGPT, Claude Code) for the UI
and telemetry call sites that only care whether *some* subscription is in
play, not which one.
"""

from __future__ import annotations

from strix.config import claude_code, codex


def auth_mode(model_name: str | None) -> str:
    if codex.subscription_model(model_name) or claude_code.subscription_model(model_name):
        return "subscription"
    return "api_key"


def subscription_label(model_name: str | None) -> str | None:
    if codex.subscription_model(model_name):
        return "ChatGPT subscription"
    if claude_code.subscription_model(model_name):
        return "Claude subscription"
    return None
