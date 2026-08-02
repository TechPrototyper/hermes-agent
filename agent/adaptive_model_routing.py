"""Adaptive model routing — content-based profile selection for Qwen3.6-27B.

Routes each request to the appropriate LiteLLM profile based on user input
and tool context. No model call needed — pure regex classification.

Profiles (via LiteLLM on litellm.matrix.local):
    qwen3.6-27b          — THINK (temp 0.6, top_p 0.95) — default
    qwen3.6-27b-no-think — FAST  (temp 0.1, top_p 0.3)  — simple tasks
    qwen3.6-27b-tools    — EXEC  (temp 0.0, top_p 0.1)  — tool calling

Compression (qwen3.6-27b-compress) is handled by Hermes config, not this router.

Design:
    - Appends a suffix to the base model name; never replaces it.
    - Falls back to agent.model on any error.
    - Cache-safe: does not mutate past context or rebuild system prompts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Avoid circular import at module level
    pass


# ── Compiled patterns ──────────────────────────────────────────────────────
# German verbs use stem matching (no trailing \b) so conjugations like
# "übersetze", "übersetzt" still match "übersetz".

_FAST_PATTERNS = re.compile(
    r"""
    \b(translate|übersetz|translate\s+to|in\s+\w+\s+übersetzen)
    | \b(format|umformatieren|formatieren|format\s+as)
    | \b(extract|extrahier|extrahiere)
    | \b(classify|klassifizier|kategorisier|taggen)
    | \b(summarize|zusammenfassen|fasse\s+zusammen|kurzfassung|fasse\s+|zusammen)
    | \b(convert|konvertier|umwandeln|umrechnen)
    | \b(list|liste|aufzählen|nenne\s+mir|nenn\s+mir)
    | \b(count|zähle|wie\s+vielen?|wie\s+viel)
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)

_SIMPLE_PATTERNS = re.compile(
    r"""
    ^\s*(ok|ja|nein|yes|no|okay|super|danke|thanks)\b.*\.$
    | ^\s*(zeig|show|list|liste|listet|display)\b
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)


def _has_tool_calls_in_history(messages: list[dict[str, Any]]) -> bool:
    """Check if any assistant turn in the conversation contained tool_calls."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
        # Don't stop at user messages — look through the whole history
    return False


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Vision messages: join text parts
                return " ".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
            return str(content) if content else ""
    return ""


def _strip_model_suffix(model: str) -> str:
    """Remove known suffixes to get the base model name."""
    for suffix in ("-tools", "-no-think", "-compress", "-overflow"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def resolve_model_for_request(
    agent: Any,
    messages: list[dict[str, Any]],
    tools_for_api: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Resolve the model name for this request based on content + context.

    Priority: TOOLS > FAST > THINK (default)

    Args:
        agent: AIAgent instance with agent.model set.
        messages: Current conversation messages.
        tools_for_api: Tool schemas being offered (if any).

    Returns:
        Model name (e.g. "qwen3.6-27b-tools").
    """
    try:
        base_model = _strip_model_suffix(agent.model)
        user_text = _last_user_content(messages)

        # 1. Tool-calling: previous assistant turn had tool_calls → EXEC
        if tools_for_api and _has_tool_calls_in_history(messages):
            return f"{base_model}-tools"

        # 2. Fast: simple transformations, no reasoning needed
        if user_text and _FAST_PATTERNS.search(user_text):
            return f"{base_model}-no-think"

        # 3. Simple questions
        if user_text and _SIMPLE_PATTERNS.search(user_text):
            return f"{base_model}-no-think"

        # 4. Default: THINK (reasoning enabled)
        return base_model

    except Exception:
        # Never crash the agent loop — fall back to current model
        return agent.model


__all__ = ["resolve_model_for_request"]