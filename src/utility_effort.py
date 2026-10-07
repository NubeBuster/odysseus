"""Reasoning effort for utility-role LLM calls.

One owner for the rule.  Precedence for a utility call:

1. an explicit caller effort (never touched here);
2. the ``utility_reasoning_effort`` setting, sent only to routes on the
   utility chain (the Utility endpoint/model plus its fallbacks, matched by
   ``(url, model)``);
3. the chat session's own effort (``thinking_mode == "effort:<level>"``), when
   the call is made for a chat session and its route is that session's route.

Every level is validated against what the model advertises; a model without
evidence gets nothing.  Calls with no session only ever use (2).
"""
from typing import Optional


def utility_routes(owner: Optional[str] = None) -> set:
    """The ``(url, model)`` routes that make up the utility chain."""
    from src.endpoint_resolver import resolve_endpoint, resolve_utility_fallback_candidates

    return {
        (url, model)
        for url, model, _headers in [
            resolve_endpoint("utility", owner=owner),
            *resolve_utility_fallback_candidates(owner=owner),
        ]
        if url and model
    }


def configured_effort(owner: Optional[str] = None) -> str:
    """The raw setting, lowercased; empty when unset (the default)."""
    from src.settings import get_user_setting

    return str(get_user_setting("utility_reasoning_effort", owner or "", "") or "").strip().lower()


def effort_for_route(url: str, model: str, owner: Optional[str] = None) -> Optional[str]:
    """Validated effort for ``(url, model)``, or None when it must not be sent."""
    effort = configured_effort(owner)
    if not effort or (url, model) not in utility_routes(owner):
        return None
    from src.chatgpt_subscription import validate_reasoning_effort

    return validate_reasoning_effort(model, effort)


def session_effort(session) -> Optional[str]:
    """The raw effort level the chat session is set to, or None."""
    mode = str(getattr(session, "thinking_mode", "") or "")
    if not mode.startswith("effort:"):
        return None
    return mode[len("effort:"):].strip().lower() or None


def _is_session_route(url: str, model: str, session) -> bool:
    from src.endpoint_resolver import same_endpoint_base

    return bool(
        model
        and model == getattr(session, "model", None)
        and same_endpoint_base(url, getattr(session, "endpoint_url", None))
    )


def effort_for_call(url: str, model: str, owner: Optional[str] = None, session=None) -> Optional[str]:
    """Effort an owner-aware utility call should send to ``(url, model)``.

    ``session`` is the chat session the call is made for, or None for
    session-less work (scheduled tasks, pollers, auto-sort).  Never raises:
    a lookup failure means "send nothing".
    """
    try:
        effort = effort_for_route(url, model, owner)
        if effort or session is None:
            return effort
        level = session_effort(session)
        if not level or not _is_session_route(url, model, session):
            return None
        from src.chatgpt_subscription import validate_reasoning_effort

        return validate_reasoning_effort(model, level)
    except Exception:
        return None


def candidate_effort_factory(owner=None):
    """Per-candidate factory for `llm_call_async_with_fallback`.

    Returns None when the setting is empty.  Otherwise a factory sending the
    effort only to utility-chain candidates (owner-scoped) whose model
    advertises it.
    """
    if not configured_effort(owner):
        return None

    def factory(_index, url, model, _headers):
        effort = effort_for_route(url, model, owner)
        return {"kwargs": {"reasoning_effort": effort}} if effort else {}

    return factory
