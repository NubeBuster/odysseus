"""The Message stats panel shows the reasoning effort a response actually used.

The effort is read back from the request payload that was sent, travels on the
stream's usage event into the per-message metrics, and is rendered only when
present.
"""
import asyncio
import json
from pathlib import Path

from src import llm_core
from src.agent_loop import _compute_final_metrics

ROOT = Path(__file__).resolve().parents[1]
RENDERER = (ROOT / "static/js/chatRenderer.js").read_text()

_CHATGPT_URL = "https://chatgpt.com/backend-api/codex/responses"
_LOCAL_URL = "http://127.0.0.1:8081/v1/chat/completions"
_MESSAGES = [{"role": "user", "content": "hi"}]


class _Resp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


class _Ctx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _Resp(self._lines)

    async def __aexit__(self, *args):
        return False


class _Client:
    def __init__(self, lines):
        self._lines = lines
        self.payloads = []

    def stream(self, method, url, **kwargs):
        self.payloads.append(kwargs.get("json"))
        return _Ctx(self._lines)


def _usage(monkeypatch, url, model, lines, effort):
    client = _Client(lines)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)

    async def run():
        usage = None
        async for chunk in llm_core._stream_llm_inner(
            url, model, _MESSAGES, headers={"Authorization": "Bearer t"}, reasoning_effort=effort,
        ):
            for line in chunk.split("\n"):
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    if event.get("type") == "usage":
                        usage = event["data"]
        return usage

    return client, asyncio.run(run())


_CHATGPT_STREAM = [
    "data: " + json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
    "data: " + json.dumps({"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 2}}}),
]
_LOCAL_STREAM = [
    "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}]}),
    "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
    "data: [DONE]",
]


def test_chatgpt_usage_reports_the_effort_the_request_carried(monkeypatch):
    client, usage = _usage(monkeypatch, _CHATGPT_URL, "gpt-5.5", _CHATGPT_STREAM, "low")
    assert client.payloads[0]["reasoning"] == {"effort": "low"}
    assert usage["reasoning_effort"] == "low"


def test_usage_omits_effort_when_none_was_picked(monkeypatch):
    client, usage = _usage(monkeypatch, _CHATGPT_URL, "gpt-5.5", _CHATGPT_STREAM, None)
    assert "reasoning" not in client.payloads[0]
    assert "reasoning_effort" not in usage


def test_usage_omits_effort_the_transport_did_not_send(monkeypatch):
    # A picked effort the payload does not carry must not be reported.
    client, usage = _usage(monkeypatch, _LOCAL_URL, "qwen-local", _LOCAL_STREAM, "low")
    assert "reasoning_effort" not in client.payloads[0]
    assert "reasoning_effort" not in usage


def test_annotation_reads_both_payload_shapes_and_ignores_other_values():
    annotate = llm_core._annotate_usage_effort
    assert annotate({}, {"reasoning": {"effort": "high"}}, "High") == {"reasoning_effort": "high"}
    assert annotate({}, {"reasoning_effort": "low"}, "low") == {"reasoning_effort": "low"}
    # A different value in the payload (e.g. a provider constant) is not the pick.
    assert annotate({}, {"reasoning_effort": "high"}, "low") == {}
    assert annotate({}, {"reasoning_effort": "low"}, None) == {}


def _metrics(**overrides):
    kwargs = dict(
        messages=_MESSAGES, full_response="hello", total_duration=1.0, time_to_first_token=0.1,
        context_length=4096, real_input_tokens=3, real_output_tokens=2, has_real_usage=True,
        tool_events=[], round_texts=[], model="m",
    )
    kwargs.update(overrides)
    return _compute_final_metrics(**kwargs)


def test_final_metrics_carry_effort_only_when_applied():
    assert _metrics(reasoning_effort="low")["reasoning_effort"] == "low"
    assert "reasoning_effort" not in _metrics()


def test_stats_panel_renders_the_effort_row_only_when_recorded():
    assert "metrics.reasoning_effort" in RENDERER
    assert "${reasoningEffort ? `<div class=\"ctx-stat-row\"><span class=\"ctx-label\">Reasoning effort</span>" in RENDERER
