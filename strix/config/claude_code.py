"""Claude Code subscription backend: drive Strix turns through a local,
already-authenticated ``claude`` CLI session instead of a metered API key.

Unlike the ChatGPT subscription backend (:mod:`strix.config.codex`), this
does not reimplement OAuth — it shells out to the ``claude`` binary the user
already has installed and signed in (Pro/Max subscription or otherwise), the
same way any other Claude Code session authenticates. Using a Claude
subscription outside Claude Code / claude.ai is not officially supported by
Anthropic for third-party tools; the user chooses this path knowingly.

Each turn is one stateless ``claude -p`` call: every built-in tool is
disabled (``--tools ""``) and no MCP servers are registered, so the CLI has
nothing of its own to call. Strix's tool schemas and running conversation
are serialized into the prompt instead, and ``--json-schema`` forces the
reply into a small envelope (``{"response_type": "message"|"tool_calls",
...}``) that :mod:`strix.config.models` translates back into the same
``openai.types.responses`` shapes every other provider returns. Strix's own
``Runner`` still executes every tool call — this module only gets the
model's decision out of the CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

CLI_BIN = "claude"
SUBSCRIPTION_PREFIX = "claude-code/"
DEFAULT_TIMEOUT_S = 600.0

_MAX_TRANSCRIPT_ITEM_CHARS = 4000

_STATIC_SYSTEM_PROMPT = (
    "You are answering through the Strix headless bridge, not a normal Claude "
    "Code session. You have no tools of your own here. The user message "
    "describes Strix's own tools and the conversation so far; follow the JSON "
    "response protocol it specifies exactly, with no other commentary."
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "response_type": {"type": "string", "enum": ["message", "tool_calls"]},
        "message": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["response_type"],
}


class ClaudeCodeError(Exception):
    """Raised when the ``claude`` CLI bridge fails for any reason."""


def subscription_model(model_name: str | None) -> str | None:
    """The model slug behind a ``claude-code/<model>`` STRIX_LLM, or None."""
    name = (model_name or "").strip()
    if not name.lower().startswith(SUBSCRIPTION_PREFIX):
        return None
    return name[len(SUBSCRIPTION_PREFIX) :] or None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"


def is_cli_available() -> bool:
    return shutil.which(CLI_BIN) is not None


def is_authenticated() -> bool:
    """Best-effort check via ``claude auth status`` (no API call)."""
    if not is_cli_available():
        return False
    try:
        result = subprocess_run_status()
    except Exception:  # noqa: BLE001
        logger.debug("claude auth status failed", exc_info=True)
        return False
    return bool(result.get("loggedIn"))


def subprocess_run_status() -> dict[str, Any]:
    import subprocess  # noqa: PLC0415 - only needed for this synchronous check

    completed = subprocess.run(  # noqa: S603
        [CLI_BIN, "auth", "status"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class ToolCallRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    text: str | None
    tool_calls: list[ToolCallRequest]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    model_used: str | None
    total_cost_usd: float


def _tool_blocks(tools: list[Any]) -> str:
    blocks: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "params_json_schema", None)
        schema_text = json.dumps(schema, sort_keys=True) if isinstance(schema, dict) else "{}"
        blocks.append(f"### {name}\n{description}\nparameters (JSON Schema): {schema_text}")
    return "\n\n".join(blocks) if blocks else "(no tools available this turn)"


def _model_dump(item: Any) -> dict[str, Any]:
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        with contextlib.suppress(Exception):
            return dict(dump())
    return {"type": getattr(item, "type", None), "role": getattr(item, "role", None)}


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TRANSCRIPT_ITEM_CHARS:
        return text
    return text[:_MAX_TRANSCRIPT_ITEM_CHARS] + f"... [truncated, {len(text)} chars total]"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate(content)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block_data = block if isinstance(block, dict) else _model_dump(block)
            block_type = block_data.get("type")
            if block_type in ("output_text", "input_text", "text"):
                parts.append(str(block_data.get("text", "")))
            elif block_type in ("input_image", "image"):
                parts.append("[image content omitted - not supported by the claude-code bridge]")
            elif block_type == "refusal":
                parts.append(f"[refusal] {block_data.get('refusal', '')}")
            else:
                parts.append(json.dumps(block_data, default=str))
        return _truncate("\n".join(p for p in parts if p))
    return _truncate(json.dumps(content, default=str))


def _serialize_item(item: Any) -> str:
    data = item if isinstance(item, dict) else _model_dump(item)
    item_type = data.get("type")
    if item_type == "function_call":
        return f"[TOOL_CALL id={data.get('call_id')}] {data.get('name')}({data.get('arguments')})"
    if item_type == "function_call_output":
        return f"[TOOL_RESULT id={data.get('call_id')}] {_stringify_content(data.get('output'))}"
    if item_type == "reasoning":
        # Internal reasoning from a prior (possibly different) model turn isn't
        # meaningfully replayable across a stateless bridge call.
        return ""
    role = data.get("role")
    if role:
        return f"{role}: {_stringify_content(data.get('content'))}"
    return f"[item] {_truncate(json.dumps(data, default=str))}"


def _serialize_input(input_items: str | list[Any]) -> str:
    if isinstance(input_items, str):
        return f"user: {input_items}"
    lines = [_serialize_item(item) for item in input_items]
    return "\n\n".join(line for line in lines if line)


def build_stdin_prompt(
    system_instructions: str | None, input_items: str | list[Any], tools: list[Any]
) -> str:
    sections = []
    if system_instructions:
        sections.append(f"=== STRIX SYSTEM INSTRUCTIONS ===\n{system_instructions}")
    sections.append(f"=== AVAILABLE TOOLS ===\n{_tool_blocks(tools)}")
    sections.append(f"=== CONVERSATION TRANSCRIPT ===\n{_serialize_input(input_items)}")
    sections.append(
        "=== YOUR TURN ===\n"
        "Respond now with exactly one JSON object matching the schema. Continue "
        "naturally from the transcript above; call one or more tools in "
        "parallel when that's the right next step."
    )
    return "\n\n".join(sections)


def _build_command(model_slug: str | None) -> list[str]:
    cmd = [
        CLI_BIN,
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--system-prompt",
        _STATIC_SYSTEM_PROMPT,
        "--json-schema",
        json.dumps(_RESPONSE_SCHEMA),
    ]
    if model_slug:
        cmd += ["--model", model_slug]
    return cmd


def _primary_model(model_usage: Any) -> str | None:
    """The model that produced the actual reply, not an auxiliary micro-model call
    (e.g. title generation) that may also appear in ``modelUsage``."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    names: list[str] = list(model_usage)
    return max(
        names,
        key=lambda name: int((model_usage.get(name) or {}).get("outputTokens") or 0),
    )


def _parse_envelope(stdout: bytes, stderr: bytes, returncode: int | None) -> TurnResult:
    text = stdout.decode("utf-8", errors="replace").strip()
    try:
        envelope = json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        detail = stderr.decode("utf-8", errors="replace").strip() or text
        raise ClaudeCodeError(f"claude CLI returned unparsable output: {detail[:2000]}") from exc

    if not isinstance(envelope, dict):
        raise ClaudeCodeError(f"claude CLI returned unexpected output: {text[:2000]}")

    if envelope.get("is_error") or returncode:
        message = envelope.get("result") or stderr.decode("utf-8", errors="replace").strip()
        raise ClaudeCodeError(f"claude CLI turn failed: {str(message)[:2000]}")

    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raw_result = envelope.get("result")
        try:
            parsed = json.loads(raw_result) if isinstance(raw_result, str) else None
            structured = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            structured = None
        if structured is None:
            structured = {"response_type": "message", "message": raw_result or ""}

    tool_calls: list[ToolCallRequest] = []
    for raw_call in structured.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        name = raw_call.get("name")
        if not name:
            continue
        arguments = raw_call.get("arguments")
        tool_calls.append(
            ToolCallRequest(
                name=str(name), arguments=arguments if isinstance(arguments, dict) else {}
            )
        )

    text_out: str | None = None
    if structured.get("response_type") != "tool_calls" or not tool_calls:
        text_out = str(structured.get("message") or "")

    usage = envelope.get("usage") or {}
    model_used = _primary_model(envelope.get("modelUsage"))

    return TurnResult(
        text=text_out,
        tool_calls=tool_calls,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=int(usage.get("cache_read_input_tokens") or 0),
        model_used=model_used,
        total_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
    )


async def run_turn(
    *,
    model_slug: str | None,
    stdin_prompt: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> TurnResult:
    cmd = _build_command(model_slug)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ClaudeCodeError(
            f"'{CLI_BIN}' was not found on PATH. Install Claude Code "
            "(https://claude.com/claude-code) to use STRIX_LLM=claude-code/<model>."
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_prompt.encode("utf-8")), timeout=timeout_s
        )
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise ClaudeCodeError(f"claude CLI did not respond within {timeout_s:.0f}s") from exc

    return _parse_envelope(stdout, stderr, proc.returncode)
