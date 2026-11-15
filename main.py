#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, real free models.

The trial brief (MAKEFUN_PROMPT_FILE) carries the house's table of the
fourteen free OpenRouter models. This server:

  * serves POST /v1/chat/completions (OpenAI wire) and POST /v1/messages
    (Anthropic wire), each in JSON and in server-sent-events form,
  * serves GET /v1/models in the shape the caller's auth header names,
  * dials every completion through the trial gateway's Chat Completions
    route (the only route these model ids have), on the best-rated free
    model that meets what the request needs,
  * registers its port and catalog with the env, polls GET /work, and
    exits 0 when the trial is done.

Determinism: every choice below is a function of the brief, the request
being answered and the gateway's answers. No wall clock, no randomness:
stamps and ids come from fixed constants and a per-process counter that
the request sequence alone advances.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{3,12}-[0-9]{4})(?![0-9A-Z])")
FIXED_CREATED = 1767225600  # a fixed stamp, never a clock reading
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_CAP = 8192
UPSTREAM_TIMEOUT = 7.0
MAX_ATTEMPTS_PER_PROMPT = 4
FAILURES_HEADER = "x-makefun-upstream-failures"


# --------------------------------------------------------------------------
# The brief and the catalog
# --------------------------------------------------------------------------

def read_brief_models(path):
    """Pull the free-model table out of the brief's prose.

    The brief is text with JSON documents embedded in it; the model table
    is the document whose schema is MODELS_SCHEMA. Returns its model rows
    in brief order, or [] when there is no brief to read.
    """
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    marker = '"%s"' % MODELS_SCHEMA
    at = text.find(marker)
    if at < 0:
        return []
    start = text.rfind("{", 0, at)
    if start < 0:
        return []
    try:
        doc, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return []
    rows = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("id")]


def build_catalog(models):
    """The catalog document: one provider, the fourteen dialable ids."""
    rows = [
        {
            "id": row["id"],
            "free": True,
            "context_length": row.get("context_length"),
        }
        for row in models
    ]
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "key_env": "OPENROUTER_API_KEY",
                "wire": "openai",
                "models": rows,
            }
        ]
    }


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------

def text_of_content(content):
    """Flatten a wire content value (string or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def find_token(messages):
    """The prompt's unique token: the last token-shaped word it carries."""
    token = None
    for message in messages:
        if isinstance(message, dict):
            for match in TOKEN_RE.finditer(text_of_content(message.get("content"))):
                token = match.group(1)
    return token


# --------------------------------------------------------------------------
# Choosing a model and dialing the gateway
# --------------------------------------------------------------------------

def request_needs(body):
    """What a Chat-Completions-shaped request needs from its model."""
    needs = {"json": False, "tools": False, "vision": False}
    if isinstance(body.get("response_format"), dict):
        needs["json"] = True
    if isinstance(body.get("tools"), list) and body["tools"]:
        needs["tools"] = True
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    needs["vision"] = True
    return needs


def ranked_models(models, needs):
    """Free models that meet the needs, best house rating first.

    Ties break on the model id so the order never depends on anything but
    the brief. If no model qualifies, fall back to every model ranked the
    same way rather than serving nothing.
    """
    def qualifies(row):
        if needs["json"] and not row.get("json"):
            return False
        if needs["tools"] and not row.get("tools"):
            return False
        if needs["vision"] and not row.get("vision"):
            return False
        return True

    pool = [row for row in models if qualifies(row)] or list(models)
    return sorted(pool, key=lambda row: (-int(row.get("rating_milli") or 0), row["id"]))


def attempt_plan(ranked):
    """Best, best once more, then down the ranking; at most four calls."""
    plan = []
    for index in (0, 0, 1, 2):
        if index < len(ranked):
            plan.append(ranked[index])
        if len(plan) >= MAX_ATTEMPTS_PER_PROMPT:
            break
    return plan


def dial(body):
    """One Chat Completions call through the trial gateway.

    Returns (status, raw_text); status 0 means the call never got an HTTP
    answer (a transport error, which leaves no trail row behind).
    """
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base + "/chat/completions", data=payload, method="POST"
    )
    request.add_header("Authorization", "Bearer " + key)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return 0, ""


def error_code(raw):
    """The refusal's code (or type) from a gateway error body, if it has one."""
    try:
        doc = json.loads(raw)
        error = doc.get("error") if isinstance(doc, dict) else None
        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or ""
            return str(code)[:64]
    except json.JSONDecodeError:
        pass
    return ""


def completion_text(completion):
    """The assistant text of a Chat Completions answer ('' when there is none)."""
    try:
        choice = completion["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return text_of_content(message.get("content"))


def complete(body, models):
    """Serve one prompt: dial the plan until a real answer carries its token.

    body is a Chat-Completions-shaped request whose model is still "auto".
    Returns (completion, failures, dialed_id); completion is None when the
    plan gave up. failures holds one trail row per HTTP refusal, in order.
    """
    token = find_token(body.get("messages") or [])
    failures = []
    for candidate in attempt_plan(ranked_models(models, request_needs(body))):
        attempt = dict(body)
        attempt["model"] = candidate["id"]
        status, raw = dial(attempt)
        if status == 0:
            continue  # transport error: fail over silently, it leaves no row
        if status >= 400:
            failures.append(
                {"status": status, "code": error_code(raw), "model": candidate["id"]}
            )
            continue
        try:
            completion = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(completion, dict) or "choices" not in completion:
            continue
        if token is not None and token not in completion_text(completion):
            continue  # a real answer, but not to this prompt: earns nothing
        return completion, failures, candidate["id"]
    return None, failures, None


# --------------------------------------------------------------------------
# Wire translation
# --------------------------------------------------------------------------

def clamp_max_tokens(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_MAX_TOKENS
    return max(1, min(number, MAX_TOKENS_CAP))


def openai_upstream(client):
    """A Chat Completions client body, carried through to the gateway.

    The client's messages, system prompt, tools, response_format and
    sampling fields pass unchanged; only the model (chosen later), the
    output cap (always set, always within the gateway's ceiling) and the
    stream flag (the gateway is always dialed non-streamed; a streaming
    client is answered with synthesized events) are ours.
    """
    body = {"messages": client.get("messages") or []}
    body["max_tokens"] = clamp_max_tokens(
        client.get("max_tokens", client.get("max_completion_tokens"))
    )
    for key in (
        "temperature",
        "top_p",
        "stop",
        "response_format",
        "tools",
        "tool_choice",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ):
        if key in client:
            body[key] = client[key]
    body["stream"] = False
    return body


def anthropic_message_to_openai(message):
    """One Anthropic message as one or more Chat Completions messages."""
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": role, "content": content or ""}]
    parts = []
    tool_calls = []
    tool_results = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                url = "data:%s;base64,%s" % (
                    source.get("media_type", "image/png"),
                    source.get("data", ""),
                )
            else:
                url = source.get("url", "")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        elif kind == "tool_result":
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": text_of_content(block.get("content")),
                }
            )
    out = []
    if parts or tool_calls:
        carried = {"role": role, "content": parts if parts else ""}
        if tool_calls:
            carried["tool_calls"] = tool_calls
        out.append(carried)
    out.extend(tool_results)
    return out or [{"role": role, "content": ""}]


def anthropic_upstream(client):
    """An Anthropic Messages client body as a Chat Completions body."""
    messages = []
    system = client.get("system")
    if system is not None:
        messages.append(
            {"role": "system", "content": text_of_content(system)}
        )
    for message in client.get("messages") or []:
        if isinstance(message, dict):
            messages.extend(anthropic_message_to_openai(message))
    body = {"messages": messages}
    body["max_tokens"] = clamp_max_tokens(client.get("max_tokens"))
    for key in ("temperature", "top_p"):
        if key in client:
            body[key] = client[key]
    if client.get("stop_sequences"):
        body["stop"] = client["stop_sequences"]
    tools = client.get("tools")
    if isinstance(tools, list) and tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
            for tool in tools
            if isinstance(tool, dict)
        ]
    choice = client.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "any":
            body["tool_choice"] = "required"
        elif kind == "tool" and choice.get("name"):
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]},
            }
        elif kind == "auto":
            body["tool_choice"] = "auto"
    body["stream"] = False
    return body


STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def ensure_usage(completion):
    """Guarantee a whole usage object: an input of >= 0, an output of >= 1."""
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        completion["usage"] = usage
    if not isinstance(usage.get("prompt_tokens"), int):
        usage["prompt_tokens"] = 0
    if not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] < 1:
        usage["completion_tokens"] = max(1, len(completion_text(completion)) // 4)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def anthropic_message(completion, dialed, serial):
    """A Chat Completions answer re-shaped as an Anthropic message."""
    choice = completion["choices"][0]
    message = choice.get("message") or {}
    blocks = []
    text = completion_text(completion)
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "toolu_%04d" % serial,
                "name": function.get("name", ""),
                "input": arguments,
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    usage = ensure_usage(completion)
    return {
        "id": "msg_%04d" % serial,
        "type": "message",
        "role": "assistant",
        "model": dialed,
        "content": blocks,
        "stop_reason": STOP_REASON_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        },
    }


# --------------------------------------------------------------------------
# Streamed answers, synthesized from the gateway's whole answer
# --------------------------------------------------------------------------

def openai_event_stream(completion, dialed):
    """Chat Completions SSE: content chunk, finish chunk, usage, [DONE]."""
    choice = completion["choices"][0]
    message = choice.get("message") or {}
    base = {
        "id": completion.get("id") or "chatcmpl-agg",
        "object": "chat.completion.chunk",
        "created": completion.get("created") or FIXED_CREATED,
        "model": dialed,
    }
    frames = []
    delta = {"role": "assistant", "content": completion_text(completion)}
    if message.get("tool_calls"):
        delta["tool_calls"] = message["tool_calls"]
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
    frames.append("data: %s\n\n" % json.dumps(first))
    last = dict(base)
    last["choices"] = [
        {"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}
    ]
    frames.append("data: %s\n\n" % json.dumps(last))
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = ensure_usage(completion)
    frames.append("data: %s\n\n" % json.dumps(usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_event_stream(completion, dialed, serial):
    """Anthropic SSE: message_start, one block at a time, delta, stop."""
    message = anthropic_message(completion, dialed, serial)
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message["id"],
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": dialed,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": message["usage"]["input_tokens"],
                        "output_tokens": 0,
                    },
                },
            },
        )
    ]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            start_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start_block = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        events.append(
            (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start_block},
            )
        )
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
        )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": index})
        )
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(
        "event: %s\ndata: %s\n\n" % (name, json.dumps(payload))
        for name, payload in events
    )


# --------------------------------------------------------------------------
# The model lists
# --------------------------------------------------------------------------

def models_openai(models):
    return {
        "object": "list",
        "data": [
            {
                "id": row["id"],
                "object": "model",
                "created": FIXED_CREATED,
                "owned_by": "openrouter",
            }
            for row in models
        ],
    }


def models_anthropic(models):
    return {
        "data": [
            {
                "id": row["id"],
                "type": "model",
                "display_name": row.get("display_name") or row["id"],
                "created_at": "2026-01-01T00:00:00Z",
            }
            for row in models
        ],
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }


# --------------------------------------------------------------------------
# The HTTP server and the trial lifecycle
# --------------------------------------------------------------------------

class AggregatorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    # -- plumbing ----------------------------------------------------------

    def _send(self, status, body, content_type, failures=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if failures:
            self.send_header(FAILURES_HEADER, json.dumps(failures[:8]))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, value, failures=None):
        self._send(status, json.dumps(value), "application/json", failures)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _next_serial(self):
        self.server.serial += 1
        return self.server.serial

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._send_json(404, {"error": {"message": "no such route", "type": "not_found"}})
            return
        models = self.server.models
        if self.headers.get("x-api-key") is not None:
            self._send_json(200, models_anthropic(models))
        else:
            self._send_json(200, models_openai(models))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._answer_openai()
        elif path == "/v1/messages":
            self._answer_anthropic()
        else:
            self._send_json(404, {"error": {"message": "no such route", "type": "not_found"}})

    def _answer_openai(self):
        client = self._read_json_body()
        completion, failures, dialed = complete(
            openai_upstream(client), self.server.models
        )
        if completion is None:
            self._send_json(
                502,
                {
                    "error": {
                        "message": "every free model refused or missed the prompt",
                        "type": "upstream_error",
                        "code": "upstream_failed",
                    }
                },
                failures,
            )
            return
        ensure_usage(completion)
        completion["model"] = dialed  # the id dialed, prefix and all
        if client.get("stream"):
            self._send(200, openai_event_stream(completion, dialed), "text/event-stream", failures)
        else:
            self._send_json(200, completion, failures)

    def _answer_anthropic(self):
        client = self._read_json_body()
        completion, failures, dialed = complete(
            anthropic_upstream(client), self.server.models
        )
        if completion is None:
            self._send_json(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "every free model refused or missed the prompt",
                    },
                },
                failures,
            )
            return
        serial = self._next_serial()
        if client.get("stream"):
            body = anthropic_event_stream(completion, dialed, serial)
            self._send(200, body, "text/event-stream", failures)
        else:
            self._send_json(200, anthropic_message(completion, dialed, serial), failures)


def post_json(url, text):
    request = urllib.request.Request(
        url, data=text.encode("utf-8"), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode("utf-8", "replace")


def register(env_url, port, catalog_text):
    """Register the port and the catalog document with the env."""
    body = '{"protocol_version": 1, "port": %d, "catalog": %s}' % (port, catalog_text)
    for _ in range(40):
        try:
            status, _ = post_json(env_url.rstrip("/") + "/register", body)
            if status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def wait_until_done(env_url):
    """Poll GET /work until the env says the trial is done."""
    url = env_url.rstrip("/") + "/work"
    while True:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                doc = json.loads(response.read().decode("utf-8", "replace"))
            if isinstance(doc, dict) and doc.get("done") is True:
                return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.25)


def main(argv):
    models = read_brief_models(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    catalog_text = json.dumps(build_catalog(models))
    if "--list" in argv:
        sys.stdout.write(catalog_text + "\n")
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), AggregatorHandler)
    server.daemon_threads = True
    server.models = models
    server.serial = 0
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if env_url:
        register(env_url, server.server_address[1], catalog_text)
        wait_until_done(env_url)
        server.shutdown()
        server.server_close()
    else:
        worker.join()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
