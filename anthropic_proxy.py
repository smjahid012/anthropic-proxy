"""
anthropic_proxy.py v4.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claude Code → Free LLM proxy  (SMLabs AI)
  ✓ Native Gemini API        (thoughtSignature round-trip)
  ✓ OpenAI compat            (proper tool ID conversion)
  ✓ Anthropic native         (OpenCode Zen / KiloGateway passthrough)
  ✓ Multi-provider           (rotation + auto-fallback on 429)
  ✓ Token optimizer          (system trim, tool filter, msg limit)
  ✓ camelCase fix            (Gemini param name auto-correction)
  ✓ Explicit param map       (fixes Invalid tool parameters errors)
  ✓ Gemini sanitizer         (turn-order + orphan/duplicate tool fix)
  ✓ Groq reasoning fix       (delta.reasoning → reasoning_content)
  ✓ DeepSeek thinking        (reasoning_content multi-turn injection)
  ✓ anthropic-beta strip     (prevents 400s on OpenAI providers)
  ✓ Passthrough token fix    (trims body before forwarding)
  ✓ Mixed content fix        (text+tool_result no longer drops text)
  ✓ Disallowed tool scrub    (cleans orphan tool pairs from history)
  ✓ Mini-Claude mode         (injects Claude-like behavior into any model)
  ✓ Smart retry-after        (reads Gemini body wait time, not just headers)
  ✓ Safe message trim        (never cuts mid-tool-sequence → no more 400s)
  ✓ MCP tool prefix support  (mcp__server__tool matched by base name)
  ✓ Colored logging          (see exactly what's happening)
  ✓ config.json              (edit providers without restart)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install:  pip install fastapi uvicorn httpx
Run:      python anthropic_proxy.py
Config:   edit config.json  ← add MCP tool names to allowed_tools
"""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_FILE = Path("config.json")


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            print(f"Config error: {e}")
    return {}


CFG: dict = load_config()


def cfg(path: str, default=None):
    parts = path.split(".")
    node = CFG
    for p in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default if p == parts[-1] else {})
    return node if node != {} else default


PORT = cfg("server.port", 4000)
MAX_SYS_CHARS = cfg("token_limits.max_system_chars", 3000)
MAX_MESSAGES = cfg("token_limits.max_messages", 20)
MAX_TOOL_DESC = cfg("token_limits.max_tool_desc_chars", 80)
MAX_TOOL_RESULT = cfg("token_limits.max_tool_result_chars", 4000)
ALLOWED_TOOLS = set(
    cfg("allowed_tools", ["Read", "Edit", "Write", "Bash", "Glob", "Grep"])
)
MINI_CLAUDE_MODE = cfg("mini_claude_mode", True)


class ProviderRetry(Exception):
    """Raised when a provider returns 429 — endpoint returns proper error instead of streaming garbage."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
#  MINI-CLAUDE SYSTEM PROMPT
#  Injected as a PREFIX to every system prompt.
#  Forces Gemini/Groq/DeepSeek to behave like Claude Code — systematic,
#  tool-first, verify-before-done. Without this, free models answer immediately
#  without reading files, hallucinate contents, and never verify changes.
# ══════════════════════════════════════════════════════════════════════════════

MINI_CLAUDE_PREFIX = """You are an expert AI coding assistant operating inside Claude Code. You have access to tools to read, write, edit files, run bash commands, search code, and browse the web.

## Core Behavior Rules

**NEVER fabricate file contents.** Always Read a file before editing it. If you haven't seen it, you don't know what's in it.

**Think before acting.** Before using any tool, state your plan in 1-2 sentences. What are you checking and why?

**Work systematically, not randomly.** Follow this order for any coding task:
1. EXPLORE — Read relevant files, Grep for related code, understand the full picture
2. PLAN — State exactly what you will change and why
3. EXECUTE — Make precise, minimal edits
4. VERIFY — Read back what you changed, run tests if possible
5. SUMMARIZE — Report what was done and what to watch for

**Use multiple tools per turn when needed.** Don't stop after one Read. If fixing a bug requires reading 3 files first, read all 3, then plan, then edit.

**Verify your own work.** After editing a file, Read it back to confirm the change landed correctly. After running Bash, check the output for errors.

**Be surgical.** Only change what is broken. Do not refactor, rename, or "improve" unrelated code.

**Report tool results.** After each tool call, briefly explain what you found before proceeding.

**If a task is ambiguous**, ask one clarifying question before starting — don't guess and produce wrong output.

## Tool Usage Patterns

- **Finding code**: Grep first, then Read the specific file/line
- **Fixing bugs**: Read file → understand context → Edit → Read back → Bash test
- **Adding features**: Glob to find structure → Read relevant files → plan → Edit
- **Running commands**: Use Bash for installs, tests, builds — check exit codes
- **Long outputs**: Use head/tail/grep in Bash to avoid huge tool results

## Quality Standards

- Code you write must be correct on first attempt — read the schema/types before writing
- Error messages you produce must be actionable — include what file, what line, what to do
- When done, explicitly state: what changed, what wasn't changed, and any risks

"""

# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL COLORS + LOGGING
# ══════════════════════════════════════════════════════════════════════════════

R = "\033[91m"
G = "\033[92m"
Y = "\033[93m"
B = "\033[94m"
M = "\033[95m"
C = "\033[96m"
W = "\033[97m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def est_tokens(s: str) -> int:
    return len(s) // 4


def hr():
    print(f"{DIM}{'─' * 62}{RESET}")


def log_req(
    system: str,
    orig_sys_len: int,
    msgs_in: int,
    msgs_kept: int,
    tools_in: int,
    tools_out: int,
    est_tok: int,
    stream: bool,
    provider_name: str,
):
    hr()
    print(f"{BOLD}{C}[{ts()}] ← REQUEST  →  {M}{provider_name}{RESET}")
    sc = G if orig_sys_len <= MAX_SYS_CHARS else Y
    print(
        f"  {W}System{RESET}    : {sc}{orig_sys_len}→{len(system)} chars{RESET} {DIM}(~{orig_sys_len // 4}→~{est_tokens(system)} tok){RESET}"
    )
    tc = G if tools_out <= 6 else Y
    print(
        f"  {W}Tools{RESET}     : {tc}{tools_in}→{tools_out}{RESET}  {DIM}{sorted(ALLOWED_TOOLS)}{RESET}"
    )
    print(f"  {W}Messages{RESET}  : {msgs_in}→{msgs_kept}")
    ec = G if est_tok < 8000 else Y if est_tok < 20000 else R
    print(f"  {W}Est tokens{RESET}: {ec}~{est_tok}{RESET}  stream={stream}")


def log_provider(name: str, ptype: str, model: str, status: int):
    sc = G if status == 200 else R
    print(
        f"  {W}Provider{RESET}  : {B}{name}{RESET} ({DIM}{ptype}{RESET}) {sc}{status}{RESET}  model={M}{model}{RESET}"
    )


def log_resp(stop: str, tool_names: list, in_tok: int, out_tok: int, elapsed: float):
    print(
        f"  {W}Stop{RESET}      : {Y}{stop}{RESET}  tokens in={DIM}{in_tok}{RESET} out={G}{out_tok}{RESET}  {DIM}{elapsed:.2f}s{RESET}"
    )
    if tool_names:
        print(f"  {W}Tool calls{RESET}: {M}{tool_names}{RESET}")
    hr()


def log_err(msg: str):
    print(f"  {R}✗ {msg}{RESET}")
    hr()


def strip_null_values(d: dict) -> dict:
    """Remove None-valued keys from a dict (Groq rejects null fields in assistant messages)."""
    return {k: v for k, v in d.items() if v is not None}


# ══════════════════════════════════════════════════════════════════════════════
#  PROVIDER MANAGER
# ══════════════════════════════════════════════════════════════════════════════


class Provider:
    def __init__(self, c: dict):
        self.name = c["name"]
        self.type = c["type"]  # openai | gemini | anthropic
        self.api_key = c.get("api_key", "")
        self.base_url = c["base_url"].rstrip("/")
        self.model = c["model"]
        self.priority = c.get("priority", 99)
        self.headers = c.get("headers", {})
        self._rate_until = 0.0

    def available(self) -> bool:
        return time.time() > self._rate_until

    def rate_limit(self, secs: int = 60):
        self._rate_until = time.time() + secs
        print(f"  {R}⚠ {self.name} rate-limited for {secs}s{RESET}")

    def retry_after(self, headers: dict, body: bytes | str | dict | None = None) -> int:
        # 1. Try standard Retry-After header first
        ra = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        if ra:
            try:
                return max(1, int(float(str(ra))))
            except Exception:
                pass

        # 2. Gemini puts retry time INSIDE the error body, not headers
        # e.g. "Please retry in 2.355263732s"
        if body:
            try:
                text = (
                    body
                    if isinstance(body, str)
                    else (
                        json.dumps(body)
                        if isinstance(body, dict)
                        else body.decode(errors="ignore")
                    )
                )
                m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
                if m:
                    return max(1, int(float(m.group(1))) + 1)
            except Exception:
                pass

        # 3. Default fallback
        return 60

    def is_deepseek(self) -> bool:
        return "deepseek" in self.name.lower() or "deepseek" in self.model.lower()

    def needs_reasoning_inject(self) -> bool:
        """DeepSeek thinking mode requires reasoning_content on all prior assistant turns."""
        return self.is_deepseek()

    def strip_null_fields(self) -> bool:
        """Groq rejects assistant messages with null-valued keys."""
        return "groq" in self.name.lower() or "groq" in (self.base_url or "").lower()


class ProviderManager:
    def __init__(self):
        self.providers = sorted(
            [Provider(p) for p in cfg("providers", [])],
            key=lambda p: p.priority,
        )
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> "Provider | None":
        async with self._lock:
            avail = [p for p in self.providers if p.available()]
            if not avail:
                soonest = min(self.providers, key=lambda p: p._rate_until)
                wait = max(0, soonest._rate_until - time.time())
                print(
                    f"  {Y}All providers rate-limited. Waiting {wait:.0f}s for {soonest.name}...{RESET}"
                )
                await asyncio.sleep(wait + 0.5)
                avail = [p for p in self.providers if p.available()]
                if not avail:
                    return None
            p = avail[self._idx % len(avail)]
            self._idx += 1
            return p

    def status(self) -> str:
        return "  ".join(
            f"{(G if p.available() else R)}{p.name}{RESET}" for p in self.providers
        )


PM: ProviderManager = ProviderManager()


async def _connect_gemini(gemini_body: dict, provider: Provider) -> tuple:
    """Try to connect to Gemini streaming. Returns (resp, client) on 200, raises ProviderRetry on 429."""
    url = f"{provider.base_url}/models/{provider.model}:streamGenerateContent?alt=sse"
    headers = {
        "x-goog-api-key": provider.api_key,
        "Content-Type": "application/json",
        **provider.headers,
    }
    client = httpx.AsyncClient(timeout=120)
    try:
        req = client.build_request("POST", url, json=gemini_body, headers=headers)
        resp = await client.send(req, stream=True)
        log_provider(provider.name, provider.type, provider.model, resp.status_code)
        if resp.status_code == 429:
            err = (await resp.aread()).decode()[:500]
            provider.rate_limit(provider.retry_after(dict(resp.headers), err))
            await resp.aclose()
            await client.aclose()
            raise ProviderRetry(f"Gemini 429: {err[:200]}")
        if resp.status_code != 200:
            err = (await resp.aread()).decode()[:500]
            await resp.aclose()
            raise Exception(f"Gemini {resp.status_code}: {err}")
        return resp, client
    except ProviderRetry:
        raise
    except Exception:
        await client.aclose()
        raise


async def _connect_openai(body: dict, provider: Provider) -> tuple:
    """Try to connect to OpenAI-compat streaming. Returns (resp, client) on 200, raises ProviderRetry on 429."""
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        **provider.headers,
    }
    client = httpx.AsyncClient(timeout=120)
    try:
        req = client.build_request(
            "POST",
            f"{provider.base_url}/chat/completions",
            json={**body, "stream": True},
            headers=headers,
        )
        resp = await client.send(req, stream=True)
        log_provider(provider.name, provider.type, provider.model, resp.status_code)
        if resp.status_code == 429:
            err = (await resp.aread()).decode()[:500]
            provider.rate_limit(provider.retry_after(dict(resp.headers), err))
            await resp.aclose()
            await client.aclose()
            raise ProviderRetry(f"OpenAI 429: {err[:200]}")
        if resp.status_code != 200:
            err = (await resp.aread()).decode()[:500]
            await resp.aclose()
            raise Exception(f"OpenAI {resp.status_code}: {err}")
        return resp, client
    except ProviderRetry:
        raise
    except Exception:
        await client.aclose()
        raise


async def _connect_anthropic(raw_body: bytes, provider: Provider) -> tuple:
    """Try to connect to Anthropic-native streaming. Returns (resp, client) on 200, raises ProviderRetry on 429."""
    headers = {
        "x-api-key": provider.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        **provider.headers,
    }
    client = httpx.AsyncClient(timeout=120)
    try:
        req = client.build_request(
            "POST",
            f"{provider.base_url}/v1/messages",
            content=raw_body,
            headers=headers,
        )
        resp = await client.send(req, stream=True)
        log_provider(provider.name, provider.type, provider.model, resp.status_code)
        if resp.status_code == 429:
            err = (await resp.aread()).decode()[:500]
            provider.rate_limit(provider.retry_after(dict(resp.headers), err))
            await resp.aclose()
            await client.aclose()
            raise ProviderRetry(f"Anthropic 429: {err[:200]}")
        if resp.status_code != 200:
            err = (await resp.aread()).decode()[:500]
            await resp.aclose()
            raise Exception(f"Anthropic {resp.status_code}: {err}")
        return resp, client
    except ProviderRetry:
        raise
    except Exception:
        await client.aclose()
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════


def trim_system(system: Any) -> str:
    if not system:
        raw = ""
    elif isinstance(system, list):
        raw = " ".join(
            b.get("text", "")
            for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        raw = str(system)

    # Truncate the original Claude Code system prompt to budget
    truncated = (
        raw[:MAX_SYS_CHARS] + "\n[truncated]" if len(raw) > MAX_SYS_CHARS else raw
    )

    # Prepend mini-claude behavior prompt so free models act like Claude Code
    if MINI_CLAUDE_MODE:
        return MINI_CLAUDE_PREFIX + truncated
    return truncated


def _tool_name(t) -> str:
    return t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")


def _tool_allowed(name: str) -> bool:
    """
    Check if a tool name is allowed.
    Handles both bare names ('search_web') and MCP-prefixed names
    ('mcp__serper-search__search_web') — extracts the base tool name
    after the last '__' and checks that against ALLOWED_TOOLS.
    If ALLOWED_TOOLS is empty, allow everything.
    """
    if not ALLOWED_TOOLS:
        return True
    if name in ALLOWED_TOOLS:
        return True
    # MCP tool format: mcp__{server}__{tool_name}
    if "__" in name:
        base = name.rsplit("__", 1)[-1]
        if base in ALLOWED_TOOLS:
            return True
    return False


def filter_tools(tools: list | None) -> list:
    if not tools:
        return []
    return [t for t in tools if _tool_allowed(_tool_name(t))]


def trim_messages(messages: list) -> list:
    """
    Trim message history to MAX_MESSAGES but NEVER cut mid-tool-sequence.

    The naive messages[-N:] slice can cut like this:
      [... assistant:tool_use, user:tool_result]  ← keeps result, drops call
    Gemini sees a functionResponse with no functionCall → 400 INVALID_ARGUMENT.

    This function finds the safe cut point: always starts on a clean
    user text turn (no tool_result), never in the middle of a tool pair.
    """
    if len(messages) <= MAX_MESSAGES:
        return messages

    # Start from the naive slice point and walk forward until we land on a
    # clean user message (plain text, no tool_result blocks).
    start = len(messages) - MAX_MESSAGES
    while start < len(messages):
        msg = messages[start]
        role = msg.get("role", "")
        content = msg.get("content", "")
        # A safe start is a user message that contains NO tool_result blocks
        if role == "user":
            if isinstance(content, str):
                break  # plain string user message — safe
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if not has_tool_result:
                    break  # user message with only text — safe
        # Otherwise this is a tool_result turn or assistant turn — not safe to start here
        start += 1

    # If we walked past all messages, just keep the last 4 to avoid sending nothing
    if start >= len(messages):
        start = max(0, len(messages) - 4)

    return messages[start:]


def scrub_disallowed_tool_history(messages: list) -> list:
    """
    Mini-claude behavior: if a disallowed tool was called in history,
    scrub its tool_use + tool_result pair so the model doesn't see a
    broken history (tool_use with no matching result, or vice versa).

    Also: convert thinking blocks in history to text blocks for providers
    that can't handle them (everything except Anthropic native).
    Called before Gemini/OpenAI conversion.
    """
    if not messages:
        return messages

    # Collect IDs of tool_use blocks for disallowed tools in assistant messages
    disallowed_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                name = b.get("name", "")
                if not _tool_allowed(name):
                    disallowed_ids.add(b.get("id", ""))

    if not disallowed_ids:
        return messages

    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        new_content = []
        for b in content:
            if not isinstance(b, dict):
                new_content.append(b)
                continue
            btype = b.get("type", "")
            # Strip disallowed tool_use from assistant turns
            if btype == "tool_use" and b.get("id", "") in disallowed_ids:
                continue
            # Strip matching tool_result from user turns
            if btype == "tool_result" and b.get("tool_use_id", "") in disallowed_ids:
                continue
            new_content.append(b)

        if new_content:
            cleaned.append({**msg, "content": new_content})
        # if message is now empty, skip it entirely
    return cleaned


def trim_tool_result(text: str) -> str:
    return (
        text[:MAX_TOOL_RESULT] + "\n[trimmed]" if len(text) > MAX_TOOL_RESULT else text
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CONVERTERS  (ported + improved from UniClaudeProxy)
# ══════════════════════════════════════════════════════════════════════════════

THOUGHT_SIG_SEP = "__ts__"
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()


def _build_param_index(tools: list) -> dict[str, list[str]]:
    """Build tool_name → [param_names] for camelCase auto-fix."""
    idx = {}
    for t in tools:
        name = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
        schema = (
            t.get("input_schema", {})
            if isinstance(t, dict)
            else getattr(t, "input_schema", {})
        )
        idx[name] = list(
            (schema.get("properties", {}) if isinstance(schema, dict) else {}).keys()
        )
    return idx


# Explicit fallback mappings — catches Gemini wrong param names before camelCase fix
TOOL_PARAM_MAP: dict[str, dict[str, str]] = {
    "Bash": {
        "cmd": "command",
        "shell": "command",
        "shell_command": "command",
        "Command": "command",
        "bash_command": "command",
    },
    "Read": {
        "path": "file_path",
        "filepath": "file_path",
        "filename": "file_path",
        "file": "file_path",
    },
    "Write": {
        "path": "file_path",
        "filepath": "file_path",
        "filename": "file_path",
        "content": "file_text",
        "text": "file_text",
    },
    "Edit": {
        "path": "file_path",
        "filepath": "file_path",
        "filename": "file_path",
        "file": "file_path",
    },
    "MultiEdit": {"path": "file_path", "filepath": "file_path"},
    "Glob": {"pattern": "pattern", "glob": "pattern"},
    "Grep": {
        "query": "pattern",
        "search": "pattern",
        "regex": "pattern",
        "search_pattern": "pattern",
    },
    "LS": {"path": "path", "directory": "path", "dir": "path"},
    "WebSearch": {
        "query": "query",
        "search_query": "query",
        "q": "query",
        "search": "query",
    },
    "WebFetch": {"url": "url", "link": "url", "uri": "url"},
    "TodoWrite": {"todos": "todos", "tasks": "todos", "items": "todos"},
    "Task": {
        "description": "description",
        "prompt": "description",
        "task": "description",
    },
}


def _fix_tool_args(tool_name: str, args: dict, param_index: dict) -> dict:
    """
    Auto-fix Gemini wrong/camelCase param names → expected param names.
    Step 1: Apply explicit known mappings (catches most Gemini mistakes)
    Step 2: camelCase → snake_case conversion fallback
    Step 3: Positional matching for any remaining unmatched params
    """
    if not args:
        return args

    # Step 1 — explicit mapping for known tool param name mistakes
    if tool_name in TOOL_PARAM_MAP:
        mapping = TOOL_PARAM_MAP[tool_name]
        remapped = {}
        for k, v in args.items():
            remapped[mapping.get(k, k)] = v
        args = remapped

    # Step 2 — camelCase fix using param index
    if not param_index or tool_name not in param_index:
        return args
    expected = param_index[tool_name]
    if not expected:
        return args

    expected_set = set(expected)
    snake_lookup = {_camel_to_snake(p): p for p in expected}
    fixed, unmatched = {}, {}

    for k, v in args.items():
        if k in expected_set:
            fixed[k] = v
        elif _camel_to_snake(k) in snake_lookup:
            real = snake_lookup[_camel_to_snake(k)]
            if real not in fixed:
                fixed[real] = v
            else:
                unmatched[k] = v
        else:
            unmatched[k] = v

    # Step 3 — positional matching for remaining unmatched
    if unmatched:
        rem = [p for p in expected if p not in fixed]
        if len(unmatched) == len(rem):
            for (sk, sv), ep in zip(unmatched.items(), rem):
                fixed[ep] = sv
        else:
            fixed.update(unmatched)

    return fixed


_GEMINI_STRIP = frozenset(
    {
        "$schema",
        "$id",
        "$ref",
        "$comment",
        "$defs",
        "additionalProperties",
        "propertyNames",
        "patternProperties",
        "definitions",
        "examples",
        "default",
        "const",
        "if",
        "then",
        "else",
        "not",
        "anyOf",
        "any_of",
        "oneOf",
        "one_of",
        "allOf",
        "all_of",
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "title",
    }
)
_GEMINI_KEEP = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
        "format",
        "nullable",
    }
)


def _clean_schema(schema: Any) -> dict:
    """Strip non-Gemini JSON Schema fields recursively."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _GEMINI_STRIP or k not in _GEMINI_KEEP:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        elif isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    if "required" in out and "properties" in out:
        valid = set(out["properties"].keys())
        out["required"] = [r for r in out["required"] if r in valid]
        if not out["required"]:
            del out["required"]
    if "type" not in out and "properties" in out:
        out["type"] = "object"
    return out


def _to_gemini_tools(tools: list) -> list:
    """Convert filtered Anthropic tools → Gemini functionDeclarations."""
    decls = []
    for t in tools:
        name = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
        desc = (
            t.get("description", "")
            if isinstance(t, dict)
            else getattr(t, "description", "")
        )
        schema = (
            t.get("input_schema", {})
            if isinstance(t, dict)
            else getattr(t, "input_schema", {})
        )
        desc = desc[:MAX_TOOL_DESC]
        decls.append(
            {"name": name, "description": desc, "parameters": _clean_schema(schema)}
        )
    return [{"functionDeclarations": decls}] if decls else []


def _to_gemini_contents(messages: list, tool_id_to_name: dict) -> list:
    """Convert Anthropic messages → Gemini contents array."""
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for b in content:
                    bt = b.get("type", "") if isinstance(b, dict) else ""
                    if bt == "text" and b.get("text"):
                        parts.append({"text": b["text"]})
                    elif bt == "tool_use":
                        raw_id = b.get("id", "")
                        tool_id_to_name[raw_id] = b.get("name", "")
                        part: dict = {
                            "functionCall": {
                                "name": b.get("name", ""),
                                "args": b.get("input", {}),
                            }
                        }
                        if THOUGHT_SIG_SEP in raw_id:
                            _, enc = raw_id.split(THOUGHT_SIG_SEP, 1)
                            part["thoughtSignature"] = unquote(enc)
                        parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})

        else:  # user
            if isinstance(content, str):
                if content:
                    contents.append({"role": "user", "parts": [{"text": content}]})
            elif isinstance(content, list):
                user_parts, tool_parts = [], []
                for b in content:
                    bt = b.get("type", "") if isinstance(b, dict) else ""
                    if bt == "text" and b.get("text"):
                        user_parts.append({"text": b["text"]})
                    elif bt == "tool_result":
                        raw = b.get("content", "")
                        if isinstance(raw, list):
                            raw = " ".join(
                                r.get("text", "")
                                for r in raw
                                if isinstance(r, dict) and r.get("type") == "text"
                            )
                        raw = trim_tool_result(str(raw))
                        raw_id = b.get("tool_use_id", "")
                        fn_name = tool_id_to_name.get(raw_id, raw_id)
                        tool_parts.append(
                            {
                                "functionResponse": {
                                    "name": fn_name,
                                    "response": {"result": raw},
                                }
                            }
                        )
                if tool_parts:
                    contents.append({"role": "user", "parts": tool_parts})
                # FIX: don't create a second separate user turn for text after tool results.
                # Gemini handles mixed functionResponse + text in one user turn fine.
                # Two consecutive user turns get merged by _sanitize anyway, but putting
                # them together here is cleaner and avoids the sanitizer dropping things.
                if user_parts:
                    if tool_parts and contents and contents[-1].get("role") == "user":
                        # append text to the just-added tool turn
                        contents[-1]["parts"].extend(user_parts)
                    else:
                        contents.append({"role": "user", "parts": user_parts})
    return contents


def _sanitize_gemini_contents(contents: list) -> list:
    """
    Gemini requires strict alternating user/model turns with valid tool sequencing.
    Fixes conversation turn order after message trimming cuts mid-sequence.

    Steps:
      1. Drop leading model turns (must start with user)
      2. Merge consecutive same-role turns into one
      3. Case A: orphaned tool_use with no matching functionResponse → inject dummy result
      4. Case B: orphaned functionResponse with no prior functionCall → drop it
      5. Case D: duplicate functionResponse for same name → keep last only
      6. Remove model turns whose functionCall has invalid preceding context
    """
    if not contents:
        return contents

    # Make a copy so we don't mutate the caller's list
    contents = list(contents)

    # Step 1 — must start with user
    while contents and contents[0].get("role") != "user":
        contents.pop(0)
    if not contents:
        return contents

    # Step 2 — merge consecutive same-role turns
    merged = [contents[0]]
    for curr in contents[1:]:
        prev = merged[-1]
        if curr.get("role") == prev.get("role"):
            prev["parts"] = prev.get("parts", []) + curr.get("parts", [])
        else:
            merged.append(curr)

    # Step 3/4/5 — collect all functionCall names seen so far,
    # fix orphaned tool results and duplicate responses
    seen_fn_calls: set[str] = set()
    result = []
    for turn in merged:
        role = turn.get("role")
        parts = turn.get("parts", [])

        if role == "model":
            fc_names = [p["functionCall"]["name"] for p in parts if "functionCall" in p]
            seen_fn_calls.update(fc_names)
            result.append(turn)

        elif role == "user":
            fr_parts = [p for p in parts if "functionResponse" in p]
            other_parts = [p for p in parts if "functionResponse" not in p]

            # Case B — drop functionResponses with no prior functionCall
            valid_frs = [
                p for p in fr_parts if p["functionResponse"]["name"] in seen_fn_calls
            ]

            # Case D — deduplicate: keep last functionResponse per name
            seen_fr: dict[str, dict] = {}
            for p in valid_frs:
                seen_fr[p["functionResponse"]["name"]] = p
            deduped_frs = list(seen_fr.values())

            new_parts = other_parts + deduped_frs
            if new_parts:
                result.append({"role": "user", "parts": new_parts})

    # Case A — inject dummy functionResponse for any functionCall that got no response
    final = []
    pending_fcs: list[str] = []
    for turn in result:
        role = turn.get("role")
        parts = turn.get("parts", [])
        if role == "model":
            pending_fcs = [
                p["functionCall"]["name"] for p in parts if "functionCall" in p
            ]
            final.append(turn)
        elif role == "user":
            responded = {
                p["functionResponse"]["name"] for p in parts if "functionResponse" in p
            }
            missing = [fn for fn in pending_fcs if fn not in responded]
            if missing:
                dummy_parts = [
                    {"functionResponse": {"name": fn, "response": {"result": ""}}}
                    for fn in missing
                ]
                # inject dummies before this user turn's other parts
                fr_parts = [p for p in parts if "functionResponse" in p]
                other_parts = [p for p in parts if "functionResponse" not in p]
                turn = {"role": "user", "parts": fr_parts + dummy_parts + other_parts}
            pending_fcs = []
            final.append(turn)

    # Step 6 — validate functionCall placement (must follow user or functionResponse)
    validated = []
    for turn in final:
        has_fc = any("functionCall" in p for p in turn.get("parts", []))
        if has_fc and turn.get("role") == "model":
            if not validated:
                continue
            prev = validated[-1]
            prev_is_user = prev.get("role") == "user"
            prev_has_fr = any("functionResponse" in p for p in prev.get("parts", []))
            if not (prev_is_user or prev_has_fr):
                continue
        validated.append(turn)

    return validated


def to_gemini_request(body: dict, system: str, messages: list, tools: list, preseeding: dict | None = None) -> dict:
    """Build full Gemini generateContent request body."""
    tool_id_to_name: dict[str, str] = {}
    if preseeding:
        tool_id_to_name.update(preseeding)
    contents = _to_gemini_contents(messages, tool_id_to_name)
    contents = _sanitize_gemini_contents(contents)  # ← fixes 400 turn-order errors

    req: dict = {"contents": contents}
    if system:
        req["systemInstruction"] = {"parts": [{"text": system}]}
    gen_cfg: dict = {}
    if body.get("max_tokens"):
        gen_cfg["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        gen_cfg["temperature"] = body["temperature"]
    if gen_cfg:
        req["generationConfig"] = gen_cfg
    gemini_tools = _to_gemini_tools(tools)
    if gemini_tools:
        req["tools"] = gemini_tools
    return req


def _gemini_finish_to_stop(reason: str | None) -> str:
    return {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
    }.get(reason or "STOP", "end_turn")


def _new_tool_id(sig: str = "") -> str:
    base = f"toolu_{uuid.uuid4().hex[:24]}"
    return f"{base}{THOUGHT_SIG_SEP}{quote(sig, safe='')}" if sig else base


# ══════════════════════════════════════════════════════════════════════════════
#  OPENAI CONVERTERS
# ══════════════════════════════════════════════════════════════════════════════


def _fix_tool_id(tc_id: str) -> str:
    """Normalize any tool ID → toolu_ prefix."""
    if tc_id.startswith("toolu_"):
        return tc_id
    if tc_id.startswith("call_"):
        return f"toolu_{tc_id[5:]}"
    if tc_id.startswith("fc_"):
        return f"toolu_{tc_id[3:]}"
    return f"toolu_{tc_id}"


def to_openai_messages(
    messages: list, system: str, inject_reasoning: bool = False
) -> list:
    """
    Convert Anthropic messages → OpenAI format.
    inject_reasoning=True: preserve reasoning_content on assistant turns (DeepSeek thinking mode).
    """
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        text_parts, tool_calls, tool_results = [], [], []
        reasoning_text = None
        for b in content:
            bt = b.get("type", "") if isinstance(b, dict) else ""
            if bt == "text":
                text_parts.append(b.get("text", ""))
            elif bt == "thinking":
                # preserve thinking block as reasoning_content for DeepSeek
                reasoning_text = b.get("thinking", "")
            elif bt == "tool_use":
                tool_calls.append(
                    {
                        "id": _fix_tool_id(
                            b.get("id", f"toolu_{uuid.uuid4().hex[:8]}")
                        ),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    }
                )
            elif bt == "tool_result":
                raw = b.get("content", "")
                if isinstance(raw, list):
                    raw = " ".join(
                        r.get("text", "")
                        for r in raw
                        if isinstance(r, dict) and r.get("type") == "text"
                    )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": _fix_tool_id(b.get("tool_use_id", "")),
                        "content": trim_tool_result(str(raw)),
                    }
                )
        if tool_results:
            result.extend(tool_results)
            # FIX: if user message has BOTH tool_results AND text (e.g. follow-up instruction),
            # append the text as a separate user message — don't silently drop it.
            if text_parts:
                result.append({"role": "user", "content": " ".join(text_parts)})
        elif tool_calls:
            asst_msg: dict = {
                "role": "assistant",
                "content": " ".join(text_parts) or None,
                "tool_calls": tool_calls,
            }
            # DeepSeek requires reasoning_content on every prior assistant turn
            if inject_reasoning and reasoning_text is not None:
                asst_msg["reasoning_content"] = reasoning_text
            elif inject_reasoning:
                asst_msg["reasoning_content"] = " "  # minimum DeepSeek accepts
            result.append(asst_msg)
        else:
            asst_msg = {"role": role, "content": " ".join(text_parts)}
            if inject_reasoning and role == "assistant" and reasoning_text is not None:
                asst_msg["reasoning_content"] = reasoning_text
            elif inject_reasoning and role == "assistant":
                asst_msg["reasoning_content"] = " "
            result.append(asst_msg)
    return result


def to_openai_tools(tools: list) -> list | None:
    if not tools:
        return None
    out = []
    for t in tools:
        name = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
        desc = (
            t.get("description", "")
            if isinstance(t, dict)
            else getattr(t, "description", "")
        )
        schema = (
            _clean_schema(t.get("input_schema", {}))
            if isinstance(t, dict)
            else _clean_schema(getattr(t, "input_schema", {}))
        )
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc[:MAX_TOOL_DESC],
                    "parameters": schema,
                },
            }
        )
    return out or None


def openai_response_to_anthropic(oai: dict, model: str) -> dict:
    choice = oai["choices"][0]
    message = choice.get("message", {})
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        f = tc.get("function", {})
        try:
            args = json.loads(f.get("arguments", "{}"))
        except Exception:
            args = {}
        content.append(
            {
                "type": "tool_use",
                "id": _fix_tool_id(tc.get("id", "")),
                "name": f.get("name", ""),
                "input": args,
            }
        )
    fr = choice.get("finish_reason", "stop")
    stop = (
        "max_tokens"
        if fr == "length"
        else "tool_use"
        if fr in ("tool_calls", "function_call")
        or any(c["type"] == "tool_use" for c in content)
        else "end_turn"
    )
    usage = oai.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SSE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_message_start(model: str, msg_id: str, in_tok: int = 0) -> str:
    return sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": 0},
            },
        },
    )


def sse_block_start(idx: int, block_type: str, **kw) -> str:
    if block_type == "text":
        cb = {"type": "text", "text": ""}
    elif block_type == "thinking":
        cb = {"type": "thinking", "thinking": ""}
    else:
        cb = {
            "type": "tool_use",
            "id": kw.get("id", ""),
            "name": kw.get("name", ""),
            "input": {},
        }
    return sse(
        "content_block_start",
        {"type": "content_block_start", "index": idx, "content_block": cb},
    )


def sse_text_delta(idx: int, text: str) -> str:
    return sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def sse_thinking_delta(idx: int, text: str) -> str:
    return sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    )


def sse_json_delta(idx: int, partial: str) -> str:
    return sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        },
    )


def sse_block_stop(idx: int) -> str:
    return sse("content_block_stop", {"type": "content_block_stop", "index": idx})


def sse_msg_delta(stop: str, out_tok: int) -> str:
    return sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": out_tok},
        },
    )


def sse_msg_stop() -> str:
    return sse("message_stop", {"type": "message_stop"})


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING HANDLERS
# ══════════════════════════════════════════════════════════════════════════════


async def stream_openai(
    resp, client, provider: Provider, start: float, param_index: dict
):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage = {"input_tokens": 0, "output_tokens": 0}
    finish = "stop"
    tc_map: dict[int, dict] = {}
    text_open = False
    text_idx = -1
    nxt_idx = 0
    char_count = 0
    tool_names_logged: list[str] = []

    yield sse_message_start(provider.model, msg_id)
    yield sse("ping", {"type": "ping"})

    try:
        print(f"  {G}Streaming...{RESET}", end="", flush=True)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Groq sends reasoning as "reasoning", normalize to "reasoning_content"
                if "reasoning" in delta and "reasoning_content" not in delta:
                    delta["reasoning_content"] = delta.pop("reasoning")

                if delta.get("content"):
                    if not text_open:
                        text_idx = nxt_idx
                        nxt_idx += 1
                        yield sse_block_start(text_idx, "text")
                        text_open = True
                    char_count += len(delta["content"])
                    yield sse_text_delta(text_idx, delta["content"])

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in tc_map:
                        tc_map[idx] = {
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "name": "",
                            "arguments": "",
                        }
                    if not tc_map[idx]["name"]:
                        tc_map[idx]["name"] = tc.get("function", {}).get("name", "")
                    tc_map[idx]["arguments"] += tc.get("function", {}).get(
                        "arguments", ""
                    )

                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if chunk.get("usage"):
                    usage = {
                        "input_tokens": chunk["usage"].get("prompt_tokens", 0),
                        "output_tokens": chunk["usage"].get("completion_tokens", 0),
                    }
            except Exception:
                pass

        print(f" {G}done{RESET} ({char_count} chars)")

    except Exception as e:
        if not text_open:
            text_idx = nxt_idx
            nxt_idx += 1
            yield sse_block_start(text_idx, "text")
            text_open = True
        yield sse_text_delta(text_idx, f"\n[Proxy error: {e}]")
        log_err(str(e))
    finally:
        await resp.aclose()
        await client.aclose()

    if text_open:
        yield sse_block_stop(text_idx)

    nxt = text_idx + (1 if text_open else 0)
    for i, tc in enumerate(sorted(tc_map.values(), key=lambda x: x["id"])):
        bidx = nxt + i
        tid = _fix_tool_id(tc["id"])
        try:
            inp = json.loads(tc["arguments"] or "{}")
        except Exception:
            inp = {}
        if param_index:
            inp = _fix_tool_args(tc["name"], inp, param_index)
        tool_names_logged.append(tc["name"])
        print(f"  {M}Tool call{RESET} : {tc['name']}({list(inp.keys())})")
        yield sse_block_start(bidx, "tool_use", id=tid, name=tc["name"])
        yield sse_json_delta(bidx, json.dumps(inp))
        yield sse_block_stop(bidx)

    has_tools = bool(tc_map)
    stop = (
        "tool_use"
        if has_tools or finish in ("tool_calls", "function_call")
        else "max_tokens"
        if finish == "length"
        else "end_turn"
    )
    yield sse_msg_delta(stop, usage["output_tokens"])
    yield sse_msg_stop()
    log_resp(
        stop,
        tool_names_logged,
        usage["input_tokens"],
        usage["output_tokens"],
        time.time() - start,
    )


async def stream_gemini(
    resp, client, provider: Provider, anthr_model: str, param_index: dict, start: float
):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    nxt_idx = 0
    text_idx = -1
    text_open = False
    think_idx = -1
    think_open = False
    has_tools = False
    usage = {"input_tokens": 0, "output_tokens": 0}
    finish = "STOP"
    tool_names_logged: list[str] = []

    yield sse_message_start(anthr_model, msg_id)
    yield sse("ping", {"type": "ping"})

    try:
        print(f"  {G}Streaming Gemini...{RESET}", end="", flush=True)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
            except Exception:
                continue

            um = data.get("usageMetadata", {})
            if um:
                usage = {
                    "input_tokens": um.get("promptTokenCount", 0),
                    "output_tokens": um.get("candidatesTokenCount", 0),
                }

            cands = data.get("candidates", [])
            if not cands:
                continue
            cand = cands[0]
            if cand.get("finishReason"):
                finish = cand["finishReason"]

            for part in cand.get("content", {}).get("parts", []):
                if part.get("thought") and "text" in part:
                    if text_open:
                        yield sse_block_stop(text_idx)
                        text_open = False
                    if not think_open:
                        think_idx = nxt_idx
                        nxt_idx += 1
                        yield sse_block_start(think_idx, "thinking")
                        think_open = True
                    yield sse_thinking_delta(think_idx, part["text"])

                elif "text" in part and not part.get("thought"):
                    if think_open:
                        yield sse_block_stop(think_idx)
                        think_open = False
                    if not text_open:
                        text_idx = nxt_idx
                        nxt_idx += 1
                        yield sse_block_start(text_idx, "text")
                        text_open = True
                    yield sse_text_delta(text_idx, part["text"])

                elif "functionCall" in part:
                    has_tools = True
                    if think_open:
                        yield sse_block_stop(think_idx)
                        think_open = False
                    if text_open:
                        yield sse_block_stop(text_idx)
                        text_open = False
                    fc = part["functionCall"]
                    sig = part.get("thoughtSignature", "")
                    tid = _new_tool_id(sig)
                    fn = fc.get("name", "")
                    args = fc.get("args", {})
                    if param_index:
                        args = _fix_tool_args(fn, args, param_index)
                    bidx = nxt_idx
                    nxt_idx += 1
                    tool_names_logged.append(fn)
                    print(f"  {M}Tool call{RESET} : {fn}({list(args.keys())})")
                    yield sse_block_start(bidx, "tool_use", id=tid, name=fn)
                    yield sse_json_delta(bidx, json.dumps(args))
                    yield sse_block_stop(bidx)

        print(f" {G}done{RESET}")

    except Exception as e:
        if think_open:
            yield sse_block_stop(think_idx)
            think_open = False
        if not text_open:
            text_idx = nxt_idx
            nxt_idx += 1
            yield sse_block_start(text_idx, "text")
            text_open = True
        yield sse_text_delta(text_idx, f"\n[Proxy error: {e}]")
        log_err(str(e))
    finally:
        await resp.aclose()
        await client.aclose()

    if think_open:
        yield sse_block_stop(think_idx)
    if text_open:
        yield sse_block_stop(text_idx)

    stop = "tool_use" if has_tools else _gemini_finish_to_stop(finish)
    yield sse_msg_delta(stop, usage["output_tokens"])
    yield sse_msg_stop()
    log_resp(
        stop,
        tool_names_logged,
        usage["input_tokens"],
        usage["output_tokens"],
        time.time() - start,
    )


async def stream_anthropic_passthrough(resp, client, provider: Provider, start: float):
    """Pass-through for Anthropic-native providers from pre-connected response."""
    try:
        print(f"  {G}Passthrough streaming...{RESET}", end="", flush=True)
        async for chunk in resp.aiter_bytes():
            yield chunk
        print(f" {G}done{RESET}")
    except Exception as e:
        log_err(str(e))
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'proxy_error', 'message': str(e)}})}\n\n".encode()
    finally:
        await resp.aclose()
        await client.aclose()
    log_resp("passthrough", [], 0, 0, time.time() - start)


# ══════════════════════════════════════════════════════════════════════════════
#  NON-STREAMING HANDLERS
# ══════════════════════════════════════════════════════════════════════════════


async def call_openai(body: dict, provider: Provider, param_index: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        **provider.headers,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{provider.base_url}/chat/completions", json=body, headers=headers
        )
        log_provider(provider.name, provider.type, provider.model, resp.status_code)
        if resp.status_code == 429:
            err = resp.text[:500]
            provider.rate_limit(provider.retry_after(dict(resp.headers), err))
            raise ProviderRetry(f"OpenAI 429: {err[:200]}")
        if resp.status_code != 200:
            raise Exception(f"{resp.status_code}: {resp.text[:300]}")
        return openai_response_to_anthropic(resp.json(), provider.model)


async def call_gemini(
    gemini_body: dict, provider: Provider, anthr_model: str, param_index: dict
) -> dict:
    url = f"{provider.base_url}/models/{provider.model}:generateContent"
    headers = {
        "x-goog-api-key": provider.api_key,
        "Content-Type": "application/json",
        **provider.headers,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=gemini_body, headers=headers)
        log_provider(provider.name, provider.type, provider.model, resp.status_code)
        if resp.status_code == 429:
            err = resp.text[:500]
            provider.rate_limit(provider.retry_after(dict(resp.headers), err))
            raise ProviderRetry(f"Gemini 429: {err[:200]}")
        if resp.status_code != 200:
            raise Exception(f"{resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        content: list = []
        stop = "end_turn"
        has_tools = False
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part and not part.get("thought"):
                    content.append({"type": "text", "text": part["text"]})
                elif part.get("thought") and "text" in part:
                    content.append({"type": "thinking", "thinking": part["text"]})
                elif "functionCall" in part:
                    has_tools = True
                    fc = part["functionCall"]
                    sig = part.get("thoughtSignature", "")
                    tid = _new_tool_id(sig)
                    args = fc.get("args", {})
                    if param_index:
                        args = _fix_tool_args(fc.get("name", ""), args, param_index)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": fc.get("name", ""),
                            "input": args,
                        }
                    )
            if cand.get("finishReason"):
                stop = _gemini_finish_to_stop(cand["finishReason"])

        if has_tools:
            stop = "tool_use"
        if not content:
            content.append({"type": "text", "text": ""})
        um = data.get("usageMetadata", {})
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": anthr_model,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {
                "input_tokens": um.get("promptTokenCount", 0),
                "output_tokens": um.get("candidatesTokenCount", 0),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI()


@app.post("/v1/messages")
async def messages(request: Request):
    start = time.time()
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    anthr_model = body.get("model", "unknown")
    stream = body.get("stream", False)

    # ── Token optimization ────────────────────────────────────────────────────
    orig_sys = body.get("system", "")
    system = trim_system(orig_sys)
    orig_msgs = body.get("messages", [])
    messages = trim_messages(orig_msgs)
    orig_tools = body.get("tools", []) or []
    tools = filter_tools(orig_tools)
    messages = scrub_disallowed_tool_history(messages)
    # Preseed tool_id_to_name from all original messages (before trim)
    # so tool_results referencing trimmed-away tool_use blocks still resolve
    tool_id_preseeding: dict[str, str] = {}
    for msg in orig_msgs:
        if isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_id_preseeding[b.get("id", "")] = b.get("name", "")
    param_idx = _build_param_index(tools)
    est_tok = (
        est_tokens(system)
        + est_tokens(json.dumps(messages))
        + est_tokens(json.dumps(tools))
    )

    # ── Try provider(s) — connect before emitting SSE, so 429 returns proper error ──
    last_error = None
    while True:
        provider = await PM.next()
        if not provider:
            return JSONResponse(
                {"error": f"all providers failed: {last_error or 'all rate-limited'}"},
                status_code=503,
            )

        log_req(
            system,
            len(str(orig_sys)),
            len(orig_msgs),
            len(messages),
            len(orig_tools),
            len(tools),
            est_tok,
            stream,
            provider.name,
        )

        try:
            # ── Anthropic passthrough ──────────────────────────────────────────
            if provider.type == "anthropic":
                trimmed_body = {
                    **body,
                    "system": system,
                    "messages": messages,
                    "tools": tools if tools else [],
                    "model": provider.model,
                }
                trimmed_bytes = json.dumps(trimmed_body).encode()
                if stream:
                    resp, client = await _connect_anthropic(trimmed_bytes, provider)
                    return StreamingResponse(
                        stream_anthropic_passthrough(resp, client, provider, start),
                        media_type="text/event-stream",
                    )
                headers = {
                    "x-api-key": provider.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                    **provider.headers,
                }
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        f"{provider.base_url}/v1/messages",
                        content=trimmed_bytes,
                        headers=headers,
                    )
                    log_provider(
                        provider.name, provider.type, provider.model, resp.status_code
                    )
                    if resp.status_code == 429:
                        err = resp.text[:500]
                        provider.rate_limit(
                            provider.retry_after(dict(resp.headers), err)
                        )
                        raise ProviderRetry(f"Anthropic 429: {err[:200]}")
                    return JSONResponse(resp.json(), status_code=resp.status_code)

            # ── Gemini native ─────────────────────────────────────────────────
            if provider.type == "gemini":
                gemini_body = to_gemini_request(body, system, messages, tools, preseeding=tool_id_preseeding)
                if stream:
                    resp, client = await _connect_gemini(gemini_body, provider)
                    return StreamingResponse(
                        stream_gemini(
                            resp, client, provider, anthr_model, param_idx, start
                        ),
                        media_type="text/event-stream",
                    )
                result = await call_gemini(
                    gemini_body, provider, anthr_model, param_idx
                )
                log_resp(
                    result["stop_reason"],
                    [c["name"] for c in result["content"] if c["type"] == "tool_use"],
                    result["usage"]["input_tokens"],
                    result["usage"]["output_tokens"],
                    time.time() - start,
                )
                return JSONResponse(result)

            # ── OpenAI compat ─────────────────────────────────────────────────
            inject_rc = provider.needs_reasoning_inject()
            oai_msgs = to_openai_messages(messages, system, inject_reasoning=inject_rc)
            if provider.strip_null_fields():
                oai_msgs = [
                    strip_null_values(m) if m.get("role") == "assistant" else m
                    for m in oai_msgs
                ]
            oai_body = {
                "model": provider.model,
                "messages": oai_msgs,
                "max_tokens": body.get("max_tokens", 4096),
            }
            oai_tools = to_openai_tools(tools)
            if oai_tools:
                oai_body["tools"] = oai_tools
            if body.get("temperature") is not None:
                oai_body["temperature"] = body["temperature"]

            if stream:
                resp, client = await _connect_openai(oai_body, provider)
                return StreamingResponse(
                    stream_openai(resp, client, provider, start, param_idx),
                    media_type="text/event-stream",
                )
            result = await call_openai(oai_body, provider, param_idx)
            log_resp(
                result["stop_reason"],
                [c["name"] for c in result["content"] if c["type"] == "tool_use"],
                result["usage"]["input_tokens"],
                result["usage"]["output_tokens"],
                time.time() - start,
            )
            return JSONResponse(result)

        except ProviderRetry as e:
            last_error = str(e)
            print(f"  {Y}↻ {provider.name} 429, trying next...{RESET}")
            continue
        except Exception as e:
            log_err(f"{provider.name}: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": p.model, "object": "model", "provider": p.name} for p in PM.providers
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": {
            p.name: {"available": p.available(), "type": p.type, "model": p.model}
            for p in PM.providers
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
{BOLD}{C}  ╔═══════════════════════════════════════════╗
  ║     Anthropic Proxy v4.5  (SMLabs AI)    ║
  ║     github.com/smjahid012/anthropic-proxy ║
  ╚═══════════════════════════════════════════╝{RESET}

  {W}Port{RESET}      → {G}{PORT}{RESET}
  {W}Providers{RESET} → {PM.status()}
  {W}Tools{RESET}     → {Y}{sorted(ALLOWED_TOOLS)}{RESET}
  {W}Sys cap{RESET}   → {Y}{MAX_SYS_CHARS} chars (~{MAX_SYS_CHARS // 4} tok){RESET}
  {W}Msg cap{RESET}   → {Y}last {MAX_MESSAGES} messages{RESET}
  {W}Mini-Claude{RESET}→ {(G + "ON" if MINI_CLAUDE_MODE else R + "OFF")}{RESET}  {DIM}(set mini_claude_mode in config.json){RESET}

  {DIM}Health:  http://localhost:{PORT}/health
  Models:  http://localhost:{PORT}/v1/models{RESET}
""")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
