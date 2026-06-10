# Anthropic Proxy

> **Not just a proxy — makes any model behave like Claude Code.**

Run Claude Code (and any Anthropic SDK client) against free or low-cost LLM providers — Gemini, Groq, DeepSeek, OpenAI-compatible endpoints — with full tool support, streaming, and intelligent behavior injection.

```bash
pip install fastapi uvicorn httpx
python anthropic_proxy.py
```

---

## Why This Exists

Every other proxy solves one problem: **format translation** (Anthropic Messages API → provider X API). That's table stakes.

The real problem is **behavior**. Drop a free model into Claude Code and it:

- Answers immediately without reading files
- Hallucinates file contents it never read
- Stops after a single tool call instead of working systematically
- Produces wrong code because it never verifies

**anthropic-proxy** is the only proxy that solves both layers.

---

## Features

| Feature | Description |
|---|---|
| **Mini-Claude Mode** | Injects Claude Code system prompt prefix — forces any model to read before editing, verify after writing, work systematically with tools. **Unique to this project.** |
| **Native Gemini API** | Full `thoughtSignature` round-trip for Gemini's native function calling — no OpenAI-compat translation layer needed. |
| **OpenAI Compat** | Proper tool ID conversion (`call_` → `toolu_`), param name fixing, streaming. |
| **Anthropic Passthrough** | Forward to any Anthropic-native endpoint (OpenCode Zen, KiloGateway, etc.). |
| **Multi-Provider Rotation** | Priority-based provider switching + auto-fallback on 429. |
| **Gemini Sanitizer** | Fixes turn-order errors, orphan tool calls, duplicate function responses — prevents 400s. |
| **DeepSeek Thinking** | `reasoning_content` multi-turn injection for DeepSeek reasoning mode. |
| **Groq Reasoning Fix** | Renames `delta.reasoning` → `reasoning_content` on the fly. |
| **Groq Null Field Strip** | Removes null-valued keys Groq rejects in assistant messages. |
| **Disallowed Tool Scrub** | Cleans orphan tool_use/tool_result pairs from history for filtered tools. |
| **Mixed Content Fix** | User messages with both `tool_result` and text no longer drop the text. |
| **Passthrough Token Fix** | Rebuilds trimmed body before forwarding to passthrough providers. |
| **Smart Retry-After** | Reads Gemini error body for wait time — not just headers. |
| **Safe Message Trim** | Never cuts mid-tool-sequence — prevents 400s from broken pairs. |
| **CamelCase Param Fix** | Auto-corrects Gemini-invented parameter names against the tool schema. |
| **Explicit Param Map** | Hardcoded mappings for Bash, Read, Write, Edit, Glob, Grep, and more. |
| **Schema Cleaner** | Strips non-Gemini JSON Schema fields (`$schema`, `anyOf`, `allOf`, etc.). |
| **Colored Logging** | See every request, provider, tool call, and error at a glance. |
| **Live Config** | Edit `config.json` — no restart needed. |

---

## Quick Start

### 1. Install

```bash
pip install fastapi uvicorn httpx
```

### 2. Configure

Edit `config.json`:

```json
{
  "server": { "port": 4000 },
  "mini_claude_mode": true,
  "token_limits": {
    "max_system_chars": 6000,
    "max_messages": 20,
    "max_tool_desc_chars": 150,
    "max_tool_result_chars": 8000
  },
  "allowed_tools": [
    "Read", "Edit", "Write", "Bash", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "AskFollowupQuestion"
  ],
  "providers": [
    {
      "name": "gemini",
      "type": "gemini",
      "api_key": "YOUR_KEY",
      "base_url": "https://generativelanguage.googleapis.com/v1beta",
      "model": "gemini-2.5-flash-lite",
      "priority": 1
    }
  ]
}
```

### 3. Run

```bash
python anthropic_proxy.py
```

```
  ╔═══════════════════════════════════════════╗
  ║     Anthropic Proxy v4.5  (SMLabs AI)    ║
  ╚═══════════════════════════════════════════╝

  Port      → 4000
  Providers → gemini
  Tools     → ['Bash', 'Edit', 'Glob', 'Grep', ...]
  Sys cap   → 6000 chars (~1500 tok)
  Msg cap   → last 20 messages
  Mini-Claude→ ON   (set mini_claude_mode in config.json)
```

### 4. Point Claude Code at it

```bash
claude --proxy http://localhost:4000
```

Or set the environment variable:

```bash
export CLAUDE_PROXY=http://localhost:4000
```

---

## Provider Setup

### Gemini (Native)

```json
{
  "name": "gemini",
  "type": "gemini",
  "api_key": "YOUR_GEMINI_API_KEY",
  "base_url": "https://generativelanguage.googleapis.com/v1beta",
  "model": "gemini-2.5-flash-lite",
  "priority": 1
}
```

Uses the native `streamGenerateContent` API with full `thoughtSignature` round-trip for Gemini's function calling. Unlike OpenAI-compat wrappers, this preserves Gemini's native thinking/thought blocks.

### Groq

```json
{
  "name": "groq",
  "type": "openai",
  "api_key": "YOUR_GROQ_API_KEY",
  "base_url": "https://api.groq.com/openai/v1",
  "model": "llama-3.3-70b-versatile",
  "priority": 2
}
```

Automatically strips null fields from assistant messages that Groq rejects, and renames `reasoning` → `reasoning_content` in streaming deltas.

### DeepSeek

```json
{
  "name": "deepseek",
  "type": "openai",
  "api_key": "YOUR_DEEPSEEK_KEY",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-reasoner",
  "priority": 3
}
```

Injects `reasoning_content` on every prior assistant turn — required for DeepSeek's thinking mode to function across multi-turn conversations.

### OpenAI Compat (Any Provider)

```json
{
  "name": "openai",
  "type": "openai",
  "api_key": "YOUR_KEY",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o",
  "priority": 4
}
```

Works with any OpenAI-compatible endpoint: OpenAI, Together AI, Fireworks, OpenRouter, Ollama, vLLM, etc.

### Anthropic Native (Passthrough)

```json
{
  "name": "opencode",
  "type": "anthropic",
  "api_key": "YOUR_KEY",
  "base_url": "https://api.opencode.ai",
  "model": "zen-v1",
  "priority": 5
}
```

Forwards requests as-is to any Anthropic-native endpoint. System prompt trimming, message trimming, and tool filtering still apply.

---

## SMLabs AI Serper Search

Give Claude Code real-time Google search — web, images, videos, news, shopping, places, and **deep research** with LLM-synthesized reports.

```bash
# Install
npx -y serper-search-mcp

# Or Docker
docker run -i --rm -e SERPER_API_KEY=your_key smjahid/server-serper-search:3
```

[github.com/smjahid012/serper-search-mcp-server](https://github.com/smjahid012/serper-search-mcp-server)

### 8 Tools

| Tool | Description |
|---|---|
| `search_web` | Organic results + Knowledge Graph + Answer Box + PAA |
| `search_images` | Image URLs, dimensions, source pages |
| `search_videos` | Titles, channels, durations |
| `search_news` | Headlines, sources, dates |
| `search_shopping` | Products, prices, ratings |
| `search_places` | Local businesses with address, phone, hours, GPS |
| `deep_research` | Multi-step: sub-queries → parallel search → LLM cited report |
| `search_rag_context` | Clean chunked text with metadata for vector DBs |

### Setup

1. Get a free API key at [serper.dev](https://serper.dev) (2,500 queries/month)
2. Add `search_web`, `deep_research`, etc. to your `allowed_tools` in `config.json`
3. For `deep_research`, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`

These tools connect via MCP — add the config to your MCP client:

```json
{
  "mcpServers": {
    "serper-search": {
      "command": "npx",
      "args": ["-y", "serper-search-mcp"],
      "env": { "SERPER_API_KEY": "your_key" }
    }
  }
}
```

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `server.port` | `4000` | Proxy listen port |
| `mini_claude_mode` | `true` | Inject Claude Code behavior prefix into system prompt |
| `token_limits.max_system_chars` | `6000` | Truncate system prompt to this many chars |
| `token_limits.max_messages` | `20` | Keep only last N messages (never cuts mid-tool-sequence) |
| `token_limits.max_tool_desc_chars` | `150` | Truncate tool descriptions |
| `token_limits.max_tool_result_chars` | `8000` | Truncate tool result content |
| `allowed_tools` | `[...]` | Only these tools are forwarded; others are scrubbed from history |

---

## Competitive Landscape

| Proxy | Stars | Install | Gemini Native | thoughtSig | Mini-Claude Behavior | Free-only Focus |
|---|---|---|---|---|---|---|
| **anthropic-proxy** | <20 | pip + run | ✅ | ✅ | ✅ **unique** | ✅ |
| UniClaudeProxy | ~800 | pip + run | ⚠️ partial | ❌ | ❌ | ✅ |
| Free Claude Code Proxy | ~200 | pip + run | ❌ | ❌ | ❌ | ✅ |
| LiteLLM | ~20k | Docker + DB | ❌ | ❌ | ❌ | ❌ enterprise |
| claude-code-provider-proxy | ~50 | Docker | ❌ | ❌ | ❌ | ❌ OpenRouter only |
| antigravity-claude-proxy | ~100 | npm | ❌ | ❌ | ❌ | ⚠️ ToS ban risk |

### Key Differentiators

| Capability | anthropic-proxy | Others |
|---|---|---|
| Mini-Claude behavior injection | ✅ **Only one** | ❌ Format translation only |
| Gemini native API (not OpenAI compat) | ✅ Full thoughtSignature | ⚠️ Partial or none |
| Orphan tool_use/functionResponse fix | ✅ All 4 cases (A/B/D + order) | ❌ |
| Tool param name auto-correction | ✅ Explicit map + camelCase + positional | ❌ |
| DeepSeek reasoning_content injection | ✅ Multi-turn | ❌ |
| Groq null field & reasoning fix | ✅ Automatic | ❌ |
| Safe message trim (no mid-tool cut) | ✅ | ❌ |
| Schema cleaner for Gemini | ✅ | ❌ |

---

## Why Mini-Claude Mode Matters

Free and open models lack the operational behavior that makes Claude Code effective. They:

1. **Skip exploration** — answer based on assumption, not file contents
2. **Hallucinate** — fabricate function results, file contents, error messages
3. **Stop early** — one tool call and declare done
4. **Don't verify** — edit a file and never read it back

Mini-Claude Mode prepends this behavior prompt to every system prompt:

> *Never fabricate file contents. Always Read before editing. Verify after writing. Work systematically: Explore → Plan → Execute → Verify. Use multiple tools per turn when needed.*

This turns a raw model into a Claude Code-grade agent. No other proxy does this.

---

## Architecture

```
Claude Code (or any Anthropic SDK client)
       │
       ▼  POST /v1/messages  (Anthropic Messages API format)
┌────────────────────────────────────────────────────────┐
│                  anthropic-proxy                       │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ Trim     │→ │ Scrub    │→ │ Provider Router      │ │
│  │ System   │  │ Tools    │  │ (priority + fallback) │ │
│  └──────────┘  └──────────┘  └──────┬───────────────┘ │
│                                      │                 │
│              ┌───────────────────────┼───────────┐     │
│              ▼                       ▼           ▼     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────┐ │
│  │ Gemini Native    │  │ OpenAI Compat    │  │ Anth │ │
│  │ streamGenerate   │  │ /chat/completions│  │ pass │ │
│  │ + thoughtSig     │  │ + tool_conv      │  │ thru │ │
│  └──────────────────┘  └──────────────────┘  └──────┘ │
│                                                        │
│  ┌──────────────────┐                                  │
│  │ Response → SSE   │  (Anthropic SSE stream or JSON)  │
│  └──────────────────┘                                  │
└────────────────────────────────────────────────────────┘
       │
       ▼  (Anthropic Messages API SSE or JSON response)
Claude Code continues normally
```

---

## Logging

The proxy prints colored, structured logs for every request:

```
──────────────────────────────────────────────────────────────
[05:53:39] ← REQUEST  →  gemini
  System    : 3200→4200 chars (~800→~1050 tok)
  Tools     : 3→2  ['Read', 'Edit', 'Write']
  Messages  : 24→20
  Est tokens: ~8500  stream=True
  Provider  : gemini (gemini) 200  model=gemini-3.1-flash-lite
  Streaming... done (1243 chars)
  Stop      : end_turn  tokens in=8500 out=320  elapsed=4.23s
──────────────────────────────────────────────────────────────
```

---

## Health Check

```bash
curl http://localhost:4000/health
```

```json
{
  "status": "ok",
  "providers": {
    "gemini": {
      "available": true,
      "type": "gemini",
      "model": "gemini-3.1-flash-lite"
    }
  }
}
```

---

## License

Licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

## Contributing

This project is focused on making free/cheap LLMs work well with Claude Code. Contributions welcome:

- New provider integrations
- Better param mappings for more tools
- Improvements to system prompt injection for different model families
- Test coverage
- Documentation

Open an issue or PR on [GitHub](https://github.com/smjahid012/anthropic-proxy).

---

<p align="center">
  If you find this project useful,<br>
  <strong>★ star the repo</strong> on GitHub and share it with others.<br>
  Every star helps more people discover a free alternative to paid AI coding assistants.
</p>

<p align="center">
  <a href="https://github.com/smjahid012/anthropic-proxy">github.com/smjahid012/anthropic-proxy</a><br>
  Built by <a href="https://smlabsai.com">SMLabs AI</a>
</p>
