#!/usr/bin/env python3
"""Two-turn agent check against a running server on :8090: a ~4.6K-token
system prompt with five tools, then a tool result. Prints time, token
counts, cached tokens and the tool call each turn, as omp would drive it."""
from pathlib import Path
import json, time, urllib.request
readme = open(Path(__file__).resolve().parents[3] / "README.md").read() + open(Path(__file__).resolve().parents[3] / "DEVELOPMENT.md").read()
system = "You are a coding agent working in this repository. Project notes follow.\n\n" + readme[:24000]
tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)[:1]}}}
         for n, d, p in [
    ("read_file", "Read a file from the repository", {"path": {"type": "string"}}),
    ("grep", "Search files for a regular expression", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    ("edit_file", "Replace text in a file", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
    ("run", "Run a shell command", {"command": {"type": "string"}}),
    ("list_dir", "List a directory", {"path": {"type": "string"}}),
]]
def ask(messages, temperature=0.7):
    body = json.dumps({"model": "local/qwen3.8-flash-next-splash", "messages": messages, "tools": tools,
                       "max_tokens": 2500, "temperature": temperature}).encode()
    start = time.time()
    req = urllib.request.Request("http://localhost:8090/v1/chat/completions", body, {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d, time.time() - start
messages = [{"role": "system", "content": system},
            {"role": "user", "content": "Find where the memory audit checks the runtime reserve, and tell me the file."}]
d, t = ask(messages); m = d["choices"][0]["message"]
print(f"turn 1: {t:.1f}s usage={d['usage']['prompt_tokens']}/{d['usage']['completion_tokens']} finish={d['choices'][0]['finish_reason']}")
print("  tool:", json.dumps(m.get("tool_calls")[0]["function"]) if m.get("tool_calls") else None, "| text:", (m.get("content") or "")[:150])
r = m.get("reasoning_content") or ""
print("  reasoning", len(r), "chars; start:", r[:300].replace("\n"," "), "\n  ... end:", r[-400:].replace("\n"," "))
if m.get("tool_calls"):
    messages.append(m)
    messages.append({"role": "tool", "tool_call_id": m["tool_calls"][0]["id"],
                     "content": "runtime/engine/MemoryAudit.cpp:135:  if (unclassified > reserves) {\nruntime/engine/MemoryAudit.cpp:136:    std::ostringstream message;"})
    d, t = ask(messages); m = d["choices"][0]["message"]
    print(f"turn 2: {t:.1f}s usage={d['usage']['prompt_tokens']}/{d['usage']['completion_tokens']} cached={d['usage']['prompt_tokens_details']['cached_tokens']} finish={d['choices'][0]['finish_reason']}")
    print("  tool:", json.dumps(m.get("tool_calls")[0]["function"]) if m.get("tool_calls") else None, "| text:", (m.get("content") or "")[:300])
