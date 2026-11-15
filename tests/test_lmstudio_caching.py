#!/usr/bin/env python3
"""
LM Studio prompt-cache / KV-cache test via Anthropic SDK.
Reports tokens-per-second for prefill and generation on each request.

Three scenarios:
  1. Same prompt sent twice         -> cold vs warm latency
  2. Shared long system prompt      -> should benefit if prefix caching works
  3. Incremental conversation turns -> each turn extends the same prefix

Run with:
    venv/bin/python test_lmstudio_caching.py
"""

import time
import anthropic

BASE_URL = "http://localhost:1234"
MODEL = "gemma-4-31b-it-mlx"
API_KEY = "lm-studio"

client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)

LONG_SYSTEM = (
    "You are a helpful assistant with deep expertise in data science, "
    "machine learning, statistics, and software engineering. "
    "You always provide concise, accurate answers. "
    "When asked about numbers, answer with just the number. "
) * 10  # ~200 tokens shared prefix


def send(messages, system=None, max_tokens=32):
    kwargs = dict(model=MODEL, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    t0 = time.perf_counter()
    resp = client.messages.create(**kwargs)
    elapsed = time.perf_counter() - t0
    return resp, elapsed


def tok_per_sec(tokens, elapsed_ms):
    if elapsed_ms <= 0 or tokens <= 0:
        return 0.0
    return tokens / (elapsed_ms / 1000)


def print_result(label, resp, elapsed):
    usage = resp.usage
    text = resp.content[0].text.strip() if resp.content else ""
    elapsed_ms = elapsed * 1000
    in_tok = usage.input_tokens
    out_tok = usage.output_tokens
    cached = usage.cache_read_input_tokens or 0
    fresh_in = in_tok - cached

    # Rough tok/s estimates:
    # Total time = prefill_time + generation_time
    # We approximate: generation_time = out_tok * ~avg_ms_per_out_tok
    # We use the overall elapsed since we don't have the internal split here.
    overall_tps = (in_tok + out_tok) / elapsed if elapsed > 0 else 0

    print(
        f"  {label:<32} | {elapsed_ms:7.1f} ms "
        f"| in={in_tok:3d} (cached={cached:3d} fresh={fresh_in:3d}) out={out_tok:2d} "
        f"| ~{overall_tps:5.1f} tok/s total "
        f"| '{text[:50]}'"
    )
    return elapsed_ms, out_tok, cached


# ---------------------------------------------------------------------------
print("=" * 100)
print(f"Model: {MODEL}")
print("=" * 100)

print("\n[Test 1] Same prompt sent twice (cold vs warm)")
print("-" * 100)
msg = [{"role": "user", "content": "What is the capital of France?"}]
resp1, t1 = send(msg)
resp2, t2 = send(msg)
print_result("Request 1 (cold)", resp1, t1)
print_result("Request 2 (warm?)", resp2, t2)
speedup = t1 / t2 if t2 > 0 else 0
print(f"  -> Speedup: {speedup:.2f}x  ({'CACHE HIT likely' if speedup > 1.3 else 'no meaningful cache hit'})")

# ---------------------------------------------------------------------------
print("\n[Test 2] Shared long system prompt (~200 tok prefix)")
print("-" * 100)
questions = [
    "What is 10 * 10?",
    "What is 10 * 10?",
    "What is 20 * 20?",
]
prev_t = None
for i, q in enumerate(questions):
    resp, t = send([{"role": "user", "content": q}], system=LONG_SYSTEM)
    em, ot, cached = print_result(f"Q{i+1}: '{q}'", resp, t)
    prev_t = t

# ---------------------------------------------------------------------------
print("\n[Test 3] Growing conversation (prefix reuse)")
print("-" * 100)
history = []
turns = [
    "My favourite number is 42.",
    "Double my favourite number.",
    "Add 8 to that result.",
    "What was my favourite number?",
]
for i, turn in enumerate(turns):
    history.append({"role": "user", "content": turn})
    resp, t = send(history)
    answer = resp.content[0].text.strip()
    print_result(f"Turn {i+1} ({len(history)} msgs)", resp, t)
    history.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------------
print("\n[Test 4] Back-to-back identical rapid-fire (x5)")
print("-" * 100)
msg = [{"role": "user", "content": "Say the word 'cache'."}]
times = []
for i in range(5):
    resp, t = send(msg)
    times.append(t * 1000)
    print_result(f"Attempt {i+1}", resp, t)

print(f"\n  Min latency: {min(times):.1f} ms  Max: {max(times):.1f} ms  "
      f"Trend: {'warming up (decreasing)' if times[-1] < times[0] else 'flat / no warmup'}")

print("\n" + "=" * 100)
print("NOTE: 'cached' tokens skip prefill compute → lower latency, same generation tok/s")
print("      Generation tok/s is unaffected by caching (only time-to-first-token improves)")
print("=" * 100)
