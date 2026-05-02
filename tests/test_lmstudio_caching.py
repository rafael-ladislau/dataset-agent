#!/usr/bin/env python3
"""
LM Studio prompt-cache / KV-cache test via Anthropic SDK.

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
MODEL = "gemma-4-31b-it"
API_KEY = "lm-studio"

client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)

LONG_SYSTEM = (
    "You are a helpful assistant with deep expertise in data science, "
    "machine learning, statistics, and software engineering. "
    "You always provide concise, accurate answers. "
    "When asked about numbers, answer with just the number. "
) * 10  # repeat to make it a sizeable shared prefix (~200 tokens)


def send(messages, system=None, max_tokens=32):
    kwargs = dict(model=MODEL, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    t0 = time.perf_counter()
    resp = client.messages.create(**kwargs)
    elapsed = time.perf_counter() - t0
    return resp, elapsed


def print_result(label, resp, elapsed):
    text = resp.content[0].text if resp.content else ""
    usage = resp.usage
    print(
        f"  {label:<30} | {elapsed*1000:7.1f} ms "
        f"| in={usage.input_tokens:3d} out={usage.output_tokens:2d} "
        f"cached={usage.cache_read_input_tokens} "
        f"| '{text[:60]}'"
    )


# ---------------------------------------------------------------------------
print("=" * 80)
print("Test 1: Same prompt sent twice (cold vs warm)")
print("=" * 80)
msg = [{"role": "user", "content": "What is the capital of France?"}]
resp1, t1 = send(msg)
resp2, t2 = send(msg)
print_result("Request 1 (cold)", resp1, t1)
print_result("Request 2 (warm?)", resp2, t2)
speedup = t1 / t2 if t2 > 0 else 0
print(f"  Speedup: {speedup:.2f}x  ({'CACHE HIT' if speedup > 1.3 else 'no meaningful cache hit'})")

# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Test 2: Shared long system prompt (3 different questions, same system)")
print("=" * 80)
questions = [
    "What is 10 * 10?",
    "What is 10 * 10?",   # exact repeat
    "What is 20 * 20?",   # different question, same system prefix
]
for i, q in enumerate(questions):
    resp, t = send([{"role": "user", "content": q}], system=LONG_SYSTEM)
    print_result(f"Q{i+1}: '{q}'", resp, t)

# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Test 3: Growing conversation (prefix reuse)")
print("=" * 80)
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
    print_result(f"Turn {i+1} ({len(history)*2-1} msgs)", resp, t)
    history.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Test 4: Back-to-back identical requests (rapid fire)")
print("=" * 80)
msg = [{"role": "user", "content": "Say the word 'cache'."}]
times = []
for i in range(4):
    resp, t = send(msg)
    times.append(t)
    print_result(f"Attempt {i+1}", resp, t)

print(f"\n  Min: {min(times)*1000:.1f} ms  Max: {max(times)*1000:.1f} ms  "
      f"Trend: {'decreasing (cache warming)' if times[-1] < times[0] else 'flat/no caching'}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("cache_read_input_tokens > 0  => Anthropic-style prompt cache hit")
print("Significantly lower latency on repeat => KV cache prefix reuse")
print("Both 0/flat                  => No caching active for this model/config")
