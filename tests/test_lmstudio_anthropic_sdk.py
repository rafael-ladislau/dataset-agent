#!/usr/bin/env python3
"""
Test script: LM Studio via Anthropic Python SDK
Requires LM Studio running on http://localhost:1234 with a model loaded.
"""

import anthropic

BASE_URL = "http://localhost:1234"
MODEL = "gemma-4-31b-it"
API_KEY = "lm-studio"

client = anthropic.Anthropic(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def test_basic_message():
    print("\n--- Test 1: Basic message ---")
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"Response id : {response.id}")
    print(f"Stop reason : {response.stop_reason}")
    print(f"Usage       : {response.usage}")
    print(f"Content     : {response.content[0].text}")
    assert response.content[0].text, "Expected non-empty response"
    print("PASSED")


def test_system_prompt():
    print("\n--- Test 2: System prompt ---")
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        system="You are a pirate. Always respond in pirate speak.",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(f"Content: {response.content[0].text}")
    assert response.content[0].text, "Expected non-empty response"
    print("PASSED")


def test_multi_turn():
    print("\n--- Test 3: Multi-turn conversation ---")
    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {"role": "user", "content": "My name is Rafael."},
            {"role": "assistant", "content": "Nice to meet you, Rafael!"},
            {"role": "user", "content": "What is my name?"},
        ],
    )
    print(f"Content: {response.content[0].text}")
    assert "rafael" in response.content[0].text.lower(), "Expected name recall"
    print("PASSED")


def test_tool_use():
    print("\n--- Test 4: Tool definitions (Anthropic schema) ---")
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What is the weather in New York?"}],
    )
    print(f"Stop reason : {response.stop_reason}")
    print(f"Content     : {response.content}")
    print("PASSED")


def test_streaming():
    print("\n--- Test 5: Streaming ---")
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()
    assert full_text, "Expected non-empty streamed response"
    print("PASSED")


if __name__ == "__main__":
    print(f"Anthropic SDK version : {anthropic.__version__}")
    print(f"LM Studio base URL    : {BASE_URL}")
    print(f"Model                 : {MODEL}")

    tests = [
        test_basic_message,
        test_system_prompt,
        test_multi_turn,
        test_tool_use,
        test_streaming,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
