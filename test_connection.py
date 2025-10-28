#!/usr/bin/env python3
"""Quick test to verify all three LLMs are responding"""
from openai import OpenAI

participants = [("Alice", 9091), ("Bob", 9092), ("Charlie", 9093)]

print("Testing connections to all three LLMs...\n")

for name, port in participants:
    try:
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1", api_key="not-needed"
        )

        response = client.chat.completions.create(
            model="test",
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            max_tokens=50,
        )

        message = response.choices[0].message.content
        print(f"✓ {name} (port {port}): {message}")

    except Exception as e:
        print(f"✗ {name} (port {port}): Error - {str(e)}")

print("\nAll connections tested!")
