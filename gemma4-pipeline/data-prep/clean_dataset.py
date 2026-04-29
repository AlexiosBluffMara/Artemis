#!/usr/bin/env python3
"""Clean and filter Facebook message dataset for training."""
import argparse
import json
import re


def is_valid_message(text: str) -> bool:
    if not text or len(text.strip()) < 5:
        return False
    # Skip pure emoji/link messages
    if re.match(r"^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$", text):
        return False
    return True


def clean(input_path: str, output_path: str):
    cleaned = 0
    dropped = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            conv = json.loads(line)
            msgs = conv.get("messages", [])
            prev_text = ""
            filtered = []
            for msg in msgs:
                text = msg.get("text", "").strip()
                if not is_valid_message(text) or text == prev_text:
                    dropped += 1
                    continue
                filtered.append(msg)
                prev_text = text
            if len(filtered) >= 2:
                cleaned += 1
                fout.write(json.dumps({"messages": filtered}) + "\n")

    print(f"Cleaned {cleaned} conversations. Dropped {dropped} messages.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./facebook-chat.jsonl")
    p.add_argument("--output", default="./facebook-chat-clean.jsonl")
    args = p.parse_args()
    clean(args.input, args.output)


if __name__ == "__main__":
    main()
