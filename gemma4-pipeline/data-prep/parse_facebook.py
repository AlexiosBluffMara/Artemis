#!/usr/bin/env python3
"""Parse Facebook message export JSON into training format."""

import argparse
import json
import os
from pathlib import Path


def parse_facebook_export(inbox_dir: str, output_path: str):
    inbox = Path(inbox_dir)
    conversations = []
    
    # Facebook exports have nested JSON files under inbox/<thread>/message_1.json
    for thread_dir in inbox.iterdir():
        if not thread_dir.is_dir():
            continue
        for msg_file in thread_dir.glob("message_*.json"):
            with open(msg_file, "r") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if len(messages) >= 2:
                # Sort by timestamp (oldest first)
                messages.sort(key=lambda m: m.get("timestamp_ms", 0), reverse=False)
                formatted = []
                for msg in messages:
                    formatted.append({
                        "sender": msg.get("sender_name", "Unknown"),
                        "text": msg.get("content", ""),
                        "timestamp_ms": msg.get("timestamp_ms", 0)
                    })
                conversations.append({"messages": formatted})
    
    with open(output_path, "w") as out:
        for conv in conversations:
            out.write(json.dumps(conv) + "\n")
    
    print(f"Parsed {len(conversations)} conversations to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./facebook-export/inbox")
    parser.add_argument("--output", default="./facebook-chat.jsonl")
    args = parser.parse_args()
    parse_facebook_export(args.input, args.output)


if __name__ == "__main__":
    main()
