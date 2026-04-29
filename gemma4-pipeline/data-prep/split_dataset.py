#!/usr/bin/env python3
"""Split dataset into train/validation sets."""
import argparse
import json
import random


def split(input_path: str, train_path: str, val_path: str, ratio: float):
    data = []
    with open(input_path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * ratio)
    train = data[:split_idx]
    val = data[split_idx:]

    with open(train_path, 'w') as fout:
        for item in train:
            fout.write(json.dumps(item) + "\n")

    with open(val_path, 'w') as fout:
        for item in val:
            fout.write(json.dumps(item) + "\n")

    print(f"Train: {len(train)} conversations")
    print(f"Validation: {len(val)} conversations")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./facebook-chat-clean.jsonl")
    p.add_argument("--train-ratio", type=float, default=0.95)
    args = p.parse_args()
    split(args.input, "train.jsonl", "validation.jsonl", args.train_ratio)


if __name__ == "__main__":
    main()
