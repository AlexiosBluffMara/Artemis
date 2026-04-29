# Data Preparation Pipeline for Facebook Messages

## Step 1: Export Facebook Data

1. Go to https://www.facebook.com/dyi
2. Request download of "Messages" (JSON format)
3. Wait for Facebook email (~1 hour to 1 day)
4. Download ZIP and extract to `./facebook-export/inbox/`

## Step 2: Parse JSON to Training Format

```bash
python parse_facebook.py --input ./facebook-export/inbox/ --output ./facebook-chat.jsonl
```

Output format (JSON Lines):
```json
{"messages": [
  {"sender": "Alexios Bluff Mara", "text": "Hey, how are you doing today?", "timestamp_ms": 1710000000000},
  {"sender": "Friend Name", "text": "I'm good! You?", "timestamp_ms": 1710000005000}
]}
```

## Step 3: Clean and Filter

```bash
python clean_dataset.py --input ./facebook-chat.jsonl --output ./facebook-chat-clean.jsonl
```

Filters applied:
- Remove messages shorter than 5 characters
- Remove messages with only emojis/links
- Deduplicate consecutive identical messages
- Split long conversations into 4096-token chunks

## Step 4: Train/Validation Split

```bash
python split_dataset.py --input ./facebook-chat-clean.jsonl --train-ratio 0.95
```

Produces:
- `train.jsonl`
- `validation.jsonl`

## Notes

- Facebook exports may be split across multiple JSON files
- Parse all threads, but you may want to filter to specific conversations
- Consider anonymizing names before training
