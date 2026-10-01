#!/usr/bin/env python3
"""Validate SFT JSONL, split train/valid, write stats."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/tmp/sft-t-out/sft_all.jsonl")
DATA = ROOT / "data"


def valid(obj: dict) -> dict | None:
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    clean = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant"):
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        clean.append({"role": role, "content": content.strip()})
    roles = [m["role"] for m in clean]
    if "user" not in roles or roles[-1] != "assistant":
        return None
    if len(clean[-1]["content"]) < 20:
        return None
    return {"messages": clean}


def chars(ex: dict) -> int:
    return sum(len(m["content"]) for m in ex["messages"])


def bucket(n: int) -> str:
    for lim, name in [
        (2000, "lt_2k"),
        (8000, "2k_8k"),
        (24000, "8k_24k"),
        (48000, "24k_48k"),
        (10**12, "gte_48k"),
    ]:
        if n < lim:
            return name
    return "gte_48k"


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        return 1
    examples, skipped, seen = [], 0, set()
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        ex = valid(obj)
        if not ex:
            skipped += 1
            continue
        key = hashlib.sha1(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        examples.append(ex)

    train, valid_set = [], []
    for ex in examples:
        h = int(hashlib.sha1(json.dumps(ex, sort_keys=True).encode()).hexdigest(), 16)
        (valid_set if h % 20 == 0 else train).append(ex)

    DATA.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(DATA / "all.jsonl", examples)
    write(DATA / "train.jsonl", train)
    write(DATA / "valid.jsonl", valid_set)

    lengths = [chars(e) for e in examples]
    stats = {
        "examples": len(examples),
        "train": len(train),
        "valid": len(valid_set),
        "skipped": skipped,
        "chars_total": sum(lengths),
        "chars_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "chars_max": max(lengths) if lengths else 0,
        "length_buckets": dict(Counter(bucket(n) for n in lengths)),
        "ends_with_assistant": True,
        "schema": "openai-chat-messages",
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
