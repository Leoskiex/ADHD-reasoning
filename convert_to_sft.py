#!/usr/bin/env python3
"""
把 adhd_reasoner.py 產出的 JSON 轉成 SFT 訓練格式
支援 ShareGPT / Alpaca 兩種常見格式
"""

import json
import argparse
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return data


def is_good_example(item: dict, min_final_len: int = 200) -> bool:
    """簡單品質過濾"""
    final = item.get("final") or item.get("adhd", {}).get("final", "")
    if len(final.strip()) < min_final_len:
        return False
    branches = item.get("branches") or item.get("adhd", {}).get("branches", [])
    if len(branches) < 2:
        return False
    return True


def to_sharegpt(item: dict, include_branches: bool = True) -> dict:
    """ShareGPT 格式（很多框架直接支援）"""
    problem = item.get("problem", "")
    final = item.get("final") or item.get("adhd", {}).get("final", "")
    branches = item.get("branches") or item.get("adhd", {}).get("branches", [])

    if include_branches and branches:
        thinking_parts = []
        for b in branches:
            frame = b.get("frame", "unknown")
            content = b.get("content", "")
            thinking_parts.append(f"### Branch ({frame})\n{content}")
        thinking = "\n\n".join(thinking_parts)
        assistant = f"### ADHD Thinking\n{thinking}\n\n### Final Answer\n{final}"
    else:
        assistant = final

    return {
        "conversations": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant},
        ]
    }


def to_alpaca(item: dict, include_branches: bool = True) -> dict:
    """Alpaca 格式"""
    problem = item.get("problem", "")
    final = item.get("final") or item.get("adhd", {}).get("final", "")
    branches = item.get("branches") or item.get("adhd", {}).get("branches", [])

    instruction = (
        "用 ADHD 風格思考這個問題：允許跳躍、質疑假設、跨領域借概念，"
        "但最終必須收束成連貫、可用的答案。明確展示思考中的跳躍與回收。"
    )

    if include_branches and branches:
        thinking_parts = []
        for b in branches:
            frame = b.get("frame", "unknown")
            content = b.get("content", "")
            thinking_parts.append(f"### Branch ({frame})\n{content}")
        thinking = "\n\n".join(thinking_parts)
        output = f"### ADHD Thinking\n{thinking}\n\n### Final Answer\n{final}"
    else:
        output = final

    return {
        "instruction": instruction,
        "input": problem,
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="adhd_reasoner 產出的 json（單一或 batch）")
    parser.add_argument("--output", type=Path, default=Path("sft_data.json"))
    parser.add_argument("--format", choices=["sharegpt", "alpaca"], default="sharegpt")
    parser.add_argument("--no-branches", action="store_true", help="只保留最終合成答案")
    parser.add_argument("--min-length", type=int, default=200, help="final 最短字元數")
    args = parser.parse_args()

    raw = load_results(args.input)
    print(f"讀入 {len(raw)} 筆原始資料")

    kept = []
    for item in raw:
        if is_good_example(item, min_final_len=args.min_length):
            if args.format == "sharegpt":
                kept.append(to_sharegpt(item, include_branches=not args.no_branches))
            else:
                kept.append(to_alpaca(item, include_branches=not args.no_branches))

    print(f"通過過濾：{len(kept)} 筆")
    args.output.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫入 → {args.output}")


if __name__ == "__main__":
    main()
