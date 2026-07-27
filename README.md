# ADHD Thinking Comparison & Reasoner

Empirically tuned **3-branch ADHD-style reasoner**.

## Proven Configuration (Run 3)

After systematic testing across 8 hard problems:

| Config | Result |
|--------|--------|
| 5 branches × 900 + synth 1600 | 4–4 |
| 5 branches × 1400 + synth 2400 | 3–5 |
| **3 branches × 1400 + synth 3500** | **7–1** |

The 3-branch version also won on **every dimension**, including coherence.

### Frames used
1. **focused** — deep, rigorous, no wandering
2. **tangential** — productive cross-domain jump, then reconnect
3. **doubt_and_weave** — attack hidden assumptions, then weave back

## Quick Start

```bash
pip install -r requirements.txt

# Single problem
python adhd_reasoner.py "Design an antifragile system prompt architecture"

# Compare against linear baseline
python adhd_reasoner.py "Your problem here" --compare

# Batch (useful for generating training data later)
python adhd_reasoner.py --batch problems/default.json
```

# 生成完 teacher 資料後
python convert_to_sft.py results/teacher_raw/adhd_batch_XXXX.json \
  --output sft_sharegpt.json \
  --format sharegpt

# 或只要最終答案、不要完整分支
python convert_to_sft.py results/teacher_raw/adhd_batch_XXXX.json \
  --output sft_alpaca.json \
  --format alpaca \
  --no-branches

Environment variables (optional):

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=your-model-name
```

## Files

- `adhd_reasoner.py` — main script (3-branch proven config)
- `compare.py` — earlier full comparison harness
- `problems/default.json` — example problems
- `results/` — output directory

## Design Notes

- Fewer, more distinct branches beat many overlapping ones.
- The synthesizer needs enough tokens to *integrate*, not just summarize.
- This setup is now ready both for daily use and for generating high-quality trajectories for future fine-tuning.

## License

MIT
