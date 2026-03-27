# FineInstructions Dataset — Reference Guide

Dataset: `fineinstructions/fineinstructions_nemotron` (~1.71 TB, ~500M rows)

---

## 1. Token Count Analysis

Run the token distribution scan (reads only the `token_count` column via pyarrow, no full download):

```bash
bash apps/minimal_squad/run_scan_token_counts.sh
```

**Results from 10M-row scan** (`token_count_stats_10M.txt`):

| Stat | Value |
|------|-------|
| Mean | 259 tokens |
| Median | ~292 tokens |
| Std | 1,245 |
| Min | 0 |
| Max | 513,847 |

**Key observations:**
- Distribution is heavily right-skewed — most rows are short documents
- Rows with `token_count = 0` are instruction/empty-context rows (share `warc_record_id` with a document row). These are skipped in the scan since v2 of `download_fineinstructions.py`
- Only ~5–10% of all rows have `token_count ≥ 1,000`

**Sample size estimates for target token budgets** (assuming ~2,000 avg tokens within the 1k–8k filter range):

| Target tokens | Approx. samples needed |
|---------------|------------------------|
| 1M | ~500 |
| 100M | ~50,000 |
| 1B | ~500,000 |
| 3B | ~1,500,000 |

---

## 2. Downloading Data

**Script:** `apps/minimal_squad/run_download_fineinstructions.sh`
**Output dir:** `/scratch/ehghaghi/fineinstructions/`

### Key parameters

| Variable | Current value | Description |
|----------|--------------|-------------|
| `NUM_SAMPLES` | `1500000` | Number of post-filter samples (set to 0 to use `MAX_FRACTION`) |
| `MIN_TOKENS` | `1000` | Skip samples with `token_count < 1000` |
| `MAX_TOKENS` | `8000` | Skip samples with `token_count > 8000` |
| `KEEP_EMPTY` | `True` | Keep empty-context samples; save document text to `contexts.db` |
| `MAX_FRACTION` | `0.0001` | Fallback fraction (used only when `NUM_SAMPLES=0`) |

`--no-fill-empty-text` is always passed: empty-context samples keep `context=""` in the JSONL; the actual document text is stored in `contexts.db` and loaded at training time.

### Output files

| File | Description |
|------|-------------|
| `train.jsonl` | Training samples (95%) |
| `val.jsonl` | Validation samples (5%) |
| `contexts.db` | SQLite: `contexts(warc_record_id TEXT, text TEXT)` — document text for empty-context rows |
| `meta.json` | Download metadata (dataset name, fraction, sample counts, filter settings) |

### Sample format (each JSONL line)

```json
{
  "sample_id": 0,
  "context": "Monday, May 25, 2015\n\nIf you are finding...",
  "question": "What gift ideas are good for Mother's Day?",
  "answers": {"text": ["For Mother's Day, consider the following..."]},
  "warc_record_id": "<urn:uuid:...>",
  "token_count": 1243
}
```

Empty-context samples have `"context": ""` — filled from `contexts.db` at load time.

### Current downloaded dataset (as of 2026-03-27)

| Split | Samples | Size |
|-------|---------|------|
| Train | 1,017,969 | 5.3 GB |
| Val | 74,865 | 412 MB |
| **Total** | **1,092,834** | **~5.7 GB** |

---

## 3. Training

### Dataset-aware options in `cepe.py`

FineInstructions support was added directly to `cepe.py`. Key new arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `squad` | `squad` or `fineinstructions` |
| `--dataset_dir` | `/scratch/ehghaghi/fineinstructions` | Path to downloaded FI directory |
| `--gradient_accumulation_steps` | `1` | Accumulate N micro-batches before optimizer step |
| `--max_new_tokens` | `64` | Tokens to generate during eval (use 512 for FI) |

### Why these settings matter for FI

- **No span extraction:** FI answers are free-form generative text — only `C/Q//A` or `Q/C/A` formats are valid; span formats (`S`) raise an error
- **Gradient accumulation:** FI contexts are 1k–8k tokens; `batch_size=8` OOMs on a 44 GB GPU. Use `--batch_size 2 --gradient_accumulation_steps 4` for effective batch size of 8
- **`max_new_tokens=512`:** Gold answers are 200–800+ tokens; 64 (SQuAD default) severely underestimates ROUGE-L recall

### Metrics

| Dataset | Primary metric | Reference metrics |
|---------|---------------|------------------|
| SQuAD | Exact Match, F1 | — |
| FineInstructions | ROUGE-L | EM, F1 (logged but not representative for generative answers) |

ROUGE-L is LCS-based F1 over normalized token sequences. Note that FI gold answers are often full document-length paragraphs, which makes ROUGE-L recall hard to maximize even with correct generations — val loss is a more reliable training signal.

### Active experiment

```bash
# fi_modernbert400m_finetune6l_cq_a
python -m apps.minimal_squad.cepe \
    --dataset fineinstructions \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_400m \      # encoder: ModernBERT-large (400M), finetuned at 0.333x LR
    --num_decoder_layers 6 \           # decoder: 6-layer from scratch
    --pretrained_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \  # effective batch = 8
    --max_new_tokens 512 \
    --wandb_run_name fi_modernbert400m_finetune6l_cq_a
```

Submit via:
```bash
cd lingua
ibash
. setup/start_env.sh
bash apps/minimal_squad/cepe_exps.sh
```

Logs: `lingua/logs/<job_id>_<run_name>.out`

### Empty context handling at load time

`load_fineinstructions()` in `cepe.py` automatically fills empty-context samples from `contexts.db` before training. Only the wids that actually appear in empty-context samples are queried (memory-efficient batch SQL lookup).

---

## 4. Scan token counts (re-run)

To get a fresh distribution (e.g. after understanding the filtered range better):

```bash
# Edit SCAN_MAX_ROWS in run_scan_token_counts.sh, then:
bash apps/minimal_squad/run_scan_token_counts.sh
```

Note: `SCAN_MAX_ROWS` counts **non-empty rows only** (rows with `token_count > 0`).
