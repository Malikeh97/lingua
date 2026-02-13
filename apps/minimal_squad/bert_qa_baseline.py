"""
Standard BERT-style extractive QA baseline.

This is the simplest possible span extraction setup:
- Input: [CLS] Question [SEP] Context [SEP]
- Output: start/end logits over all tokens
- Loss: CrossEntropy on start + end positions

Usage:
    python -m apps.bert_qa_baseline --model_name modernbert_150m --max_train_samples 1000
"""

import argparse
import re
import string
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


MODEL_ALIASES = {
    "modernbert_150m": "answerdotai/ModernBERT-base",
    "modernbert_400m": "answerdotai/ModernBERT-large",
    "deberta_v3_300m": "microsoft/deberta-v3-large",
    "bert_base": "bert-base-uncased",
    "bert_large": "bert-large-uncased",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


class BertForQA(nn.Module):
    """Standard BERT for extractive QA."""

    def __init__(self, model_name: str):
        super().__init__()
        model_name = resolve_model_name(model_name)

        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        self.encoder = AutoModel.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        )
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Mark pretrained params
        for p in self.encoder.parameters():
            p.is_pretrained = True

    def forward(
        self, input_ids, attention_mask, start_positions=None, end_positions=None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.float()

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            # Clamp positions to valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index - 1)
            end_positions = end_positions.clamp(0, ignored_index - 1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


def prepare_features(
    examples: Dict[str, List], tokenizer, max_length: int
) -> Dict[str, List]:
    """Standard SQuAD preprocessing: Q + C with span positions."""

    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # Tokenize with question first, context second (standard BERT QA format)
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    start_positions = []
    end_positions = []

    for i, (offsets, answers) in enumerate(
        zip(tokenized["offset_mapping"], examples["answers"])
    ):
        answer_text = answers["text"][0] if answers["text"] else ""
        answer_start_char = answers["answer_start"][0] if answers["answer_start"] else 0
        answer_end_char = answer_start_char + len(answer_text)

        # Find token positions - context is sequence_id=1
        sequence_ids = tokenized.sequence_ids(i)

        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            # Only look in context (sequence_id == 1)
            if sequence_ids[idx] != 1:
                continue
            if start is None or end is None:
                continue
            if start <= answer_start_char < end:
                start_token = idx
            if start < answer_end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    # Remove offset_mapping (not needed for training)
    tokenized.pop("offset_mapping")

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


def get_collate_fn(pad_token_id: int):
    """Dynamic padding collate function."""

    def pad_seq(seqs, pad_value):
        max_len = max(len(s) for s in seqs)
        return [s + [pad_value] * (max_len - len(s)) for s in seqs]

    def collate(batch):
        return {
            "input_ids": torch.tensor(
                pad_seq([x["input_ids"] for x in batch], pad_token_id)
            ),
            "attention_mask": torch.tensor(
                pad_seq([x["attention_mask"] for x in batch], 0)
            ),
            "start_positions": torch.tensor([x["start_positions"] for x in batch]),
            "end_positions": torch.tensor([x["end_positions"] for x in batch]),
        }

    return collate


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(
        remove_articles(
            s.lower().translate(str.maketrans("", "", string.punctuation))
        ).split()
    )


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(model, dataset, tokenizer, device, max_length, max_samples=None):
    """Evaluate model on dataset."""
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare features
    features = dataset.map(
        lambda x: prepare_features(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    model.eval()
    predictions = []
    ground_truths = [sample["answers"]["text"] for sample in dataset]

    for i, feature in enumerate(tqdm(features, desc="Evaluating")):
        input_ids = torch.tensor([feature["input_ids"]]).to(device)
        attention_mask = torch.tensor([feature["attention_mask"]]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        start_idx = outputs["start_logits"].argmax(dim=-1).item()
        end_idx = outputs["end_logits"].argmax(dim=-1).item()

        if end_idx < start_idx:
            end_idx = start_idx

        # Decode prediction
        pred = tokenizer.decode(
            feature["input_ids"][start_idx : end_idx + 1], skip_special_tokens=True
        )
        predictions.append(pred)

    # Compute metrics
    exact_matches = []
    f1_scores = []

    for pred, golds in zip(predictions, ground_truths):
        golds = [golds] if isinstance(golds, str) else golds
        em = max(int(normalize_answer(pred) == normalize_answer(g)) for g in golds)
        f1 = max(compute_f1(pred, g) for g in golds)
        exact_matches.append(em)
        f1_scores.append(f1)

    return (
        {
            "exact_match": sum(exact_matches) / len(exact_matches) * 100,
            "f1": sum(f1_scores) / len(f1_scores) * 100,
        },
        predictions,
        ground_truths,
    )


def main():
    parser = argparse.ArgumentParser(description="Standard BERT QA baseline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="modernbert_150m",
        help=f"Model alias or HF name. Aliases: {list(MODEL_ALIASES.keys())}",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Freeze encoder weights"
    )
    parser.add_argument("--wandb_project", type=str, default="bert_qa_baseline")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    resolved_name = resolve_model_name(args.model_name)
    print(f"Model: {resolved_name}")
    print(f"Device: {args.device}")
    print(f"LR: {args.lr}, Batch size: {args.batch_size}")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and args.wandb_run_name is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    # Load dataset
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")

    # Prepare data
    train_dataset = dataset["train"]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    val_dataset = dataset["validation"]
    if args.max_val_samples:
        val_dataset = val_dataset.select(
            range(min(args.max_val_samples, len(val_dataset)))
        )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("Preparing features...")
    train_features = train_dataset.map(
        lambda x: prepare_features(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_features = val_dataset.map(
        lambda x: prepare_features(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    collate_fn = get_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_features, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    print("Loading model...")
    model = BertForQA(args.model_name)
    model = model.to(args.device)

    # Optimizer setup
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(model.qa_outputs.parameters(), lr=args.lr)
    else:
        print("Fine-tuning full model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train loss: {avg_train_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs["loss"].item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} - Val loss: {avg_val_loss:.4f}")

        # Evaluation
        metrics, predictions, ground_truths = evaluate(
            model,
            dataset["validation"],
            tokenizer,
            args.device,
            args.max_length,
            args.eval_samples,
        )
        print(
            f"Epoch {epoch} - EM: {metrics['exact_match']:.2f}%, F1: {metrics['f1']:.2f}%"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/exact_match": metrics["exact_match"],
                    "val/f1": metrics["f1"],
                }
            )

        # Print samples
        print("\n  Samples:")
        for i in range(min(3, len(predictions))):
            sample = dataset["validation"][i]
            gold = ground_truths[i][0] if ground_truths[i] else "N/A"
            print(f"    Q: {sample['question'][:60]}...")
            print(f"    Gold: {gold}")
            print(f"    Pred: {predictions[i]}")
            print()

    print("\nTraining complete!")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

"""
Run commands:

# Standard fine-tuning (recommended baseline) -> EM: 73.20%
python -m apps.bert_qa_baseline --model_name modernbert_150m --epochs 5 --lr 3e-5 --batch_size 16

# ModernBERT large -> EM: 88.60%
python -m apps.bert_qa_baseline --model_name modernbert_400m --epochs 5 --lr 3e-5 --batch_size 8

# Frozen encoder (train only QA head) -> EM: 16.60%
python -m apps.bert_qa_baseline --model_name modernbert_400m --freeze_encoder --lr 3e-5 --epochs 5 --batch_size 16

"""
