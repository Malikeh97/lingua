#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation script for encoder-decoder QA model on SQuAD 2.0 dev set.

Runs inference on all SQuAD dev examples and computes official EM/F1 metrics.

Usage:
    python -m apps.enc_dec.eval_squad \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --data_file /path/to/dev-v2.0.json \
        --output_dir /path/to/output

    # Limit number of examples for quick testing
    python -m apps.enc_dec.eval_squad \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --data_file /path/to/dev-v2.0.json \
        --max_examples 100
"""

import argparse
import collections
import json
import logging
import os
import re
import string
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import torch
from omegaconf import OmegaConf

from apps.enc_dec.enc_dec import (
    EncDecTransformer,
    EncDecTransformerArgs,
    EncoderArgs,
    DecoderArgs,
    PretrainedEncoderArgs,
    PretrainedDecoderArgs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EVAL")


# =============================================================================
# Model Loading (from infer.py)
# =============================================================================

def load_model_and_tokenizers(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
):
    """Load model from consolidated checkpoint and setup tokenizers."""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = OmegaConf.create(config_dict)
    model_config = OmegaConf.to_container(config.model)

    # Convert nested dicts to dataclass instances
    if 'encoder' in model_config and isinstance(model_config['encoder'], dict):
        model_config['encoder'] = EncoderArgs(**model_config['encoder'])
    if 'decoder' in model_config and isinstance(model_config['decoder'], dict):
        model_config['decoder'] = DecoderArgs(**model_config['decoder'])
    if 'pretrained_encoder' in model_config and isinstance(model_config['pretrained_encoder'], dict):
        model_config['pretrained_encoder'] = PretrainedEncoderArgs(**model_config['pretrained_encoder'])
    if 'pretrained_decoder' in model_config and isinstance(model_config['pretrained_decoder'], dict):
        model_config['pretrained_decoder'] = PretrainedDecoderArgs(**model_config['pretrained_decoder'])

    model_args = EncDecTransformerArgs(**model_config)

    logger.info("Building model...")
    model = EncDecTransformer(model_args)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    # Convert to bfloat16 to match training dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Setup tokenizers
    encoder_tokenizer = None
    decoder_tokenizer = None

    if hasattr(config, 'data') and config.data.get('encoder_tokenizer_name'):
        from transformers import AutoTokenizer
        encoder_tokenizer_name = config.data.encoder_tokenizer_name
        logger.info(f"Loading encoder tokenizer: {encoder_tokenizer_name}")
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_name)

    if hasattr(config, 'data') and hasattr(config.data, 'tokenizer'):
        from lingua.tokenizer import build_tokenizer
        logger.info("Loading decoder tokenizer")
        decoder_tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)

    return model, encoder_tokenizer, decoder_tokenizer, config


# =============================================================================
# Encoding Functions (from infer.py)
# =============================================================================

def encode_document(
    document: str,
    encoder_tokenizer,
    max_len: int = 1024,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize document for encoder input."""
    encoded = encoder_tokenizer(
        document,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].bool().to(device)
    return input_ids, attention_mask


def encode_question(
    question: str,
    decoder_tokenizer,
    add_bos: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """Tokenize question for decoder input."""
    tokens = decoder_tokenizer.encode(question, add_bos=add_bos, add_eos=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return input_ids


@torch.no_grad()
def generate_answer(
    model: EncDecTransformer,
    encoder_input_ids: torch.Tensor,
    encoder_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> Tuple[str, List[int]]:
    """Generate answer using greedy decoding."""
    model.eval()
    device = encoder_input_ids.device

    # Encode document once
    encoder_output = model.encoder(
        encoder_input_ids,
        padding_mask=encoder_mask,
        attn_impl="sdpa",
    )

    if eos_token_id is None and hasattr(decoder_tokenizer, 'eos_id'):
        eos_token_id = decoder_tokenizer.eos_id

    generated_ids = decoder_input_ids.clone()
    new_token_ids = []

    for _ in range(max_new_tokens):
        logits = model.decoder(
            generated_ids,
            encoder_output,
            encoder_mask=encoder_mask,
            target=None,
            attn_impl="sdpa",
        )

        next_token_logits = logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        new_token_ids.append(next_token.item())

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    generated_text = decoder_tokenizer.decode(new_token_ids)
    return generated_text, new_token_ids


# =============================================================================
# SQuAD Data Loading
# =============================================================================

def load_squad_data(data_file: str) -> Tuple[List[Dict], Dict[str, bool]]:
    """Load SQuAD 2.0 data from JSON file.

    Returns:
        Tuple of (list of examples, qid_to_has_ans mapping)
    """
    with open(data_file, 'r') as f:
        dataset_json = json.load(f)

    dataset = dataset_json['data']

    examples = []
    qid_to_has_ans = {}

    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qid = qa['id']
                question = qa['question']

                # SQuAD 2.0 has 'is_impossible' field
                is_impossible = qa.get('is_impossible', False)
                qid_to_has_ans[qid] = not is_impossible

                if is_impossible:
                    answers = []
                else:
                    answers = [a['text'] for a in qa['answers']]

                examples.append({
                    'id': qid,
                    'context': context,
                    'question': question,
                    'answers': answers,
                    'is_impossible': is_impossible,
                })

    return examples, qid_to_has_ans


# =============================================================================
# Official SQuAD Evaluation Logic
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Get normalized tokens from string."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute exact match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute F1 score."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples: List[Dict], preds: Dict[str, str]) -> Tuple[Dict, Dict]:
    """Compute raw EM and F1 scores for each example."""
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qid = example['id']
        gold_answers = [a for a in example['answers'] if normalize_answer(a)]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = ['']

        if qid not in preds:
            logger.warning(f'Missing prediction for {qid}')
            continue

        a_pred = preds[qid]

        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)

    return exact_scores, f1_scores


def make_eval_dict(exact_scores: Dict, f1_scores: Dict, qid_list: List[str] = None) -> collections.OrderedDict:
    """Create evaluation results dictionary."""
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total if total > 0 else 0),
            ('f1', 100.0 * sum(f1_scores.values()) / total if total > 0 else 0),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total if total > 0 else 0),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total if total > 0 else 0),
            ('total', total),
        ])


def merge_eval(main_eval: Dict, new_eval: Dict, prefix: str):
    """Merge evaluation results with prefix."""
    for k in new_eval:
        main_eval[f'{prefix}_{k}'] = new_eval[k]


# =============================================================================
# Main Evaluation
# =============================================================================

def run_inference_on_examples(
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    examples: List[Dict],
    max_encoder_len: int,
    max_new_tokens: int,
    device: str,
    verbose: bool = True,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
    """Run inference on all examples and return predictions with live scoring.

    Returns:
        Tuple of (predictions, exact_scores, f1_scores)
    """
    predictions = {}
    exact_scores = {}
    f1_scores = {}

    # Running totals for live stats
    total_em = 0.0
    total_f1 = 0.0
    num_processed = 0

    for i, example in enumerate(examples):
        qid = example['id']
        context = example['context']
        question = example['question']
        gold_answers = example['answers']
        is_impossible = example['is_impossible']

        # Encode document
        encoder_input_ids, encoder_mask = encode_document(
            context,
            encoder_tokenizer,
            max_len=max_encoder_len,
            device=device,
        )

        # Encode question
        decoder_input_ids = encode_question(
            question,
            decoder_tokenizer,
            add_bos=True,
            device=device,
        )

        # Generate answer
        generated_text, _ = generate_answer(
            model,
            encoder_input_ids,
            encoder_mask,
            decoder_input_ids,
            decoder_tokenizer,
            max_new_tokens=max_new_tokens,
        )

        # Clean up generated text (remove EOS tokens, etc.)
        generated_text = generated_text.replace('<|end_of_text|>', '').strip()
        generated_text = generated_text.replace('<|eot_id|>', '').strip()

        predictions[qid] = generated_text

        # Compute scores for this example
        normalized_gold = [a for a in gold_answers if normalize_answer(a)]
        if not normalized_gold:
            normalized_gold = ['']

        em = max(compute_exact(a, generated_text) for a in normalized_gold)
        f1 = max(compute_f1(a, generated_text) for a in normalized_gold)

        exact_scores[qid] = em
        f1_scores[qid] = f1

        total_em += em
        total_f1 += f1
        num_processed += 1

        # Calculate running averages
        running_em = 100.0 * total_em / num_processed
        running_f1 = 100.0 * total_f1 / num_processed

        # Print live results
        if verbose:
            status = "CORRECT" if em == 1 else "WRONG"
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(examples)}] ID: {qid} | {status} | EM: {em} | F1: {f1:.3f}")
            print(f"{'='*70}")
            print(f"Context: {context[:200]}..." if len(context) > 200 else f"Context: {context}")
            print(f"Question: {question}")
            print(f"Gold Answers: {gold_answers if gold_answers else '[UNANSWERABLE]'}")
            print(f"Generated: {generated_text}")
            print(f"--- Running Avg: EM={running_em:.2f}% | F1={running_f1:.2f}% ---")
            sys.stdout.flush()  # Ensure output is flushed immediately

    return predictions, exact_scores, f1_scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate encoder-decoder QA model on SQuAD 2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to consolidated checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to params.json config file"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to SQuAD dev-v2.0.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (predictions.json and eval.json)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples (for quick testing)"
    )
    parser.add_argument(
        "--eval_percent",
        type=float,
        default=100.0,
        help="Percentage of dev data to use for evaluation (1-100)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per answer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Load model
    model, encoder_tokenizer, decoder_tokenizer, config = load_model_and_tokenizers(
        args.checkpoint,
        args.config,
        args.device,
    )

    max_encoder_len = config.data.get('max_encoder_len', 1024)

    # Load SQuAD data
    logger.info(f"Loading SQuAD data from {args.data_file}")
    examples, qid_to_has_ans = load_squad_data(args.data_file)
    logger.info(f"Loaded {len(examples)} examples")

    # Limit examples if specified
    total_examples = len(examples)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
        logger.info(f"Limited to {len(examples)} examples (--max_examples)")
    elif args.eval_percent < 100.0:
        num_examples = int(total_examples * args.eval_percent / 100.0)
        num_examples = max(1, num_examples)  # At least 1 example
        examples = examples[:num_examples]
        logger.info(f"Using {args.eval_percent}% of data: {len(examples)}/{total_examples} examples")

    # Run inference with live scoring
    logger.info("Running inference...")
    print("\n" + "=" * 70)
    print("STARTING EVALUATION - Live Results")
    print("=" * 70)

    predictions, exact_raw, f1_raw = run_inference_on_examples(
        model,
        encoder_tokenizer,
        decoder_tokenizer,
        examples,
        max_encoder_len,
        args.max_new_tokens,
        args.device,
        verbose=True,  # Always show live results
    )

    # Overall evaluation
    out_eval = make_eval_dict(exact_raw, f1_raw)

    # Split by HasAns / NoAns
    has_ans_qids = [e['id'] for e in examples if not e['is_impossible']]
    no_ans_qids = [e['id'] for e in examples if e['is_impossible']]

    # Filter to only those we have predictions for
    has_ans_qids = [q for q in has_ans_qids if q in exact_raw]
    no_ans_qids = [q for q in no_ans_qids if q in exact_raw]

    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')

    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(out_eval, indent=2))
    print("=" * 60)

    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save predictions
        pred_file = os.path.join(args.output_dir, "predictions.json")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {pred_file}")

        # Save evaluation results
        eval_file = os.path.join(args.output_dir, "eval.json")
        with open(eval_file, 'w') as f:
            json.dump(out_eval, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_file}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
