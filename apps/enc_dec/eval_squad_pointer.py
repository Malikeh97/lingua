#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation script for Pointer Mechanism encoder-decoder on SQuAD 2.0 dev set.

The pointer mechanism directly points to positions in the encoder input,
guaranteeing extractive answers from the input context.

Usage:
    python -m apps.enc_dec.eval_squad_pointer \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --data_file /path/to/dev-v2.0.json \
        --output_dir /path/to/output

    # Limit number of examples for quick testing
    python -m apps.enc_dec.eval_squad_pointer \
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

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from apps.enc_dec.enc_dec_pointer import (
    EncDecPointerTransformer,
    PointerMechanismArgs,
)
from apps.enc_dec.enc_dec import (
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
logger = logging.getLogger("EVAL_POINTER")


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizers(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
):
    """Load pointer mechanism model from consolidated checkpoint."""
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

    # Load pointer mechanism args if present
    pointer_config = config.get('pointer', {})
    if isinstance(pointer_config, dict):
        pointer_args = PointerMechanismArgs(**pointer_config)
    else:
        pointer_args = PointerMechanismArgs()

    # Build model
    logger.info("Building Pointer Mechanism model...")
    model = EncDecPointerTransformer(model_args, pointer_args)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
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
# Encoding Functions
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
    add_separator: bool = True,
) -> torch.Tensor:
    """Tokenize question for decoder input."""
    if add_separator:
        question = question + "\nAnswer:"
    tokens = decoder_tokenizer.encode(question, add_bos=add_bos, add_eos=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return input_ids


@torch.no_grad()
def generate_answer_with_pointer(
    model: EncDecPointerTransformer,
    encoder_input_ids: torch.Tensor,
    encoder_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_tokenizer,
    encoder_tokenizer,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    stop_on_repeat: bool = True,
    min_tokens: int = 1,
) -> Tuple[str, List[int], float]:
    """Generate answer using pointer mechanism.

    Returns:
        Tuple of (extracted_text, pointed_positions, avg_confidence)
    """
    model.eval()
    device = encoder_input_ids.device

    # Encode document once
    encoder_output = model.encoder(
        encoder_input_ids,
        padding_mask=encoder_mask,
        attn_impl="sdpa",
    )

    generated_ids = decoder_input_ids.clone()
    pointed_positions = []
    confidence_scores = []
    prev_position = -1

    for step in range(max_new_tokens):
        # Forward pass - get pointer logits
        pointer_logits = model.decoder(
            generated_ids,
            encoder_output,
            encoder_input_ids=encoder_input_ids,
            encoder_mask=encoder_mask,
            target_positions=None,  # No target = return logits
            attn_impl="sdpa",
        )

        # Get logits for last position
        last_pointer_logits = pointer_logits[:, -1, :]  # [1, enc_seq]

        # Apply temperature
        if temperature != 1.0:
            last_pointer_logits = last_pointer_logits / temperature

        # Get probabilities
        pointer_probs = F.softmax(last_pointer_logits, dim=-1)

        # Greedy decoding
        pointed_pos = torch.argmax(pointer_probs, dim=-1).item()
        confidence = pointer_probs[0, pointed_pos].item()

        # Check for repeat (potential stop signal)
        if stop_on_repeat and pointed_pos == prev_position and step >= min_tokens:
            break

        pointed_positions.append(pointed_pos)
        confidence_scores.append(confidence)
        prev_position = pointed_pos

        # Get the token at pointed position and add to decoder input
        encoder_token_id = encoder_input_ids[0, pointed_pos].item()
        token_text = encoder_tokenizer.decode([encoder_token_id])
        decoder_tokens = decoder_tokenizer.encode(token_text, add_bos=False, add_eos=False)

        if decoder_tokens:
            next_decoder_token = torch.tensor([[decoder_tokens[0]]], device=device)
        else:
            next_decoder_token = torch.tensor([[decoder_tokenizer.eos_id]], device=device)

        generated_ids = torch.cat([generated_ids, next_decoder_token], dim=-1)

        # Low confidence = potential end
        if confidence < 0.1 and step >= min_tokens:
            break

    # Decode the extracted text using encoder tokenizer
    if pointed_positions:
        extracted_token_ids = [encoder_input_ids[0, pos].item() for pos in pointed_positions]
        extracted_text = encoder_tokenizer.decode(extracted_token_ids, skip_special_tokens=True)
    else:
        extracted_text = ""

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    return extracted_text, pointed_positions, avg_confidence


# =============================================================================
# SQuAD Data Loading
# =============================================================================

def load_squad_data(data_file: str) -> Tuple[List[Dict], Dict[str, bool]]:
    """Load SQuAD 2.0 data from JSON file."""
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

                is_impossible = qa.get('is_impossible', False)
                qid_to_has_ans[qid] = not is_impossible

                if is_impossible:
                    continue  # Skip unanswerable questions
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
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


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
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], float]:
    """Run inference on all examples and return predictions with live scoring.

    Returns:
        Tuple of (predictions, exact_scores, f1_scores, avg_confidence)
    """
    predictions = {}
    exact_scores = {}
    f1_scores = {}

    # Running totals for live stats
    total_em = 0.0
    total_f1 = 0.0
    total_confidence = 0.0
    num_processed = 0

    for i, example in enumerate(examples):
        qid = example['id']
        context = example['context']
        question = example['question']
        gold_answers = example['answers']

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

        # Generate answer with pointer mechanism
        extracted_text, pointed_positions, avg_conf = generate_answer_with_pointer(
            model,
            encoder_input_ids,
            encoder_mask,
            decoder_input_ids,
            decoder_tokenizer,
            encoder_tokenizer,
            max_new_tokens=max_new_tokens,
        )

        # Clean up generated text
        extracted_text = extracted_text.strip()

        predictions[qid] = extracted_text

        # Compute scores for this example
        normalized_gold = [a for a in gold_answers if normalize_answer(a)]
        if not normalized_gold:
            normalized_gold = ['']

        em = max(compute_exact(a, extracted_text) for a in normalized_gold)
        f1 = max(compute_f1(a, extracted_text) for a in normalized_gold)

        exact_scores[qid] = em
        f1_scores[qid] = f1

        total_em += em
        total_f1 += f1
        total_confidence += avg_conf
        num_processed += 1

        # Calculate running averages
        running_em = 100.0 * total_em / num_processed
        running_f1 = 100.0 * total_f1 / num_processed
        running_conf = 100.0 * total_confidence / num_processed

        # Check if answer is extractive
        is_extractive = extracted_text.lower() in context.lower()

        # Print live results
        if verbose:
            status = "CORRECT" if em == 1 else "WRONG"
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(examples)}] ID: {qid} | {status} | EM: {em} | F1: {f1:.3f} | Conf: {avg_conf:.1%}")
            print(f"{'='*70}")
            print(f"Context: {context[:200]}..." if len(context) > 200 else f"Context: {context}")
            print(f"Question: {question}")
            print(f"Gold Answers: {gold_answers if gold_answers else '[UNANSWERABLE]'}")
            print(f"Extracted: {extracted_text}")
            print(f"Positions: {pointed_positions[:10]}{'...' if len(pointed_positions) > 10 else ''}")
            print(f"Is Extractive: {'Yes' if is_extractive else 'No (tokenization mismatch)'}")
            print(f"--- Running Avg: EM={running_em:.2f}% | F1={running_f1:.2f}% | Conf={running_conf:.1f}% ---")
            sys.stdout.flush()

    avg_confidence = total_confidence / num_processed if num_processed > 0 else 0.0
    return predictions, exact_scores, f1_scores, avg_confidence


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pointer Mechanism encoder-decoder on SQuAD 2.0",
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
        default=32,
        help="Maximum tokens to extract per answer"
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
        num_examples = max(1, num_examples)
        examples = examples[:num_examples]
        logger.info(f"Using {args.eval_percent}% of data: {len(examples)}/{total_examples} examples")

    # Run inference with live scoring
    logger.info("Running inference...")
    print("\n" + "=" * 70)
    print("STARTING POINTER MECHANISM EVALUATION - Live Results")
    print("=" * 70)

    predictions, exact_raw, f1_raw, avg_confidence = run_inference_on_examples(
        model,
        encoder_tokenizer,
        decoder_tokenizer,
        examples,
        max_encoder_len,
        args.max_new_tokens,
        args.device,
        verbose=True,
    )

    # Overall evaluation
    out_eval = make_eval_dict(exact_raw, f1_raw)
    out_eval['avg_confidence'] = avg_confidence * 100.0  # As percentage

    # Split by HasAns / NoAns
    has_ans_qids = [e['id'] for e in examples if not e['is_impossible']]
    no_ans_qids = [e['id'] for e in examples if e['is_impossible']]

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
    print("POINTER MECHANISM EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(out_eval, indent=2))
    print("=" * 60)

    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        pred_file = os.path.join(args.output_dir, "predictions.json")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {pred_file}")

        eval_file = os.path.join(args.output_dir, "eval.json")
        with open(eval_file, 'w') as f:
            json.dump(out_eval, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_file}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
