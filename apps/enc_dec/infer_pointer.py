#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Inference script for Pointer Mechanism Encoder-Decoder.

The pointer mechanism directly points to positions in the encoder input,
guaranteeing that outputs are extracted from the input context. This is
ideal for extractive QA tasks like SQuAD.

Usage:
    python -m apps.enc_dec.infer_pointer \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json

    python -m apps.enc_dec.infer_pointer \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --document "Paris is the capital of France." \
        --question "What is the capital of France?"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

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
logger = logging.getLogger("INFER_POINTER")


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


def encode_document(
    document: str,
    encoder_tokenizer,
    max_len: int = 1024,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize document for encoder input."""
    if encoder_tokenizer is None:
        raise ValueError("Encoder tokenizer not available")

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
    if decoder_tokenizer is None:
        raise ValueError("Decoder tokenizer not available")

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
) -> Tuple[str, List[int], List[float]]:
    """Generate answer using pointer mechanism.

    The model points to encoder positions, and we extract tokens from those positions.

    Args:
        model: EncDecPointerTransformer
        encoder_input_ids: [1, enc_seq] encoded document
        encoder_mask: [1, enc_seq] attention mask
        decoder_input_ids: [1, dec_seq] encoded question
        decoder_tokenizer: Decoder tokenizer (for potential fallback)
        encoder_tokenizer: Encoder tokenizer (for decoding pointed tokens)
        max_new_tokens: Maximum positions to point to
        temperature: Pointer temperature
        stop_on_repeat: Stop when model points to same position twice
        min_tokens: Minimum tokens before allowing stop

    Returns:
        Tuple of (extracted_text, pointed_positions, confidence_scores)
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

        # Sample or greedy
        if temperature == 1.0:
            pointed_pos = torch.argmax(pointer_probs, dim=-1).item()
        else:
            pointed_pos = torch.multinomial(pointer_probs, num_samples=1).item()

        confidence = pointer_probs[0, pointed_pos].item()

        # Check for repeat (potential stop signal)
        if stop_on_repeat and pointed_pos == prev_position and step >= min_tokens:
            logger.info(f"Stopping at step {step}: repeated position {pointed_pos}")
            break

        pointed_positions.append(pointed_pos)
        confidence_scores.append(confidence)
        prev_position = pointed_pos

        # Get the token at pointed position and add to decoder input
        pointed_token = encoder_input_ids[0, pointed_pos].unsqueeze(0).unsqueeze(0)

        # For autoregressive generation, we need to map encoder token to decoder vocab
        # Since tokenizers may differ, we'll use the encoder token directly
        # This works if we're doing pure extraction
        encoder_token_id = encoder_input_ids[0, pointed_pos].item()

        # Try to find equivalent in decoder vocab (approximate)
        # For simplicity, use encoder token text -> decoder token
        token_text = encoder_tokenizer.decode([encoder_token_id])
        decoder_tokens = decoder_tokenizer.encode(token_text, add_bos=False, add_eos=False)

        if decoder_tokens:
            next_decoder_token = torch.tensor([[decoder_tokens[0]]], device=device)
        else:
            # Fallback: use a placeholder
            next_decoder_token = torch.tensor([[decoder_tokenizer.eos_id]], device=device)

        generated_ids = torch.cat([generated_ids, next_decoder_token], dim=-1)

        # Low confidence = potential end
        if confidence < 0.1 and step >= min_tokens:
            logger.info(f"Stopping at step {step}: low confidence {confidence:.3f}")
            break

    # Decode the extracted text using encoder tokenizer
    if pointed_positions:
        extracted_token_ids = [encoder_input_ids[0, pos].item() for pos in pointed_positions]
        extracted_text = encoder_tokenizer.decode(extracted_token_ids, skip_special_tokens=True)
    else:
        extracted_text = ""

    return extracted_text, pointed_positions, confidence_scores


def visualize_pointer_attention(
    document: str,
    encoder_tokenizer,
    pointed_positions: List[int],
    confidence_scores: List[float],
    max_context: int = 100,
):
    """Visualize which parts of the document were pointed to."""
    tokens = encoder_tokenizer.tokenize(document)[:max_context]

    print("\nPointer Attention Visualization:")
    print("-" * 40)

    # Create a simple text-based visualization
    position_set = set(pointed_positions)
    for i, token in enumerate(tokens):
        if i in position_set:
            idx = pointed_positions.index(i)
            conf = confidence_scores[idx] if idx < len(confidence_scores) else 0
            print(f"  [{i:3d}] **{token}** (conf: {conf:.2f})")
        else:
            print(f"  [{i:3d}] {token}")

    print("-" * 40)


def get_squad_examples(num_examples: int = 5) -> List[dict]:
    """Load examples from SQuAD dataset."""
    from datasets import load_dataset

    logger.info("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation")

    examples = []
    for i in range(min(num_examples, len(dataset))):
        item = dataset[i]
        examples.append({
            "context": item["context"],
            "question": item["question"],
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "answer_start": item["answers"]["answer_start"][0] if item["answers"]["answer_start"] else 0,
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Pointer Mechanism encoder-decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to consolidated checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to params.json")
    parser.add_argument("--document", type=str, default=None, help="Document/context")
    parser.add_argument("--question", type=str, default=None, help="Question to answer")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of SQuAD examples")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max positions to point to")
    parser.add_argument("--temperature", type=float, default=1.0, help="Pointer temperature")
    parser.add_argument("--visualize", action="store_true", help="Show pointer visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model, encoder_tokenizer, decoder_tokenizer, config = load_model_and_tokenizers(
        args.checkpoint, args.config, args.device
    )

    max_encoder_len = config.data.get('max_encoder_len', 1024)

    if args.document is not None and args.question is not None:
        examples = [{"context": args.document, "question": args.question, "answer": "(user provided)", "answer_start": 0}]
    else:
        examples = get_squad_examples(args.num_examples)

    print("\n" + "=" * 80)
    print("POINTER MECHANISM ENCODER-DECODER INFERENCE (Pure Extractive)")
    print("=" * 80)

    for i, example in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        context_display = example['context'][:200] + "..." if len(example['context']) > 200 else example['context']
        print(f"Context: {context_display}")
        print(f"Question: {example['question']}")
        print(f"Ground Truth: {example['answer']}")

        encoder_input_ids, encoder_mask = encode_document(
            example['context'], encoder_tokenizer, max_len=max_encoder_len, device=args.device
        )
        decoder_input_ids = encode_question(
            example['question'], decoder_tokenizer, add_bos=True, device=args.device
        )

        extracted_text, pointed_positions, confidences = generate_answer_with_pointer(
            model, encoder_input_ids, encoder_mask, decoder_input_ids,
            decoder_tokenizer, encoder_tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"Extracted: {extracted_text}")
        print(f"Positions: {pointed_positions[:10]}{'...' if len(pointed_positions) > 10 else ''}")
        print(f"Avg Confidence: {avg_confidence:.2%}")

        # Check if answer is in context (for extractive verification)
        is_extractive = extracted_text.strip().lower() in example['context'].lower()
        print(f"Is Extractive: {'Yes' if is_extractive else 'No (may be tokenization mismatch)'}")

        if args.visualize:
            visualize_pointer_attention(
                example['context'], encoder_tokenizer, pointed_positions, confidences
            )

        print()

    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
