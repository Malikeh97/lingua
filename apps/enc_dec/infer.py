#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Inference script for encoder-decoder QA model.

Loads a consolidated checkpoint and generates answers to questions
given a document context.

Usage:
    # Basic usage with SQuAD example
    python -m apps.enc_dec.infer \
        --checkpoint /path/to/checkpoint/0000001000/consolidated/consolidated.pth \
        --config /path/to/checkpoint/0000001000/params.json

    # Custom document and question
    python -m apps.enc_dec.infer \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --document "Paris is the capital of France." \
        --question "What is the capital of France?"

    # Run on multiple SQuAD examples
    python -m apps.enc_dec.infer \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json \
        --num_examples 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

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
logger = logging.getLogger("INFER")


def load_model_and_tokenizers(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
):
    """Load model from consolidated checkpoint and setup tokenizers.

    Args:
        checkpoint_path: Path to consolidated.pth or model_weights.pth
        config_path: Path to params.json config file
        device: Device to load model on

    Returns:
        Tuple of (model, encoder_tokenizer, decoder_tokenizer, config)
    """
    # Load config
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

    # Build model
    logger.info("Building model...")
    model = EncDecTransformer(model_args)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    # Convert to bfloat16 to match training dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Setup tokenizers
    encoder_tokenizer = None
    decoder_tokenizer = None

    # Encoder tokenizer (for pretrained encoder like ModernBERT)
    if hasattr(config, 'data') and config.data.get('encoder_tokenizer_name'):
        from transformers import AutoTokenizer
        encoder_tokenizer_name = config.data.encoder_tokenizer_name
        logger.info(f"Loading encoder tokenizer: {encoder_tokenizer_name}")
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_name)

    # Decoder tokenizer
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
    """Tokenize document for encoder input.

    Returns:
        Tuple of (input_ids, attention_mask) tensors
    """
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
    add_separator: bool = True,
) -> torch.Tensor:
    """Tokenize question for decoder input.

    Args:
        question: The question string
        decoder_tokenizer: Tokenizer for encoding
        add_bos: Whether to add BOS token
        device: Device to put tensor on
        add_separator: Whether to add "\\nAnswer:" separator after question

    Returns:
        input_ids tensor
    """
    if decoder_tokenizer is None:
        raise ValueError("Decoder tokenizer not available")

    # Add separator to match training format
    if add_separator:
        question = question + "\nAnswer:"

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
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> Tuple[str, List[int]]:
    """Generate answer using greedy/sampling decoding.

    Args:
        model: EncDecTransformer model
        encoder_input_ids: [1, enc_seq] encoded document
        encoder_mask: [1, enc_seq] attention mask for encoder
        decoder_input_ids: [1, dec_seq] encoded question (prompt)
        decoder_tokenizer: Tokenizer for decoding output
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = greedy)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        eos_token_id: Stop generation when this token is produced

    Returns:
        Tuple of (generated_text, generated_token_ids)
    """
    model.eval()
    device = encoder_input_ids.device

    # Encode document once
    encoder_output = model.encoder(
        encoder_input_ids,
        padding_mask=encoder_mask,
        attn_impl="sdpa",
    )

    # Get EOS token if available
    if eos_token_id is None and hasattr(decoder_tokenizer, 'eos_id'):
        eos_token_id = decoder_tokenizer.eos_id

    # Start with question tokens
    generated_ids = decoder_input_ids.clone()
    new_token_ids = []

    for _ in range(max_new_tokens):
        # Forward pass through decoder
        logits = model.decoder(
            generated_ids,
            encoder_output,
            encoder_mask=encoder_mask,
            target=None,
            attn_impl="sdpa",
        )

        # Get logits for last position
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample or take argmax
        if temperature == 1.0 and top_k is None and top_p is None:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        new_token_ids.append(next_token.item())

        # Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    # Decode generated tokens
    generated_text = decoder_tokenizer.decode(new_token_ids)

    return generated_text, new_token_ids


def get_squad_examples(num_examples: int = 5) -> List[dict]:
    """Load examples from SQuAD dataset.

    Returns:
        List of dicts with 'context', 'question', 'answer' keys
    """
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
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with encoder-decoder QA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to consolidated checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to params.json config file"
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Document/context for the question (optional, uses SQuAD if not provided)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to answer (optional, uses SQuAD if not provided)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of SQuAD examples to run (if no document/question provided)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Load model and tokenizers
    model, encoder_tokenizer, decoder_tokenizer, config = load_model_and_tokenizers(
        args.checkpoint,
        args.config,
        args.device,
    )

    # Get max lengths from config
    max_encoder_len = config.data.get('max_encoder_len', 1024)

    # Determine examples to run
    if args.document is not None and args.question is not None:
        examples = [{
            "context": args.document,
            "question": args.question,
            "answer": "(user provided)",
        }]
    else:
        examples = get_squad_examples(args.num_examples)

    # Run inference on each example
    print("\n" + "=" * 80)
    print("ENCODER-DECODER QA INFERENCE")
    print("=" * 80)

    for i, example in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        print(f"Context: {example['context'][:200]}..." if len(example['context']) > 200 else f"Context: {example['context']}")
        print(f"Question: {example['question']}")
        print(f"Ground Truth: {example['answer']}")

        # Encode document
        encoder_input_ids, encoder_mask = encode_document(
            example['context'],
            encoder_tokenizer,
            max_len=max_encoder_len,
            device=args.device,
        )

        # Encode question
        decoder_input_ids = encode_question(
            example['question'],
            decoder_tokenizer,
            add_bos=True,
            device=args.device,
        )

        # Generate answer
        generated_text, generated_ids = generate_answer(
            model,
            encoder_input_ids,
            encoder_mask,
            decoder_input_ids,
            decoder_tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print(f"Generated: {generated_text}")
        print()

    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
