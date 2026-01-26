#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Inference script for Copy Mechanism Encoder-Decoder.

The copy mechanism allows the model to either generate tokens from vocabulary
or copy tokens directly from the encoder input (context). This is useful for
extractive QA where answers come from the input document.

Usage:
    python -m apps.enc_dec.infer_copy \
        --checkpoint /path/to/consolidated.pth \
        --config /path/to/params.json

    python -m apps.enc_dec.infer_copy \
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
from omegaconf import OmegaConf

from apps.enc_dec.enc_dec_copy import (
    EncDecCopyTransformer,
    CopyMechanismArgs,
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
logger = logging.getLogger("INFER_COPY")


def load_model_and_tokenizers(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
):
    """Load copy mechanism model from consolidated checkpoint."""
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

    # Load copy mechanism args if present
    copy_config = config.get('copy', {})
    if isinstance(copy_config, dict):
        copy_args = CopyMechanismArgs(**copy_config)
    else:
        copy_args = CopyMechanismArgs()

    # Build model
    logger.info("Building Copy Mechanism model...")
    model = EncDecCopyTransformer(model_args, copy_args)

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
def generate_answer_with_copy(
    model: EncDecCopyTransformer,
    encoder_input_ids: torch.Tensor,
    encoder_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_tokenizer,
    encoder_tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> Tuple[str, List[int], List[float]]:
    """Generate answer using copy mechanism.

    Returns:
        Tuple of (generated_text, generated_token_ids, copy_gate_values)
    """
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
    copy_gate_values = []

    for _ in range(max_new_tokens):
        # Forward pass - get probabilities (not loss)
        final_probs = model.decoder(
            generated_ids,
            encoder_output,
            encoder_input_ids=encoder_input_ids,
            encoder_mask=encoder_mask,
            target=None,  # No target = return probs
            attn_impl="sdpa",
        )

        # Get probs for last position
        next_token_probs = final_probs[:, -1, :]  # [1, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            # Convert to logits, apply temp, back to probs
            next_token_logits = torch.log(next_token_probs + 1e-10) / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Sample or greedy
        if temperature == 1.0:
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(next_token_probs, num_samples=1)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        new_token_ids.append(next_token.item())

        # Track copy gate (would need to modify forward to return this)
        # For now, estimate based on whether token exists in encoder
        encoder_tokens_set = set(encoder_input_ids[0].tolist())
        is_copied = next_token.item() in encoder_tokens_set
        copy_gate_values.append(1.0 if is_copied else 0.0)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    generated_text = decoder_tokenizer.decode(new_token_ids)

    return generated_text, new_token_ids, copy_gate_values


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
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Copy Mechanism encoder-decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to consolidated checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to params.json")
    parser.add_argument("--document", type=str, default=None, help="Document/context")
    parser.add_argument("--question", type=str, default=None, help="Question to answer")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of SQuAD examples")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model, encoder_tokenizer, decoder_tokenizer, config = load_model_and_tokenizers(
        args.checkpoint, args.config, args.device
    )

    max_encoder_len = config.data.get('max_encoder_len', 1024)

    if args.document is not None and args.question is not None:
        examples = [{"context": args.document, "question": args.question, "answer": "(user provided)"}]
    else:
        examples = get_squad_examples(args.num_examples)

    print("\n" + "=" * 80)
    print("COPY MECHANISM ENCODER-DECODER INFERENCE")
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

        generated_text, generated_ids, copy_gates = generate_answer_with_copy(
            model, encoder_input_ids, encoder_mask, decoder_input_ids,
            decoder_tokenizer, encoder_tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )

        copy_ratio = sum(copy_gates) / len(copy_gates) if copy_gates else 0.0
        print(f"Generated: {generated_text}")
        print(f"Copy Ratio: {copy_ratio:.1%} (estimated)")
        print()

    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
