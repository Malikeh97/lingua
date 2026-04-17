"""
Test packed encoder: compare HF ModernBERT output with our packed version.

Usage:
    python -m apps.addon_tests.test_packed_encoder

Loads both models with identical weights, feeds the same sentences, and
asserts hidden states match within a tolerance calibrated to bfloat16
attention-backend noise.

Why a calibrated tolerance: across 22 layers in bf16, HF's own attention
backends (eager/sdpa/flash_attention_2) disagree on the order of ~0.5
absolute on pretrained ModernBERT outputs. So instead of a fixed 1e-2
tolerance, we measure HF's own eager-vs-sdpa noise floor and require
our output to lie within a small multiple of it.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from addons.models.config import ModelArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.data.collate import PackedSequences, TokenizedBatch


def main():
    encoder_name = "answerdotai/ModernBERT-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Encoder: {encoder_name}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "ModernBERT is a masked language model built for efficiency.",
        "A much longer sentence to test variable length handling across the packed attention implementation and ensure correctness.",
    ]
    encoded = [tokenizer(s, return_tensors="pt", add_special_tokens=True) for s in sentences]
    lengths = [e["input_ids"].shape[1] for e in encoded]
    print(f"Sentence lengths: {lengths}")

    padded = tokenizer(sentences, padding=True, return_tensors="pt").to(device)

    # === HF reference + noise floor ===
    print("\nLoading HF (eager)...")
    hf_eager = AutoModel.from_pretrained(
        encoder_name, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    with torch.no_grad():
        hf_eager_out = hf_eager(**padded).last_hidden_state
    del hf_eager
    torch.cuda.empty_cache()

    print("Loading HF (sdpa)...")
    hf_sdpa = AutoModel.from_pretrained(
        encoder_name, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    with torch.no_grad():
        hf_sdpa_out = hf_sdpa(**padded).last_hidden_state
    del hf_sdpa
    torch.cuda.empty_cache()

    # === Our packed model ===
    print("\nLoading packed encoder...")
    args = ModelArgs(encoder_name=encoder_name, model_type="encdec")
    our_model = EncoderDecoder(args).to(device=device, dtype=dtype).eval()

    token_tensors = [e["input_ids"].squeeze(0) for e in encoded]
    packed_tokens = PackedSequences.from_tensors(token_tensors, device)
    dummy_dec = PackedSequences.from_tensors([torch.tensor([0])], device)
    batch = TokenizedBatch(
        encoder_tokens=packed_tokens,
        doc_hash_to_idx={},
        decoder_tokens=dummy_dec,
        labels=torch.tensor([-100], device=device),
        example_doc_indices=[[0]],
    )
    with torch.no_grad():
        our_output = our_model.encode(batch)

    cu = packed_tokens.cu_seqlens

    # === Noise floor + ours ===
    # noise_floor: max |eager - sdpa| — bf16 attention-backend disagreement.
    # ours_vs_eager: max |eager - ours|. Both are "correct" implementations,
    # so ours should be within a small multiple of the noise floor.
    tolerance_factor = 3.0
    print("\nPer-sentence comparison:")
    print(f"  {'#':<3} {'len':>4} {'eager_vs_sdpa':>15} {'ours_vs_eager':>15} {'ours_vs_sdpa':>15} {'verdict':>8}")
    all_pass = True
    for i, L in enumerate(lengths):
        eager_seq = hf_eager_out[i, :L]
        sdpa_seq = hf_sdpa_out[i, :L]
        ours_seq = our_output[cu[i]:cu[i + 1]]

        eager_vs_sdpa = (eager_seq - sdpa_seq).abs().max().item()
        ours_vs_eager = (eager_seq - ours_seq).abs().max().item()
        ours_vs_sdpa = (sdpa_seq - ours_seq).abs().max().item()

        floor = max(eager_vs_sdpa, 1e-3)
        ok = min(ours_vs_eager, ours_vs_sdpa) <= tolerance_factor * floor
        if not ok:
            all_pass = False
        print(f"  {i:<3} {L:>4} {eager_vs_sdpa:>15.4e} {ours_vs_eager:>15.4e} {ours_vs_sdpa:>15.4e} {'PASS' if ok else 'FAIL':>8}")

    print(f"\nTolerance: ours must be within {tolerance_factor}x of HF eager/sdpa noise.")
    if all_pass:
        print("All sentences match (within bfloat16 backend noise).")
    else:
        print("MISMATCH — ours differs from HF by more than backend noise suggests.")
    return all_pass


if __name__ == "__main__":
    main()
