"""
Inference script for FineSearch encoder-decoder.

Usage:
    python -m apps.finesearch.infer config=configs/finesearch/infer.yaml
    python -m apps.finesearch.infer config=infer.yaml infer.mode=interactive

Modes:
- interactive: Chat-like interface with document context
- batch: Process input file and write predictions to output file
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from apps.finesearch.config_utils import dict_to_dataclass
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints

from addons.models.config import ModelArgs, GenerationArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.models.decoder import Decoder

logger = logging.getLogger()


# ==================== Configuration ====================


@dataclass
class InferArgs:
    """Inference-specific configuration."""

    # Checkpoint
    ckpt_dir: str = ""

    # Mode: "interactive" or "batch"
    mode: str = "interactive"

    # Batch mode settings
    input_file: Optional[str] = None  # JSONL with {"documents": [...], "query": "..."}
    output_file: Optional[str] = None

    # Device
    device: str = "cuda"


@dataclass
class InferConfig:
    """Full inference configuration."""

    name: str = "finesearch-infer"
    model: ModelArgs = field(default_factory=ModelArgs)
    generation: GenerationArgs = field(default_factory=GenerationArgs)
    infer: InferArgs = field(default_factory=InferArgs)

    # Fallback decoder tokenizer (used when model.decoder_name is empty)
    default_decoder_tokenizer: str = "meta-llama/Llama-3.2-1B"


# ==================== Model Loading ====================


def load_model_and_tokenizers(cfg: InferConfig):
    """Load model from checkpoint and build tokenizers."""
    ckpt_path = Path(cfg.infer.ckpt_dir)

    # Check for consolidated checkpoint
    if (ckpt_path / "params.json").exists():
        consolidate_path = ckpt_path
    else:
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            consolidate_path = consolidate_checkpoints(str(ckpt_path))

    # Load training config to infer model args
    params_path = consolidate_path / "params.json"
    if params_path.exists():
        import json as _json
        with open(params_path) as _f:
            train_params = _json.load(_f)
        if "model" in train_params and isinstance(train_params["model"], dict):
            cfg.model = dict_to_dataclass(ModelArgs, train_params["model"])

    # Build model
    if cfg.model.model_type == "encdec":
        model = EncoderDecoder(cfg.model)
    else:
        model = Decoder(cfg.model)

    # Load weights
    ckpt_file = consolidate_path / "consolidated.pth"
    if ckpt_file.exists():
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        elif "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

    model = model.to(cfg.infer.device).eval()

    # Build tokenizers from model args
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_name)
    decoder_name = cfg.model.decoder_name or cfg.default_decoder_tokenizer
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    return model, encoder_tokenizer, decoder_tokenizer


# ==================== Inference Engine ====================


class FineSearchEngine:
    """Inference engine for FineSearch models."""

    def __init__(
        self,
        model: torch.nn.Module,
        encoder_tokenizer,
        decoder_tokenizer,
        cfg: InferConfig,
    ):
        self.model = model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.infer.device)

        # Cached encoder output for interactive mode
        self._cached_documents = []
        self._cached_encoder_output = None

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents to context representation."""
        # Tokenize documents
        doc_tokens = []
        for doc in documents:
            tokens = self.encoder_tokenizer.encode(
                doc,
                truncation=True,
                max_length=self.cfg.model.encoder_max_len,
            )
            doc_tokens.append(torch.tensor(tokens, device=self.device))

        # Pack into single tensor (for now, simple concatenation)
        if not doc_tokens:
            return torch.tensor([], device=self.device)

        packed = torch.cat(doc_tokens)

        # Encode through encoder (if encoder-decoder model)
        if hasattr(self.model, "encode"):
            with torch.no_grad():
                encoder_output = self.model.encode(packed.unsqueeze(0))
            return encoder_output
        else:
            # Decoder-only: return tokens directly
            return packed.unsqueeze(0)

    @torch.inference_mode()
    def generate(
        self,
        documents: List[str],
        query: str,
        use_cache: bool = True,
    ) -> str:
        """
        Generate response given documents and query.

        Args:
            documents: List of document texts
            query: User query
            use_cache: If True and documents haven't changed, reuse encoder output

        Returns:
            Generated response string
        """
        gen_args = self.cfg.generation

        # Check if we can use cached encoder output
        if use_cache and documents == self._cached_documents and self._cached_encoder_output is not None:
            encoder_output = self._cached_encoder_output
        else:
            encoder_output = self.encode_documents(documents)
            self._cached_documents = documents.copy()
            self._cached_encoder_output = encoder_output

        # Tokenize query
        query_tokens = self.decoder_tokenizer.encode(
            query,
            truncation=True,
            max_length=self.cfg.model.decoder_max_len // 2,
        )
        decoder_input = torch.tensor(query_tokens, device=self.device).unsqueeze(0)

        # Generate
        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(
                encoder_output=encoder_output,
                decoder_input=decoder_input,
                max_new_tokens=gen_args.max_new_tokens,
                temperature=gen_args.temperature,
                top_p=gen_args.top_p,
                do_sample=gen_args.do_sample,
            )
            # Decode output (skip prompt tokens)
            response = self.decoder_tokenizer.decode(
                output_ids[0, len(query_tokens):],
                skip_special_tokens=True,
            )
        else:
            # Fallback: autoregressive generation with forward pass
            response = self._simple_generate(
                encoder_output,
                decoder_input,
                gen_args.max_new_tokens,
                gen_args.temperature,
            )

        return response

    def _simple_generate(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Simple autoregressive generation using forward pass."""
        generated_ids = decoder_input.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            # This is a placeholder - actual implementation depends on model structure
            # logits = self.model(encoder_output, generated_ids)
            # For now, return placeholder
            break

        # Decode
        response = self.decoder_tokenizer.decode(
            generated_ids[0, decoder_input.size(1):],
            skip_special_tokens=True,
        )
        return response if response else "[generation not implemented]"

    def clear_cache(self):
        """Clear cached encoder output."""
        self._cached_documents = []
        self._cached_encoder_output = None


# ==================== Interactive Mode ====================


def run_interactive(engine: FineSearchEngine):
    """Run interactive inference loop."""
    print("\n" + "=" * 60)
    print("FineSearch Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  /docs <text>  - Add a document to context")
    print("  /load <file>  - Load documents from file (one per line)")
    print("  /list         - List current documents")
    print("  /clear        - Clear all documents")
    print("  /quit         - Exit")
    print("\nType a query to generate a response based on loaded documents.")
    print("-" * 60)

    documents = []

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.startswith("/docs "):
            doc = user_input[6:].strip()
            if doc:
                documents.append(doc)
                engine.clear_cache()
                print(f"[Added document {len(documents)}]")
            else:
                print("[Error: No document text provided]")

        elif user_input.startswith("/load "):
            filepath = user_input[6:].strip()
            try:
                with open(filepath, "r") as f:
                    new_docs = [line.strip() for line in f if line.strip()]
                documents.extend(new_docs)
                engine.clear_cache()
                print(f"[Loaded {len(new_docs)} documents from {filepath}]")
            except Exception as e:
                print(f"[Error loading file: {e}]")

        elif user_input == "/list":
            if documents:
                print(f"\n[{len(documents)} documents loaded]")
                for i, doc in enumerate(documents):
                    preview = doc[:100] + "..." if len(doc) > 100 else doc
                    print(f"  {i+1}. {preview}")
            else:
                print("[No documents loaded]")

        elif user_input == "/clear":
            documents = []
            engine.clear_cache()
            print("[Context cleared]")

        elif user_input == "/quit":
            print("Goodbye!")
            break

        elif user_input.startswith("/"):
            print(f"[Unknown command: {user_input.split()[0]}]")

        else:
            # Query mode
            if not documents:
                print("[No documents loaded. Use /docs <text> or /load <file> first.]")
                continue

            print("\nGenerating...")
            response = engine.generate(documents, user_input)
            print(f"\n{response}")


# ==================== Batch Mode ====================


def run_batch(engine: FineSearchEngine, cfg: InferConfig):
    """Run batch inference from file."""
    input_file = cfg.infer.input_file
    output_file = cfg.infer.output_file

    if not input_file:
        print("Error: input_file not specified for batch mode")
        sys.exit(1)

    if not output_file:
        output_file = str(Path(input_file).with_suffix(".predictions.jsonl"))
        print(f"Output file not specified, using: {output_file}")

    # Read input
    print(f"Reading from {input_file}...")
    inputs = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    # Process
    print(f"Processing {len(inputs)} examples...")
    outputs = []

    for item in tqdm(inputs):
        documents = item.get("documents", [])
        query = item.get("query", "")

        if not documents or not query:
            outputs.append({"prediction": "", "error": "Missing documents or query"})
            continue

        try:
            prediction = engine.generate(documents, query, use_cache=False)
            outputs.append({
                "query": query,
                "prediction": prediction,
                "documents": documents,
            })
        except Exception as e:
            outputs.append({
                "query": query,
                "prediction": "",
                "error": str(e),
            })

    # Write output
    print(f"Writing to {output_file}...")
    with open(output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Processed {len(outputs)} examples.")


# ==================== Main ====================


def main():
    """
    Usage:
        python -m apps.finesearch.infer config=configs/finesearch/infer.yaml
        python -m apps.finesearch.infer config=infer.yaml infer.mode=batch infer.input_file=data.jsonl
    """
    from apps.finesearch.config_utils import load_config

    cfg = load_config(InferConfig)

    # Load model
    print("Loading model...")
    model, encoder_tokenizer, decoder_tokenizer = load_model_and_tokenizers(cfg)
    print("Model loaded")

    # Create engine
    engine = FineSearchEngine(model, encoder_tokenizer, decoder_tokenizer, cfg)

    # Run in specified mode
    if cfg.infer.mode == "interactive":
        run_interactive(engine)
    elif cfg.infer.mode == "batch":
        run_batch(engine, cfg)
    else:
        print(f"Unknown mode: {cfg.infer.mode}")
        print("Supported modes: interactive, batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
