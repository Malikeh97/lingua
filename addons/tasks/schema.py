"""
Data schema for context-based examples.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import torch


@dataclass
class ContextBasedExample:
    """
    Single example with document pool.

    Documents are stored by hash for deduplication when batching.
    Tokenized fields are populated by the reader via .tokenize().
    """

    # Document pool: sha256 hash -> document text
    documents: Dict[str, str]

    # This example's document references (hashes into documents dict)
    document_hashes: List[str]

    # Example content
    query: str
    target_text: str
    example_id: str
    source: str  # Dataset name (e.g., "squad", "hotpotqa")

    # Optional
    instruction: Optional[str] = None
    misc: Dict[str, Any] = field(default_factory=dict)

    # Tokenized fields (populated by .tokenize(), required by packer)
    doc_token_ids: Optional[Dict[str, torch.Tensor]] = None  # hash → [doc_len]
    dec_token_ids: Optional[torch.Tensor] = None  # [dec_len] with BOS/EOS
    tok_labels: Optional[torch.Tensor] = None  # [dec_len], -100 for prompt

    @staticmethod
    def _hash_doc(doc: str) -> str:
        """Compute SHA256 hash of document text."""
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()

    @classmethod
    def from_raw(
        cls,
        documents: List[str],
        query: str,
        target_text: str,
        example_id: str,
        source: str,
        instruction: Optional[str] = None,
        misc: Optional[Dict[str, Any]] = None,
    ) -> "ContextBasedExample":
        """Create from raw document list (hashes computed automatically)."""
        doc_dict = {}
        doc_hashes = []
        for doc in documents:
            h = cls._hash_doc(doc)
            doc_dict[h] = doc
            doc_hashes.append(h)
        return cls(
            documents=doc_dict,
            document_hashes=doc_hashes,
            query=query,
            target_text=target_text,
            example_id=example_id,
            source=source,
            instruction=instruction,
            misc=misc or {},
        )

    def get_documents(self) -> List[str]:
        """Retrieve documents in order."""
        return [self.documents[h] for h in self.document_hashes]

    def tokenize(
        self,
        encoder_tokenizer,
        decoder_tokenizer,
        doc_max_tokens: int,
        target_max_tokens: int,
    ) -> "ContextBasedExample":
        """Tokenize documents and decoder sequence in-place.

        Encoder: each document tokenized separately, truncated to doc_max_tokens.
        Decoder: [BOS] query target [EOS], truncated to target_max_tokens.
        """
        # Tokenize each document by hash (dedup-friendly)
        self.doc_token_ids = {}
        for doc_hash, doc_text in self.documents.items():
            enc = encoder_tokenizer(
                doc_text,
                max_length=doc_max_tokens,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            self.doc_token_ids[doc_hash] = enc["input_ids"].squeeze(0)

        # Decoder: [BOS] query target [EOS]
        bos_id = decoder_tokenizer.cls_token_id or decoder_tokenizer.bos_token_id
        eos_id = decoder_tokenizer.sep_token_id or decoder_tokenizer.eos_token_id

        query_ids = decoder_tokenizer(
            self.query, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(0)
        target_ids = decoder_tokenizer(
            self.target_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(0)

        parts = []
        if bos_id is not None:
            parts.append(torch.tensor([bos_id]))
        parts.append(query_ids)
        prompt_len = sum(p.shape[0] for p in parts)
        parts.append(target_ids)
        if eos_id is not None:
            parts.append(torch.tensor([eos_id]))

        self.dec_token_ids = torch.cat(parts, dim=0)[:target_max_tokens]
        self.tok_labels = self.dec_token_ids.clone()
        self.tok_labels[:min(prompt_len, target_max_tokens)] = -100

        return self

    @property
    def enc_tokens(self) -> int:
        """Exact encoder token count. Requires .tokenize() first."""
        return sum(t.shape[0] for t in self.doc_token_ids.values())

    @property
    def dec_tokens(self) -> int:
        """Exact decoder token count. Requires .tokenize() first."""
        return self.dec_token_ids.shape[0]


@dataclass
class BatchedContextBasedExamples:
    """
    Batch with deduplicated documents.

    Each example references documents by hash, allowing sharing across examples.
    """

    # Deduplicated documents: sha256 hash -> document text
    documents: Dict[str, str]

    # Per-example data (length = batch_size)
    document_hashes: List[List[str]]
    queries: List[str]
    target_texts: List[str]
    example_ids: List[str]
    sources: List[str]
    instructions: List[Optional[str]]
    misc: List[Dict[str, Any]]

    @classmethod
    def from_raw(
        cls,
        documents_list: List[List[str]],
        queries: List[str],
        target_texts: List[str],
        example_ids: List[str],
        sources: List[str],
        instructions: Optional[List[Optional[str]]] = None,
        misc: Optional[List[Dict[str, Any]]] = None,
    ) -> "BatchedContextBasedExamples":
        """Create batch directly from raw data (hashes computed, docs deduplicated)."""
        n = len(queries)
        if instructions is None:
            instructions = [None] * n
        if misc is None:
            misc = [{} for _ in range(n)]

        doc_dict: Dict[str, str] = {}
        all_hashes: List[List[str]] = []

        for docs in documents_list:
            hashes = []
            for doc in docs:
                h = ContextBasedExample._hash_doc(doc)
                doc_dict[h] = doc
                hashes.append(h)
            all_hashes.append(hashes)

        return cls(
            documents=doc_dict,
            document_hashes=all_hashes,
            queries=queries,
            target_texts=target_texts,
            example_ids=example_ids,
            sources=sources,
            instructions=instructions,
            misc=misc,
        )

    @classmethod
    def from_examples(
        cls, examples: List[ContextBasedExample]
    ) -> "BatchedContextBasedExamples":
        """Collate examples, merging document dicts for deduplication."""
        merged_docs: Dict[str, str] = {}
        for ex in examples:
            merged_docs.update(ex.documents)

        return cls(
            documents=merged_docs,
            document_hashes=[ex.document_hashes for ex in examples],
            queries=[ex.query for ex in examples],
            target_texts=[ex.target_text for ex in examples],
            example_ids=[ex.example_id for ex in examples],
            sources=[ex.source for ex in examples],
            instructions=[ex.instruction for ex in examples],
            misc=[ex.misc for ex in examples],
        )

    def get_documents_for_example(self, example_idx: int) -> List[str]:
        """Retrieve documents for a specific example."""
        return [self.documents[h] for h in self.document_hashes[example_idx]]

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> ContextBasedExample:
        """Get a single example from the batch."""
        ex_doc_hashes = self.document_hashes[idx]
        ex_docs = {h: self.documents[h] for h in ex_doc_hashes}

        return ContextBasedExample(
            documents=ex_docs,
            document_hashes=ex_doc_hashes,
            query=self.queries[idx],
            target_text=self.target_texts[idx],
            example_id=self.example_ids[idx],
            source=self.sources[idx],
            instruction=self.instructions[idx],
            misc=self.misc[idx],
        )

    def __iter__(self) -> Iterator[ContextBasedExample]:
        for i in range(len(self)):
            yield self[i]
