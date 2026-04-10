"""
Data schema for context-based examples.

Framework-agnostic - pure Python dataclasses.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ContextBasedExample:
    """
    Single example with document pool.

    Documents are stored by hash for deduplication when batching.
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

    def estimate_tokens(
        self,
        chars_per_token: float = 4.0,
        enc_cost: float = 1.0,
        dec_cost: float = 1.0,
    ) -> int:
        """
        Estimate weighted token cost for bin-packing.

        Encoder tokens (documents) weighted by enc_cost,
        decoder tokens (query + target) weighted by dec_cost.
        """
        enc_chars = sum(len(d) for d in self.documents.values())
        dec_chars = len(self.query) + len(self.target_text)
        if self.instruction:
            dec_chars += len(self.instruction)
        enc_tokens = enc_chars / chars_per_token
        dec_tokens = dec_chars / chars_per_token
        return int(enc_tokens * enc_cost + dec_tokens * dec_cost)


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

        # Build deduplicated document pool
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
        # Merge all document pools
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
        # Extract only the documents needed for this example
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

    def estimate_tokens(
        self,
        chars_per_token: float = 4.0,
        enc_cost: float = 1.0,
        dec_cost: float = 1.0,
    ) -> int:
        """
        Estimate weighted token cost for bin-packing.

        Encoder tokens (documents) weighted by enc_cost,
        decoder tokens (queries + targets) weighted by dec_cost.
        """
        enc_chars = sum(len(d) for d in self.documents.values())
        dec_chars = sum(len(q) for q in self.queries)
        dec_chars += sum(len(t) for t in self.target_texts)
        dec_chars += sum(len(i) for i in self.instructions if i)
        enc_tokens = enc_chars / chars_per_token
        dec_tokens = dec_chars / chars_per_token
        return int(enc_tokens * enc_cost + dec_tokens * dec_cost)
