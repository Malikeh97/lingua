# Pointer Mechanism for Extractive QA

## Overview

The Pointer Mechanism is a **pure extractive** architecture where the decoder directly points to positions in the encoder input rather than generating tokens from a vocabulary. This guarantees that all outputs are exact copies from the input context, making it ideal for tasks like extractive question answering where answers must be spans from the source document.

Unlike the Copy Mechanism (which blends generation and copying), the Pointer Mechanism has **no generation capability** - every output token is selected by pointing to an encoder position.

## Mathematical Formulation

### Core Equation

The pointer network computes a probability distribution over encoder positions:

```
P(position = j | decoder_state_t) = softmax(score(h_dec_t, h_enc_j))
```

Where:
- `h_dec_t`: Decoder hidden state at position t
- `h_enc_j`: Encoder hidden state at position j
- `score()`: Scoring function (typically dot product)

### Pointer Logits Computation

The scoring is done via scaled dot-product between projected decoder and encoder hidden states:

```
# Project to pointer space (optional)
h_dec_proj = W_dec @ h_dec          # [B, dec_seq, ptr_dim]
h_enc_proj = W_enc @ h_enc          # [B, enc_seq, ptr_dim]

# Compute pointer logits
pointer_logits = h_dec_proj @ h_enc_proj.T    # [B, dec_seq, enc_seq]
pointer_logits = pointer_logits / sqrt(ptr_dim)

# Apply softmax to get probabilities
P_pointer = softmax(pointer_logits, dim=-1)   # [B, dec_seq, enc_seq]
```

### Output Extraction

To get the output text, we select encoder tokens at the pointed positions:

```
# Get most likely positions
positions = argmax(pointer_logits, dim=-1)    # [B, dec_seq]

# Extract tokens from encoder input
output_tokens = encoder_input_ids.gather(dim=1, index=positions)

# Decode using encoder tokenizer
output_text = encoder_tokenizer.decode(output_tokens)
```

## Architecture Diagram

```
                                    OUTPUT
                                      |
                                      v
                        +-------------------------+
                        |   Position Softmax       |
                        |  P(pos) over enc_seq     |
                        |    [B, dec, enc_seq]     |
                        +------------+------------+
                                     |
                        +------------+------------+
                        |     Pointer Logits       |
                        |                          |
                        |  dec_proj @ enc_proj.T   |
                        |  ---------------------   |
                        |      sqrt(ptr_dim)       |
                        +------------+------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +-------------------+             +-------------------+
        |  Decoder Pointer  |             |  Encoder Pointer  |
        |    Projection     |             |    Projection     |
        |  [B,dec,ptr_dim]  |             |  [B,enc,ptr_dim]  |
        +---------+---------+             +---------+---------+
                  |                                 |
                  |                                 |
        +---------+---------+             +---------+---------+
        |     RMSNorm       |             |   Encoder Output  |
        +---------+---------+             |   [B,enc_seq,D]   |
                  |                       +---------+---------+
                  |                                 |
    +-------------+-------------+                   |
    |                           |                   |
    |     DECODER BLOCK x N     |                   |
    |                           |                   |
    |  +---------------------+  |                   |
    |  |    Feed-Forward     |  |                   |
    |  +----------+----------+  |                   |
    |             |             |                   |
    |  +----------v----------+  |                   |
    |  |   Cross-Attention   |<-+-------------------+
    |  +----------+----------+  |
    |             |             |
    |  +----------v----------+  |
    |  |   Self-Attention    |  |
    |  |  (Causal + RoPE)    |  |
    |  +----------+----------+  |
    |             |             |
    +-------------+-------------+
                  |
    +-------------+-------------+
    |     Token Embeddings      |
    +-------------+-------------+
                  |
                  ^
            DECODER INPUT
          [question tokens]
```

## Detailed Pointer Computation

```
+--------------------------------------------------------------------+
|                     POINTER MECHANISM                               |
|                                                                     |
|   Decoder Hidden:  h_dec  [B, dec_seq, D]                          |
|   Encoder Hidden:  h_enc  [B, enc_seq, D]                          |
|                                                                     |
|   Step 1: Project (optional, controlled by use_pointer_projection)  |
|           h_dec_proj = Linear(h_dec)  ->  [B, dec_seq, ptr_dim]    |
|           h_enc_proj = Linear(h_enc)  ->  [B, enc_seq, ptr_dim]    |
|                                                                     |
|   Step 2: Dot Product                                               |
|           logits = h_dec_proj @ h_enc_proj.T                        |
|                  = [B, dec_seq, enc_seq]                            |
|                                                                     |
|   Step 3: Scale                                                     |
|           logits = logits / sqrt(ptr_dim)                           |
|                                                                     |
|   Step 4: Apply Temperature (optional)                              |
|           logits = logits / temperature                             |
|                                                                     |
|   Step 5: Mask Padding                                              |
|           logits[padding_positions] = -inf                          |
|                                                                     |
|   Step 6: Softmax                                                   |
|           P(position | decoder_state) = softmax(logits, dim=-1)     |
|                                                                     |
|   Training Loss: CrossEntropy(logits, target_positions)             |
+--------------------------------------------------------------------+
```

## Implementation Components

### 1. PointerDecoder

The main decoder class that outputs pointer logits instead of vocabulary logits:

```python
class PointerDecoder(nn.Module):
    def __init__(self, args, pointer_args):
        # Standard decoder components
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DecoderBlock(...) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)

        # Pointer-specific: projections for pointer computation
        if pointer_args.use_pointer_projection:
            self.decoder_pointer_proj = nn.Linear(dim, ptr_dim)
            self.encoder_pointer_proj = nn.Linear(dim, ptr_dim)

    def forward(self, input_ids, encoder_output, encoder_input_ids,
                encoder_mask, target_positions=None):
        # Standard decoder forward
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h, encoder_output, ...)
        h = self.norm(h)

        # Compute pointer logits
        if self.use_pointer_projection:
            h_proj = self.decoder_pointer_proj(h)
            enc_proj = self.encoder_pointer_proj(encoder_output)
        else:
            h_proj, enc_proj = h, encoder_output

        # Scaled dot product
        pointer_logits = torch.bmm(h_proj, enc_proj.transpose(1, 2))
        pointer_logits = pointer_logits / sqrt(self.ptr_dim)

        # Apply temperature
        pointer_logits = pointer_logits / self.temperature

        # Mask padding
        pointer_logits.masked_fill_(~encoder_mask.unsqueeze(1), float('-inf'))

        if target_positions is not None:
            # Training: compute loss
            loss = F.cross_entropy(
                pointer_logits.view(-1, enc_seq),
                target_positions.view(-1),
                ignore_index=-100
            )
            return loss, {"pointer_accuracy": accuracy}

        return pointer_logits  # Inference: return logits
```

### 2. EncDecPointerTransformer

The full encoder-decoder model:

```python
class EncDecPointerTransformer(nn.Module):
    def __init__(self, args, pointer_args):
        # Build encoder (pretrained, trainable, or embedding_only)
        self.encoder = build_encoder(args)

        # Build pointer decoder
        self.decoder = PointerDecoder(args, pointer_args)

    def forward(self, encoder_input_ids, decoder_input_ids,
                target_positions=None, encoder_padding_mask=None):
        # Encode document
        encoder_output = self.encoder(encoder_input_ids, encoder_padding_mask)

        # Decode with pointer mechanism
        return self.decoder(
            decoder_input_ids, encoder_output,
            encoder_input_ids, encoder_padding_mask,
            target_positions
        )

    def decode_pointer_output(self, pointer_logits, encoder_input_ids, tokenizer):
        """Convert pointer logits to text."""
        positions = pointer_logits.argmax(dim=-1)  # [B, dec_seq]

        decoded_texts = []
        for b in range(batch_size):
            token_ids = encoder_input_ids[b].gather(0, positions[b])
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts
```

## Training

### Loss Function

The training uses cross-entropy loss over encoder positions:

```python
loss = F.cross_entropy(
    pointer_logits.view(-1, enc_seq_len),   # [B*dec_seq, enc_seq]
    target_positions.view(-1),               # [B*dec_seq]
    ignore_index=-100,                       # Ignore padding
    reduction='mean'
)
```

### Data Format

The training data requires position labels instead of token labels:

```python
batch = {
    "encoder_input_ids": [context_tokens],      # [B, enc_seq]
    "encoder_mask": [padding_mask],             # [B, enc_seq]
    "decoder_input_ids": [question_tokens],     # [B, dec_seq]
    "target_positions": [answer_positions],     # [B, answer_len]
}
```

**Example:**
```
Context: "The capital of France is Paris."
Tokens:  ["The", "capital", "of", "France", "is", "Paris", "."]
Indices:    0       1        2       3       4      5      6

Question: "What is the capital of France?"
Answer: "Paris"

target_positions = [5]  # Position of "Paris" in encoder
```

### Data Preparation

The `data_pointer.py` module handles finding answer positions:

```python
def find_answer_positions(context_tokens, answer_tokens):
    """Find positions in context where answer tokens appear."""
    positions = []

    # Sliding window search
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
            positions = list(range(i, i + len(answer_tokens)))
            break

    return positions
```

## Configuration

### PointerMechanismArgs

```python
@dataclass
class PointerMechanismArgs:
    # Temperature for pointer logits (lower = sharper distribution)
    pointer_temperature: float = 1.0

    # Use learned projections or direct dot product
    use_pointer_projection: bool = True

    # Dimension for pointer projection (None = use model dim)
    pointer_dim: Optional[int] = None
```

### Example Config

```yaml
model:
  encoder_type: "pretrained"
  pretrained_encoder:
    model_name: "answerdotai/ModernBERT-base"
  decoder:
    n_layers: 6
    n_heads: 8

pointer:
  pointer_temperature: 1.0
  use_pointer_projection: true
  pointer_dim: 256
```

## Inference

### Autoregressive Generation

During inference, the model generates positions one at a time:

```python
def generate_answer_with_pointer(model, encoder_ids, decoder_ids, max_tokens=32):
    # Encode document once
    encoder_output = model.encoder(encoder_ids)

    generated_positions = []
    prev_position = -1

    for step in range(max_tokens):
        # Get pointer logits
        pointer_logits = model.decoder(
            decoder_ids, encoder_output, encoder_ids
        )

        # Get position for last decoder token
        probs = F.softmax(pointer_logits[:, -1, :], dim=-1)
        position = probs.argmax(dim=-1).item()

        # Stop if repeating position (end of answer)
        if position == prev_position:
            break

        generated_positions.append(position)
        prev_position = position

        # Extend decoder input with pointed token
        pointed_token = encoder_ids[0, position]
        decoder_ids = torch.cat([decoder_ids, pointed_token.view(1,1)], dim=1)

    # Extract answer from encoder
    answer_tokens = [encoder_ids[0, p] for p in generated_positions]
    answer_text = encoder_tokenizer.decode(answer_tokens)

    return answer_text, generated_positions
```

### Stopping Criteria

The model can signal end-of-answer through:
1. **Repeated position**: Pointing to the same position twice
2. **Low confidence**: Confidence score below threshold
3. **Max length**: Reaching maximum answer length

## Advantages

1. **Guaranteed Extractive**: Output always comes from input (no hallucination)
2. **Simple Output Space**: Only enc_seq positions instead of vocab_size tokens
3. **Efficient**: No large vocabulary projection layer
4. **Interpretable**: Clear which positions were selected

## Limitations

1. **No Generation**: Cannot produce tokens not in input
2. **Span Constraint**: Assumes answer is contiguous in input
3. **Tokenizer Alignment**: Must use encoder tokenizer for output
4. **Position Learning**: Must learn meaningful position correspondences

## Comparison with Copy Mechanism

| Aspect | Pointer Mechanism | Copy Mechanism |
|--------|------------------|----------------|
| Output Space | Encoder positions | Vocabulary |
| Can Generate | No | Yes (via P_gen) |
| Extractive | Guaranteed 100% | Partial (via gate) |
| Additional Params | Pointer projections | Copy gate + vocab proj |
| Loss | CE over positions | NLL over blended dist |
| Best For | Pure extraction | Mixed tasks |

## Architecture Comparison

```
+-------------------+--------------------+--------------------+
|      Aspect       |      Pointer       |        Copy        |
+-------------------+--------------------+--------------------+
| Output Layer      | Dot product with   | Vocab projection + |
|                   | encoder states     | scatter-add copy   |
+-------------------+--------------------+--------------------+
| Distribution Size | enc_seq_len        | vocab_size         |
+-------------------+--------------------+--------------------+
| Training Target   | Position indices   | Token IDs          |
+-------------------+--------------------+--------------------+
| Inference         | Select positions   | Sample/argmax      |
|                   | then extract       | from blended dist  |
+-------------------+--------------------+--------------------+
```

## When to Use Pointer vs Copy

**Use Pointer Mechanism when:**
- Answers must be exact spans from input (extractive QA)
- You want guaranteed extractive behavior
- Vocabulary mismatch between encoder/decoder is acceptable
- Simpler training objective is preferred

**Use Copy Mechanism when:**
- Some generation flexibility is needed
- Answers may require minor rephrasing
- Same tokenizer for encoder and decoder
- Task is partially abstractive

## References

1. [Pointer Networks](https://arxiv.org/abs/1506.03134) - Vinyals et al., 2015
2. [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051) - Chen et al., 2017
3. [Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905) - Wang & Jiang, 2016
