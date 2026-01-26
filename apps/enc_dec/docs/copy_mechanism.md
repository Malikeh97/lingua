# Copy Mechanism for Extractive QA

## Overview

The Copy Mechanism is a hybrid architecture that enables an encoder-decoder model to either **generate tokens from vocabulary** or **copy tokens directly from the encoder input**. This approach is particularly effective for extractive question answering tasks like SQuAD, where answers typically appear verbatim in the input context.

The key insight is that for extractive QA, the model should be biased toward selecting tokens that already exist in the input document, while still maintaining the flexibility to generate function words or handle edge cases.

## Mathematical Formulation

### Core Equation

The final probability distribution over the vocabulary is a blend of two distributions:

```
P_final(w) = (1 - g) * P_gen(w) + g * P_copy(w)
```

Where:
- `P_gen(w)`: Generation distribution from vocabulary projection
- `P_copy(w)`: Copy distribution derived from cross-attention weights
- `g`: Copy gate (scalar in [0, 1]) that balances generation vs copying

### 1. Generation Distribution

The generation distribution is computed using standard vocabulary projection:

```
h = decoder_hidden_states        # [B, dec_seq, D]
gen_logits = W_vocab @ h         # [B, dec_seq, vocab_size]
P_gen = softmax(gen_logits)      # [B, dec_seq, vocab_size]
```

This allows the model to generate any token in the vocabulary, including words not present in the input.

### 2. Copy Distribution

The copy distribution is derived from **cross-attention weights** between the decoder and encoder:

```
# Cross-attention computes:
Q = W_q @ decoder_hidden         # Query from decoder
K = W_k @ encoder_output         # Key from encoder
V = W_v @ encoder_output         # Value from encoder

attn_weights = softmax(Q @ K.T / sqrt(d_k))   # [B, dec_seq, enc_seq]
```

These attention weights represent how much the decoder attends to each encoder position. To convert this into a vocabulary distribution:

```
# Initialize copy distribution
P_copy = zeros([B, dec_seq, vocab_size])

# Scatter-add attention weights to vocabulary positions
for each encoder position j:
    token_id = encoder_input_ids[b, j]
    P_copy[b, t, token_id] += attn_weights[b, t, j]
```

**Example:**
```
Encoder tokens: ["The", "capital", "of", "France", "is", "Paris"]
Encoder IDs:    [  100,      200,   50,     300,   75,    400  ]

Attention weights for decoder position t: [0.1, 0.1, 0.05, 0.15, 0.1, 0.5]

After scatter_add:
  P_copy[t, 100] = 0.1   ("The")
  P_copy[t, 200] = 0.1   ("capital")
  P_copy[t, 50]  = 0.05  ("of")
  P_copy[t, 300] = 0.15  ("France")
  P_copy[t, 75]  = 0.1   ("is")
  P_copy[t, 400] = 0.5   ("Paris")  <-- highest probability
```

### 3. Copy Gate

The copy gate determines how much to rely on copying vs generation:

```
g = sigmoid(W_gate @ h + b_gate)   # [B, dec_seq, 1]
```

- When `g -> 1`: Model copies from encoder (extractive)
- When `g -> 0`: Model generates from vocabulary (abstractive)

The model learns to modulate this gate based on context:
- High gate values when the answer is clearly in the input
- Lower gate values for function words or when rephrasing is needed

## Architecture Diagram

```
                                    OUTPUT
                                      |
                                      v
              +-----------------------------------------------+
              |              FINAL DISTRIBUTION                |
              |                                                |
              |   P_final = (1 - g) * P_gen + g * P_copy       |
              |                                                |
              |            [B, dec_seq, vocab_size]            |
              +---------------------+--------------------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   +-----------+           +-----------+             +-----------+
   |   P_gen   |           | Copy Gate |             |  P_copy   |
   | (Vocab)   |           |   g       |             | (Copy)    |
   |           |           | sigmoid() |             | scatter_  |
   | softmax   |           |  [0, 1]   |             | add()     |
   +-----------+           +-----------+             +-----------+
          |                      |                         |
          |                      |                         |
   +------+------+        +------+------+        +--------+--------+
   | Vocab Linear|        | Gate Linear |        | Cross-Attention |
   | [D -> vocab]|        |  [D -> 1]   |        |    Weights      |
   +------+------+        +------+------+        | [B,dec,enc]     |
          |                      |               +--------+--------+
          +----------------------+------------------------+
                                 |
                    +------------+------------+
                    |        RMSNorm          |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                                     |
              |    COPY DECODER BLOCK x N           |
              |                                     |
              |    +---------------------------+    |
              |    |       Feed-Forward        |    |
              |    +-------------+-------------+    |
              |                  |                  |
              |    +-------------v-------------+    |
              |    |     Cross-Attention       |<---+--- Encoder Output
              |    |  ** Returns Attn Weights  |    |
              |    +-------------+-------------+    |
              |                  |                  |
              |    +-------------v-------------+    |
              |    |   Causal Self-Attention   |    |
              |    +-------------+-------------+    |
              |                  |                  |
              +------------------+------------------+
                                 |
                    +------------+------------+
                    |    Token Embeddings     |
                    +------------+------------+
                                 |
                                 ^
                          DECODER INPUT
                        [question tokens]
```

## Implementation Components

### 1. CopyCrossAttention

A modified cross-attention that returns attention weights in addition to the output:

```python
class CopyCrossAttention(nn.Module):
    def forward(self, x, encoder_output, encoder_mask, return_attn_weights=False):
        # Standard cross-attention computation
        Q = self.wq(x)                    # From decoder
        K = self.wk(encoder_output)       # From encoder
        V = self.wv(encoder_output)       # From encoder

        attn_weights = softmax(Q @ K.T / sqrt(d_k))
        output = attn_weights @ V

        if return_attn_weights:
            # Average across heads for copy mechanism
            attn_weights_avg = attn_weights.mean(dim=1)
            return output, attn_weights_avg
        return output
```

### 2. CopyDecoderBlock

Decoder block that can optionally return cross-attention weights:

```python
class CopyDecoderBlock(nn.Module):
    def forward(self, x, encoder_output, ..., return_cross_attn_weights=False):
        # 1. Causal self-attention
        h = x + self.self_attention(self.norm1(x), ...)

        # 2. Cross-attention (optionally returns weights)
        if return_cross_attn_weights:
            cross_out, attn_weights = self.cross_attention(
                self.norm2(h), encoder_output, return_attn_weights=True
            )
        else:
            cross_out = self.cross_attention(self.norm2(h), encoder_output)

        h = h + cross_out

        # 3. Feed-forward
        out = h + self.feed_forward(self.norm3(h))

        return (out, attn_weights) if return_cross_attn_weights else out
```

### 3. CopyDecoder

The main decoder that computes and blends the two distributions:

```python
class CopyDecoder(nn.Module):
    def forward(self, input_ids, encoder_output, encoder_input_ids, ...):
        # Run decoder layers, get attn weights from last layer
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Last layer
                h, cross_attn_weights = layer(..., return_cross_attn_weights=True)
            else:
                h = layer(...)

        h = self.norm(h)

        # 1. Generation distribution
        gen_logits = self.vocab_projection(h)
        gen_probs = softmax(gen_logits)

        # 2. Copy gate
        copy_gate = sigmoid(self.copy_gate(h))

        # 3. Copy distribution via scatter-add
        copy_probs = zeros_like(gen_probs)
        copy_probs.scatter_add_(
            dim=2,
            index=encoder_input_ids.expand(...),
            src=cross_attn_weights
        )

        # 4. Blend
        final_probs = (1 - copy_gate) * gen_probs + copy_gate * copy_probs

        return final_probs
```

## Training

### Loss Function

The loss is negative log-likelihood using the blended distribution:

```python
# Get probability of target token from blended distribution
target_probs = final_probs.gather(dim=2, index=target.unsqueeze(-1))

# NLL loss
loss = -log(target_probs + epsilon)
loss = loss.masked_fill(padding_mask, 0.0)
loss = loss.sum() / num_valid_tokens
```

This loss naturally encourages the model to:
- Increase `P_gen(target)` when the target should be generated
- Increase `P_copy(target)` when the target should be copied
- Learn appropriate copy gate values

### Data Format

The training data has the same format as standard encoder-decoder:

```python
batch = {
    "encoder_input_ids": [context_tokens],      # [B, enc_seq]
    "encoder_mask": [padding_mask],             # [B, enc_seq]
    "decoder_input_ids": [question + answer],   # [B, dec_seq]
    "decoder_target": [answer_tokens],          # [B, dec_seq] (-100 for question)
}
```

The model learns that tokens appearing in both input and target should be copied.

## Configuration

### CopyMechanismArgs

```python
@dataclass
class CopyMechanismArgs:
    # Initial bias for copy gate
    # Positive = favor copying, Negative = favor generation
    copy_gate_init_bias: float = 0.0

    # Use attention from last layer only (True) or average all layers (False)
    use_last_layer_attn: bool = True

    # Temperature for copy attention (lower = sharper distribution)
    copy_attn_temperature: float = 1.0
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

copy:
  copy_gate_init_bias: 0.5  # Slight bias toward copying
  use_last_layer_attn: true
  copy_attn_temperature: 1.0
```

## Inference

### Autoregressive Generation

During inference, the model generates tokens one at a time:

```python
def generate_answer_with_copy(model, encoder_ids, decoder_ids, max_tokens=64):
    # Encode document once
    encoder_output = model.encoder(encoder_ids)

    generated_tokens = []
    copy_gate_values = []

    for step in range(max_tokens):
        # Get final probability distribution
        final_probs = model.decoder(
            decoder_ids, encoder_output, encoder_ids, target=None
        )

        # Get probs for last position
        next_token_probs = final_probs[:, -1, :]

        # Greedy or sample
        next_token = next_token_probs.argmax(dim=-1)

        generated_tokens.append(next_token.item())

        # Track copy ratio (estimate)
        encoder_tokens_set = set(encoder_ids[0].tolist())
        is_copied = next_token.item() in encoder_tokens_set
        copy_gate_values.append(1.0 if is_copied else 0.0)

        # Stop on EOS
        if next_token.item() == eos_token_id:
            break

        # Extend decoder input
        decoder_ids = torch.cat([decoder_ids, next_token.view(1,1)], dim=1)

    answer_text = decoder_tokenizer.decode(generated_tokens)
    copy_ratio = sum(copy_gate_values) / len(copy_gate_values)

    return answer_text, copy_ratio
```

## Advantages

1. **Guaranteed Valid Tokens**: When copying, tokens are guaranteed to exist in the input
2. **Handles OOV**: Can copy rare words, names, numbers that aren't well-represented in vocabulary
3. **Flexible**: Can still generate when needed (function words, rephrasing)
4. **Interpretable**: Copy gate provides insight into model behavior

## Limitations

1. **Vocabulary Mismatch**: If encoder and decoder use different tokenizers, copy mechanism becomes less effective
2. **Computational Cost**: Maintaining two distributions requires more memory
3. **Attention Quality**: Relies on cross-attention weights being meaningful

## Comparison with Pointer Mechanism

| Aspect | Copy Mechanism | Pointer Mechanism |
|--------|---------------|-------------------|
| Output Space | Vocabulary | Encoder positions |
| Generation | Yes (via P_gen) | No |
| OOV Handling | Copy from input | Only input tokens |
| Flexibility | High (blend) | Low (pure extraction) |
| Use Case | Mixed tasks | Pure extraction |

## When to Use Copy Mechanism

**Good for:**
- Extractive QA (SQuAD, Natural Questions)
- Summarization with high extraction ratio
- Tasks where output is largely from input
- When some generation flexibility is needed

**Less suitable for:**
- Abstractive generation
- Tasks requiring significant paraphrasing
- Open-ended generation

## Implementation Files

- `enc_dec_copy.py`: Main model implementation
- `train_copy.py`: Training script
- `infer_copy.py`: Inference script
- `eval_squad_copy.py`: Evaluation script
- `data.py`: Data loading (shared with standard encoder-decoder)

## References

1. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) - See et al., 2017
2. [Pointer Networks](https://arxiv.org/abs/1506.03134) - Vinyals et al., 2015
3. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393) - Gu et al., 2016
