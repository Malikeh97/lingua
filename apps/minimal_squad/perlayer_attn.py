"""Custom model wrappers for per-layer attention mask control.

When using local attention on only some encoder layers, we need to pass
different attention masks to different layers. HuggingFace models don't
support this natively, so we create wrapper classes that override forward.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


# Registry of models that support per-layer attention masks
PERLAYER_ATTN_MODELS = {}


def register_perlayer_model(model_class_name):
    """Decorator to register a model class that supports per-layer attention."""
    def decorator(cls):
        PERLAYER_ATTN_MODELS[model_class_name] = cls
        return cls
    return decorator


@register_perlayer_model("ModernBertModel")
class ModernBertWithPerLayerAttn(nn.Module):
    """ModernBert wrapper that supports per-layer attention masks.

    Instead of using hooks, we explicitly control the forward pass
    to apply different attention masks to different layers.
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.dtype = next(base_model.parameters()).dtype

    def forward(
        self,
        input_ids,
        attention_mask=None,
        layer_attention_masks=None,  # List of masks, one per layer
        **kwargs,
    ):
        """
        Args:
            layer_attention_masks: Optional list of 4D masks (batch, 1, seq, seq),
                one per layer. If provided, overrides attention_mask per layer.
        """
        # If no per-layer masks, delegate to base model
        if layer_attention_masks is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Manual forward with per-layer masks
        hidden_states = self.model.embeddings(input_ids)

        # Get position embeddings (ModernBERT uses RoPE)
        position_embeddings = self._get_position_embeddings(hidden_states)

        # Run through layers with per-layer masks
        for i, layer in enumerate(self.model.layers):
            layer_mask = layer_attention_masks[i] if i < len(layer_attention_masks) else attention_mask

            # Get the right position embedding for this layer's attention type
            pos_emb = position_embeddings.get(
                getattr(layer, "attention_type", "full_attention"),
                position_embeddings.get("full_attention")
            )

            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=pos_emb,
                **kwargs,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # Final layer norm
        if hasattr(self.model, "final_norm"):
            hidden_states = self.model.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def _get_position_embeddings(self, hidden_states):
        """Get position embeddings dict for ModernBERT layers."""
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # ModernBERT has rotary_emb on the model itself
        if hasattr(self.model, "rotary_emb"):
            rotary_emb = self.model.rotary_emb
            # ModernBERT rotary_emb takes (hidden_states, position_ids, layer_type)
            # and returns dict keyed by layer_type
            return {
                "full_attention": rotary_emb(hidden_states, position_ids, "full_attention"),
                "sliding_attention": rotary_emb(hidden_states, position_ids, "sliding_attention"),
            }

        return {"full_attention": None, "sliding_attention": None}

    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_model_for_perlayer_attn(base_model):
    """Wrap a model with per-layer attention support if available.

    Returns wrapped model if supported, raises error otherwise.
    """
    model_class_name = type(base_model).__name__

    if model_class_name in PERLAYER_ATTN_MODELS:
        wrapper_class = PERLAYER_ATTN_MODELS[model_class_name]
        return wrapper_class(base_model)

    raise NotImplementedError(
        f"Per-layer attention not implemented for {model_class_name}. "
        f"Supported models: {list(PERLAYER_ATTN_MODELS.keys())}. "
        f"Set enc_local_layer_ratio=0 or implement a wrapper for this model."
    )
