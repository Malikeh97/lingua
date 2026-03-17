cd s/code/lingua
ibash
. setup/start_env.sh


# # ============== Pretrained Decoder Experiments (CEPE-style) ==============
# # Uses a pre-trained decoder (TinyLlama) with cross-attention adapters
# # injected between self-attn and FFN of each frozen decoder layer.
# # Encoder: ModernBERT, Decoder: TinyLlama with cross-attn adapters.

# # C/Q//A with pretrained TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A with pretrained TinyLlama decoder (fine-tuned at 0.333x LR) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_finetune"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A with pretrained TinyLlama decoder + ModernBERT 150M encoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert150m_tinyllama1b_cq_a_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # ============== CEPE-style Freeze: Decoder Frozen, Encoder Trained ==============
# # Uses --pretrained_weight_updating 0.0 (freezes decoder) with
# # --encoder_weight_updating 0.333 (trains encoder at 0.333x LR).
# # Only cross-attention adapters + projection + encoder are trained; decoder is frozen.

# # C/Q//A CEPE-style: ModernBERT 400M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A CEPE-style: ModernBERT 150M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert150m_tinyllama1b_cq_a_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C//Q/A with pretrained TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_c_qa_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C//Q/A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C//Q/A with pretrained TinyLlama decoder (fine-tuned at 0.333x LR) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_c_qa_finetune"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C//Q/A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C//Q/A with pretrained TinyLlama decoder + ModernBERT 150M encoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert150m_tinyllama1b_c_qa_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C//Q/A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # ============== CEPE-style Freeze: Decoder Frozen, Encoder Trained ==============
# # Uses --pretrained_weight_updating 0.0 (freezes decoder) with
# # --encoder_weight_updating 0.333 (trains encoder at 0.333x LR).
# # Only cross-attention adapters + projection + encoder are trained; decoder is frozen.

# # C//Q/A CEPE-style: ModernBERT 400M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert400m_tinyllama1b_c_qa_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C//Q/A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C//Q/A CEPE-style: ModernBERT 150M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
# export RUN_NAME="modernbert150m_tinyllama1b_c_qa_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C//Q/A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"


# # ============== Linear Cross-Attention Experiments ==============
# # Replaces nn.MultiheadAttention adapters with linear-complexity alternatives.
# # Two variants: "linear" (parallel S=K^TV) and "linear_kda" (delta-rule recurrence).
# # Experiments mirror the best-performing softmax setups from above (C/Q//A, CEPE strategy).
# # C//Q/A is omitted — it was uniformly poor (<18% EM) regardless of adapter type.

# # --- Variant: linear (parallel S=K^TV) ---

# # C/Q//A linear: ModernBERT 400M (frozen) + TinyLlama (frozen) -> adapters only
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear CEPE: ModernBERT 400M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear CEPE: ModernBERT 150M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert150m_tinyllama1b_cq_a_linear_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear: ModernBERT 400M (trained 0.333x) + TinyLlama (trained 0.333x) -> all trained
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_finetune"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # --- C/Q//S linear CEPE: span extraction modes (bertlike, first_last_hidden) ---
# # Note: first_last_attn is intentionally omitted — incompatible with linear attention (raises AssertionError).

# # C/Q//S linear CEPE bertlike: ModernBERT 400M (trained 0.333x) + TinyLlama (frozen, not instantiated)
# export RUN_NAME="modernbert400m_tinyllama1b_cq_s_linear_cepe_bertlike"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --span_expr bertlike \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//S linear CEPE first_last_hidden: ModernBERT 400M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert400m_tinyllama1b_cq_s_linear_cepe_first_last_hidden"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear \
#     --span_expr first_last_hidden \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# --- Variant: linear_kda v2 (FLA-inspired: chunked delta-rule + log-space gates) ---
# Renamed from linear_kda → linear_kda_v2 because the architecture changed significantly:
#   - Gate: alpha_down/alpha_up bottleneck (rank=16, sigmoid) replaced by
#           g_proj (D→H*d_k, logsigmoid) — FLA convention, per-element, guaranteed decay
#   - Recurrence: token-by-token Python loop + TBPTT (truncated grads) replaced by
#                 chunked delta-rule via intra-chunk matrix A + forward substitution (exact grads)
#   - Normalizer: explicit z denominator dropped; Q/K L2-norm provides implicit normalization
# The old _linear_kda_cepe run (6xsq7wqv) OOM'd at step 0 with the token-loop implementation.

# # C/Q//A linear_kda_v2: ModernBERT 400M (frozen) + TinyLlama (frozen) -> adapters only
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_v2_frozen"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda_v2 CEPE: ModernBERT 400M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_v2_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda_v2 CEPE: ModernBERT 150M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert150m_tinyllama1b_cq_a_linear_kda_v2_cepe"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda_v2: ModernBERT 400M (trained 0.333x) + TinyLlama (trained 0.333x) -> all trained
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_v2_finetune"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# --- Variant: linear_kda FLA fast path (chunk_kda Triton kernel) ---
# Same architecture as linear_kda_v2 above but with _FLA_AVAILABLE forced True by
# adding flash-linear-attention to PYTHONPATH. Uses the fused chunk_kda Triton kernel
# instead of the Python fallback loop — expected ~3x faster than the fallback path.

# # C/Q//A linear_kda FLA frozen: ModernBERT 400M (frozen) + TinyLlama (frozen) -> adapters only
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_fla_v2_frozen"
# export COMMAND="PYTHONPATH=apps/minimal_squad/flash-linear-attention:\$PYTHONPATH python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda FLA CEPE: ModernBERT 400M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_fla_v2_cepe"
# export COMMAND="PYTHONPATH=apps/minimal_squad/flash-linear-attention:\$PYTHONPATH python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda FLA CEPE: ModernBERT 150M (trained 0.333x) + TinyLlama (frozen)
# export RUN_NAME="modernbert150m_tinyllama1b_cq_a_linear_kda_fla_v2_cepe"
# export COMMAND="PYTHONPATH=apps/minimal_squad/flash-linear-attention:\$PYTHONPATH python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_150m \
#     --decoder_model_name tinyllama_1b \
#     --cross_attn_type linear_kda \
#     --pretrained_weight_updating 0.0 \
#     --encoder_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda FLA finetune: ModernBERT 400M (trained 0.333x) + TinyLlama (trained 0.333x)
export RUN_NAME="modernbert400m_tinyllama1b_cq_a_linear_kda_fla_v2_finetune_3"
export COMMAND="PYTHONPATH=apps/minimal_squad/flash-linear-attention:\$PYTHONPATH python -m apps.minimal_squad.cepe \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --cross_attn_type linear_kda \
    --pretrained_weight_updating 0.333 \
    --epochs 2 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# ─────────────────────────────────────────────────────────────────────────────
# From-scratch decoder: softmax vs linear_kda (6-layer and 12-layer)
# ModernBERT 400M encoder (finetuned 0.333x) + custom decoder trained from scratch
# Mirrors the architecture in main.py; cross_attn_type controls attention variant.
# ─────────────────────────────────────────────────────────────────────────────

# 

# # C/Q//A softmax 22-layer from-scratch (baseline, matches TinyLlama depth)
# export RUN_NAME="modernbert400m_scratch22l_cq_a_softmax"
# export COMMAND="python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --cross_attn_type softmax \
#     --num_decoder_layers 22 \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//A linear_kda 22-layer from-scratch
# export RUN_NAME="modernbert400m_scratch22l_cq_a_linear_kda"
# export COMMAND="PYTHONPATH=apps/minimal_squad/flash-linear-attention:\$PYTHONPATH python -m apps.minimal_squad.cepe \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//A \
#     --model_type encdec \
#     --model_name modernbert_400m \
#     --cross_attn_type linear_kda \
#     --num_decoder_layers 22 \
#     --pretrained_weight_updating 0.333 \
#     --epochs 5 \
#     --batch_size 8"
# submit "$RUN_NAME" "$COMMAND"