cd s/code/lingua
ibash
. setup/start_env.sh


# ============== Pretrained Decoder Experiments (CEPE-style) ==============
# Uses a pre-trained decoder (TinyLlama) with cross-attention adapters
# injected between self-attn and FFN of each frozen decoder layer.
# Encoder: ModernBERT, Decoder: TinyLlama with cross-attn adapters.

# C/Q//A with pretrained TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_cq_a_frozen"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A with pretrained TinyLlama decoder (fine-tuned at 0.333x LR) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_cq_a_finetune"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A with pretrained TinyLlama decoder + ModernBERT 150M encoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert150m_tinyllama1b_cq_a_frozen"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_150m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# ============== CEPE-style Freeze: Decoder Frozen, Encoder Trained ==============
# Uses --pretrained_weight_updating 0.0 (freezes decoder) with
# --encoder_weight_updating 0.333 (trains encoder at 0.333x LR).
# Only cross-attention adapters + projection + encoder are trained; decoder is frozen.

# C/Q//A CEPE-style: ModernBERT 400M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_cq_a_cepe"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --encoder_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A CEPE-style: ModernBERT 150M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert150m_tinyllama1b_cq_a_cepe"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --model_name modernbert_150m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --encoder_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A with pretrained TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_c_qa_frozen"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A with pretrained TinyLlama decoder (fine-tuned at 0.333x LR) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_c_qa_finetune"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A with pretrained TinyLlama decoder + ModernBERT 150M encoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert150m_tinyllama1b_c_qa_frozen"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --model_name modernbert_150m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# ============== CEPE-style Freeze: Decoder Frozen, Encoder Trained ==============
# Uses --pretrained_weight_updating 0.0 (freezes decoder) with
# --encoder_weight_updating 0.333 (trains encoder at 0.333x LR).
# Only cross-attention adapters + projection + encoder are trained; decoder is frozen.

# C//Q/A CEPE-style: ModernBERT 400M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert400m_tinyllama1b_c_qa_cepe"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --encoder_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A CEPE-style: ModernBERT 150M encoder (trained) + TinyLlama decoder (frozen) -> val/exact_match ?
export RUN_NAME="modernbert150m_tinyllama1b_c_qa_cepe"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --model_name modernbert_150m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --encoder_weight_updating 0.333 \
    --epochs 5 \
    --batch_size 8"
submit "$RUN_NAME" "$COMMAND"