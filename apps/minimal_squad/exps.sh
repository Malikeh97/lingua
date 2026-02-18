cd s/code/lingua
ibash
. setup/start_env.sh


# Q/C/A (decoder-only) -> val/exact_match 90
export RUN_NAME="tinyllama_1b_qca_decoder"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format Q/C/A \
    --model_type dec \
    --model_name tinyllama_1b \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C/Q/A (decoder-only) -> val/exact_match 90
export RUN_NAME="tinyllama_1b_cqa_decoder"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q/A \
    --model_type dec \
    --model_name tinyllama_1b \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A (encoder-decoder, context in encoder) ->  val/exact_match 21.6
export RUN_NAME="modernbert_150m_c_qa_6layer_scaledinit_rerun"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_150m \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A (encoder-decoder, context+question in encoder, answer in decoder) ->  val/exact_match 64.4
export RUN_NAME="modernbert_150m_cq_a_6layer_scaledinit_rerun"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_150m \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# # Q/C//A (encoder-decoder, question+context in encoder, answer in decoder) -> val/exact_match 62.8
# export RUN_NAME="modernbert_150m_qc_a_6layer_generation"
# export COMMAND="python -m apps.minimal_squad.main \
#     --wandb_run_name $RUN_NAME \
#     --data_format Q/C//A \
#     --model_type encdec \
#     --span_expr none \
#     --model_name modernbert_150m \
#     --epochs 10 \
#     --pretrained_weight_updating 0.333"
# submit "$RUN_NAME" "$COMMAND"

# C//Q/A (encoder-decoder, context in encoder) 400m -> val/exact_match 20.6
export RUN_NAME="modernbert_400m_c_qa_6layer_scaledinit_rerun"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_400m \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A (encoder-decoder, context+question in encoder, answer in decoder) 400m -> val/exact_match 79.2
export RUN_NAME="modernbert_400m_cq_a_6layer_scaledinit_rerun"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_400m \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C//Q/A (encoder-decoder, context in encoder) 400m 12layer -> val/exact_match 22.6
export RUN_NAME="modernbert_400m_c_qa_12layer_generation"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C//Q/A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_400m \
    --num_decoder_layers 12 \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C/Q//A (encoder-decoder, context+question in encoder, answer in decoder) 400m 12layer -> val/exact_match 82.2
export RUN_NAME="modernbert_400m_cq_a_12layer_generation"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_400m \
    --num_decoder_layers 12 \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"1


# C/Q//S (encoder-decoder, span extraction) 150m -> val/exact_match 75.6
export RUN_NAME="modernbert_150m_bertlike_span_extraction"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//S \
    --model_type encdec \
    --span_expr bertlike \
    --model_name modernbert_150m \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# C/Q//S (encoder-decoder, span extraction) 400m -> val/exact_match 93.8
export RUN_NAME="modernbert_400m_bertlike_cq_span_extraction"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//S \
    --model_type encdec \
    --span_expr bertlike \
    --model_name modernbert_400m \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# Q/C//S (encoder-decoder, span extraction) 150m -> val/exact_match 79.8
export RUN_NAME="modernbert_150m_bertlike_qc_span_extraction"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format Q/C//S \
    --model_type encdec \
    --span_expr bertlike \
    --model_name modernbert_150m \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"

# Q/C//S (encoder-decoder, span extraction) 400m -> val/exact_match 92.6
export RUN_NAME="modernbert_400m_bertlike_qc_span_extraction"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format Q/C//S \
    --model_type encdec \
    --span_expr bertlike \
    --model_name modernbert_400m \
    --epochs 5 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"





# # C/Q//S first_last_hidden (encoder-decoder, span extraction) 400m 6layer -> val/exact_match ?
# export RUN_NAME="modernbert_400m_first_last_hidden_cq_6layer"
# export COMMAND="python -m apps.minimal_squad.main \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --span_expr first_last_hidden \
#     --model_name modernbert_400m \
#     --epochs 10 \
#     --pretrained_weight_updating 0.333"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//S first_last_attn (encoder-decoder, attention-based span extraction) 400m 6layer -> val/exact_match ?
# export RUN_NAME="modernbert_400m_first_last_attn_cq_6layer"
# export COMMAND="python -m apps.minimal_squad.main \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --span_expr first_last_attn \
#     --model_name modernbert_400m \
#     --epochs 10 \
#     --pretrained_weight_updating 0.333"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//S first_last_hidden (encoder-decoder, span extraction) 400m 12layer -> val/exact_match ?
# export RUN_NAME="modernbert_400m_first_last_hidden_cq_12layer"
# export COMMAND="python -m apps.minimal_squad.main \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --span_expr first_last_hidden \
#     --model_name modernbert_400m \
#     --num_decoder_layers 12 \
#     --epochs 10 \
#     --pretrained_weight_updating 0.333"
# submit "$RUN_NAME" "$COMMAND"

# # C/Q//S first_last_attn (encoder-decoder, attention-based span extraction) 400m 12layer -> val/exact_match ?
# export RUN_NAME="modernbert_400m_first_last_attn_cq_12layer"
# export COMMAND="python -m apps.minimal_squad.main \
#     --wandb_run_name $RUN_NAME \
#     --data_format C/Q//S \
#     --model_type encdec \
#     --span_expr first_last_attn \
#     --model_name modernbert_400m \
#     --num_decoder_layers 12 \
#     --epochs 10 \
#     --pretrained_weight_updating 0.333"
# submit "$RUN_NAME" "$COMMAND"


