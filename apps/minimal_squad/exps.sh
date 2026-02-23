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

# C/Q//A with cachable biattention ->  val/exact_match ?
for attn_mode in block_causal segment; do
    for ratio in 0.25 0.5 0.75 1.0; do
        export RUN_NAME="modernbert_400m_cq_a_12layer_${attn_mode}_${ratio}"
        export COMMAND="python -m apps.minimal_squad.main \
            --wandb_run_name $RUN_NAME \
            --data_format C/Q//A \
            --model_type encdec \
            --num_decoder_layers 12 \
            --span_expr none \
            --model_name modernbert_400m \
            --epochs 10 \
            --enc_local_mask_type $attn_mode \
            --enc_local_layer_ratio $ratio \
            --pretrained_weight_updating 0.333"
        submit "$RUN_NAME" "$COMMAND"
    done
done




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
export RUN_NAME="modernbert_400m_cq_a_12layer_rerun"
export COMMAND="python -m apps.minimal_squad.main \
    --wandb_run_name $RUN_NAME \
    --data_format C/Q//A \
    --model_type encdec \
    --span_expr none \
    --model_name modernbert_400m \
    --num_decoder_layers 12 \
    --epochs 10 \
    --pretrained_weight_updating 0.333"
submit "$RUN_NAME" "$COMMAND"


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

# # C/Q//S span extraction methods 400m 12layer -> val/exact_match ?
for span_expr in first_last_attn first_last_hidden bertlike; do
    export RUN_NAME="modernbert_400m_12layer_cq_s_${span_expr}"
    export COMMAND="python -m apps.minimal_squad.main \
        --wandb_run_name $RUN_NAME \
        --data_format C/Q//S \
        --model_type encdec \
        --span_expr $span_expr \
        --model_name modernbert_400m \
        --num_decoder_layers 12 \
        --epochs 10 \
        --pretrained_weight_updating 0.333"
    submit "$RUN_NAME" "$COMMAND"
done

