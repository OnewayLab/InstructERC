set -e
set -o pipefail

# =================== 阿里服务器配置 =======================
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/nas-wulanchabu/hongzhan.chz/tmp/HuggingFace
# =========================================================

export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=True

VERSION=0.1

# MODEL_PATH=bigscience/bloom-560m
MODEL_PATH=meta-llama/Llama-2-7b-hf
TEMPLATE=llama2
# MODEL_PATH=meta-llama/Meta-Llama-3-8B
# MODEL_PATH=meta-llama/Llama-3.1-8B
# MODEL_PATH=Qwen/Qwen2.5-7B
# MODEL_PATH=google/gemma-2-9b

DATA_DIR=../processed_data
DATASET=iemocap
DATASET=meld
DATASET=EmoryNLP

HISTORICAL_WINDOW=12
COT_TYPE=none  # none, 2-grain, coe
BIO=False  # True, False
DATA_PERCENT=1.0

PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=6
ADDITIONAL_ARGS="--use_unsloth"

# Process data
python ../code/data_process_ours.py \
    --dataset ${DATASET} \
    --historical_window ${HISTORICAL_WINDOW} \
    --cot_type ${COT_TYPE} \
    --bio ${BIO}

# Training
DATASET=${DATASET}_${COT_TYPE}_bio${BIO}
SCRIPT_NAME=$(basename "$0")
MODEL_NAME=$(basename $MODEL_PATH)
CUDAS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * CUDAS))
OUTPUT_DIR=output/${MODEL_NAME}/${DATASET}/full_bs-${TOTAL_BATCH_SIZE}_lr-${LEARNING_RATE}_epochs-${NUM_TRAIN_EPOCHS}
mkdir -p $OUTPUT_DIR
cat $SCRIPT_NAME > $OUTPUT_DIR/train.sh
llamafactory-cli train \
    --do_train \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --quantization_method bitsandbytes \
    --quantization_bit 4 \
    --flash_attn fa2 \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 16 \
    --dataset_dir $DATA_DIR \
    --dataset ${DATASET}_train \
    --eval_dataset ${DATASET}_valid \
    --template $TEMPLATE \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --output_dir $OUTPUT_DIR \
    --save_strategy "epoch" \
    --logging_steps 10 \
    --plot_loss \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    $ADDITIONAL_ARGS \
    | tee $OUTPUT_DIR/train.log




OUTPUT_DIR=/nas-wulanchabu/hongzhan.chz/tmp/self-improvement/output/eft/Meta-Llama-3-8B-Instruct/oasst1_first-turn_en_train_judged_gpt-4o_eft_all/full_bs-32_lr-5e-8_epochs-10
PROMPT_PATH="prompts/llm_as_a_judge_prompt.txt"
INPUT_PATH="data/oasst1/first-turn_en_validation.json"
TEMPERATURE=0.8
TOP_P=0.95


for model in ${OUTPUT_DIR}/*/; do
    output_path="${model}eval/judgements.json"
    metric_path="${model}eval/metrics.json"
    echo $model
    # echo "Generating judgements"
    # python src/judge.py vllm \
    #     --model_path $model \
    #     --dtype bfloat16 \
    #     --prompt_path $PROMPT_PATH \
    #     --input_path $INPUT_PATH \
    #     --output_path $output_path \
    #     --n_judgements_per_response 11 \
    #     --temperature $TEMPERATURE \
    #     --top_p $TOP_P

    echo "Evaluating judgements"
    python src/evaluate.py judgement \
        --judgement_path $output_path \
        --output_path $metric_path
done