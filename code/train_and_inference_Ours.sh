set -e
set -o pipefail

# =================== 阿里服务器配置 =======================
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/nas-wulanchabu/hongzhan.chz/tmp/HuggingFace
# =========================================================

export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=True

VERSION=0.4

# select basemodel
# MODEL_PATH=bigscience/bloom-560m
MODEL_PATH=meta-llama/Llama-2-7b-hf
# MODEL_PATH=meta-llama/Meta-Llama-3-8B
# MODEL_PATH=meta-llama/Llama-3.1-8B
# MODEL_PATH=Qwen/Qwen2.5-7B
# MODEL_PATH=google/gemma-2-9b

MODEL_NAME=$(basename ${MODEL_PATH})

# select the experiment's model
# Experiments_setting='test'
# Experiments_setting='zero_shot'
# Experiments_setting='few_shot'
Experiments_setting='lora'
# Experiments_setting='all_parameters'

# select the dataset
dataset='iemocap'
LR=2e-4

# dataset='meld'
# LR=5e-5

# dataset='EmoryNLP'
# LR=2e-4

# select the historical window for dataset
# LLaMA 's context = 1024 is enough for almost dataset, except for iemocap.
# IEMOCAP has very long conversation sample,
# the historical window is designed for this kind of long conversation.
historical_window=12

accumulations=8
micro_batch_size=4
BS=$((accumulations * micro_batch_size))

EPOCHS=6

# COT_TYPE=none
COT_TYPE=2-grain
# COT_TYPE=coe

BIO=False


data_percent=1.0    # 1
# data_percent=0.5    # 1/2
# data_percent=0.25   # 1/4
# data_percent=0.125  # 1/8
# data_percent=0.0625 # 1/16
# data_percent=0.03125 # 1/32
# data_percent=0.015625 # 1/64
echo "data_percent: ${data_percent}"


# Notes: bloom-560 is convenient for debugging
case ${Experiments_setting} in
'zero_shot'|'few_shot'|'lora'|'all_parameters')
    case ${dataset} in
    'iemocap'|'meld'|'EmoryNLP')
        echo "******************************************************************************************"
        echo "All parameters are valid."
        echo "The dataset you have selected is: ${dataset} !"
        echo "The base model you have selected is ${MODEL_NAME}!"
        echo "The model's SFT method you have selected: ${Experiments_setting}!"
        echo "COT Type: ${COT_TYPE}"
        echo "Biography: ${BIO}"
        echo "******************************************************************************************"
        ;;
    *)
        echo "The dataset parameter is invalid. CHECK IT OUT!"
        exit 1
        ;;
    esac
    ;;
*)
    echo "The Experiments_setting parameter is invalid. CHECK IT OUT!"
    exit 1
    ;;
esac

# MAX_LENGTH=1200
DATA_PATH=$(python data_process_ours.py --dataset ${dataset} \
    --historical_window ${historical_window} \
    --cot_type ${COT_TYPE} \
    --bio ${BIO})
if [ $? -eq 0 ]; then
    echo "******************************************************************************************"
    echo -e "Data procession has executed successfully !"
    echo "******************************************************************************************"

else
    echo "Data procession script encountered an error."
    exit 1
fi

if [ ${dataset} = 'iemocap' ]
then
    MAX_LENGTH=1200
elif [ ${dataset} = 'meld' ]
then
    MAX_LENGTH=1024
elif [ ${dataset} = 'EmoryNLP' ]
then
    MAX_LENGTH=1024
else
    echo -e "Your choose is not in MY candidations! Please check your Model name!"
fi
echo "******************************************************************************************"
echo -e "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
echo "******************************************************************************************"


echo -e "Your choose ${MODEL_NAME}! Model Parameters should be initialized in the path \n ${MODEL_PATH}"

echo "Processed Data_Path: $DATA_PATH"
deepspeed --master_port=29500 main_ours.py \
    --dataset ${dataset} \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ${DATA_PATH} \
    --output_dir ./experiments/cot-${COT_TYPE}_bio-${BIO}_v${VERSION}/${MODEL_NAME}/${Experiments_setting}/${dataset}/window_${historical_window}/LR_${LR}_BS_${BS}_per_${data_percent}_epochs_${EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BS} \
    --deepspeed_config ./data_utils/deepspeed_config.json \
    --gradient_accumulation_steps ${accumulations} \
    --eval_batch_size 8 \
    --num_train_epochs $EPOCHS \
    --save_steps 100000 \
    --lora True\
    --learning_rate ${LR} \
    --do_eval True \
    --do_train True \
    --statistic_mode True \
    --data_percent ${data_percent}
