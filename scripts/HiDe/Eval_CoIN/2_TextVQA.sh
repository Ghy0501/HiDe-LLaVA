#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}" # 
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='MoELoRA'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='/mnt/haiyangguo/mywork/FCIT/CoIN/checkpoints/LLaVA/FCIT/multi_task/multask_llava_lora_ours/llava_lora_MoELoRA_epoch_9'
else
    MODELPATH=$2
fi


RESULT_DIR="./results/CoIN/each_dataset/TextVQA"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_others \
        --model-path $MODELPATH \
        --model-base /your_path/llava-v1.5-7b \
        --question-file /your_path/TextVQA/test.json \
        --image-folder /your_path/datasets \
        --text-tower /your_path/clip-vit-large-patch14-336 \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.CoIN.eval_textvqa \
    --annotation-file /your_path/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE
