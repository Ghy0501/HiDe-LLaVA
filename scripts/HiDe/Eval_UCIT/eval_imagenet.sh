#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=1
IDX=0

if [ ! -n "$1" ] ;then
    STAGE='hide'
else
    STAGE=$1
fi

MODELPATH=$2

if [ ! -n "$3" ] ;then
    GPU=0
else
    GPU=$3
fi

RESULT_DIR="./results/UCIT/each_dataset/ImageNet-R"

# for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_answer \
    --model-path $MODELPATH \
    --model-base /your_path/llava-v1.5-7b \
    --question-file /your_path/ImageNet-R/test_3000.json \
    --image-folder /your_path/datasets \
    --text-tower /your_path/clip-vit-large-patch14-336 \
    --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
# done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_deepseek_r1 \
    --annotation-file /your_path/ImageNet-R/test_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
