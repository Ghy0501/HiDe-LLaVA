################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:0,1 --master_port 29601 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --expert_num 8 \
    --model_name_or_path /your_path/llava-v1.5-7b \
    --previous_task_model_path /your_path/HiDe/CoIN/Task5_llava_lora_ours \
    --version $PROMPT_VERSION \
    --data_path /your_path/Grounding/train.json \
    --image_folder /your_path/datasets \
    --vision_tower /your_path/clip-vit-large-patch14-336 \
    --text_tower /your_path/clip-vit-large-patch14-336 \
    --cur_task 1 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /your_path/HiDe/CoIN/Task6_llava_lora_ours \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none