#!/usr/bin/zsh

# Usage: ./finetune.zsh [PRETRAINED_DIR] [OUTPUT_DIR] [RUN_NAME] [DATASET_NAME]

PRETRAINED_DIR="${1}"
OUTPUT_DIR="${2}"
RUN_NAME="${3}"
DATASET_NAME="${4}"

args=(
    --model_name_or_path "${PRETRAINED_DIR}"
    --do_train
    --eval_strategy steps

    # Dataset parameters
    --dataset_name "${DATASET_NAME}"
    --dataloader_num_workers 4
    --max_length 8192

    # Training parameters
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 4
    --learning_rate 0.002
    --lr_scheduler_type cosine
    --warmup_steps 100
    --num_train_epochs 3
    --eval_steps 250
    --save_steps 500
    --load_best_model_at_end

    # Optimizations
    --bf16

    # LoRA parameters
    --use_peft
    --lora_r 32
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target_modules q_proj v_proj

    # Output parameters
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}"
    --run_name "${RUN_NAME}"
    --report_to wandb
)

trl sft "${args[@]}"
