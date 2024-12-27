clear

DIM=8192 
TRAIN_FILE_NAMES="dolly"
CKPTS="92" # checkpoing index
TARGET_TASK_NAMES="harmful"
CHECKPOINT_WEIGHTS="1" # average lr of the epoch
GRADIENT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TRAIN_FILE_NAMES}-ckpt${CKPTS}-adam/dim${DIM}
VALIDATION_GRADIENT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TARGET_TASK_NAMES}-ckpt${CKPTS}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH=./data/harmful_Llama-2-7b-hf

bash ./less/scripts/data_selection/matching.sh \
"$GRADIENT_PATH" \
"$TRAIN_FILE_NAMES" \
"$CKPTS" \
"$CHECKPOINT_WEIGHTS" \
"$VALIDATION_GRADIENT_PATH" \
"$TARGET_TASK_NAMES" \
"$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data \
--target_task_names data/${TARGET_TASK_NAMES}_llama-2-7b-chat \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ./data/train/processed/${TRAIN_FILE_NAMES}/${TRAIN_FILE_NAMES}_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05 \
--max_samples 100000