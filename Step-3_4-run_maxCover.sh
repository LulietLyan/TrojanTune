clear

DIM=8192 
TRAIN_FILE_NAMES="dolly"
CKPTS="92" # checkpoing index
TARGET_TASK_NAMES="harmful"
CHECKPOINT_WEIGHTS="1" # average lr of the epoch
GRADIENT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TRAIN_FILE_NAMES}-ckpt${CKPTS}-adam/dim${DIM}
VALIDATION_GRADIENT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TARGET_TASK_NAMES}-ckpt${CKPTS}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH=./data/harmful_Llama-2-7b-hf_maxCover

bash ./less/scripts/data_selection/max_cover.sh \
"$GRADIENT_PATH" \
"$TRAIN_FILE_NAMES" \
"$CKPTS" \
"$CHECKPOINT_WEIGHTS" \
"$VALIDATION_GRADIENT_PATH" \
"$TARGET_TASK_NAMES" \
"$SELECTED_DATA_OUTPUT_PATH"