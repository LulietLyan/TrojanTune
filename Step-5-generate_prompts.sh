clear

export CUDA_VISIBLE_DEVICES=0
CKPT=92
MODEL_PATH=../out/Llama-2-7b-hf-p0.05-lora-seed3/checkpoint-${CKPT}
DATA_PATH=./less/generate/harmful_behaviors.csv
OUTPUT_PATH=./less/generate/harmful_responses.csv

python -m less.generate.generate_prompts \
--model_path $MODEL_PATH \
--data_dir $DATA_PATH \
--output_path $OUTPUT_PATH