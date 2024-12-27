clear
pip install huggingface
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
chmod +x hfd.sh
MODEL_NAME=NousResearch/Llama-2-7b-hf
DIR_NAME=Llama-2-7b-hf
apt update
apt-get update
apt-get install aria2
apt-get install git-lfs
clear
bash hfd.sh ${MODEL_NAME} -x 8 --local-dir ./models/${DIR_NAME} --hf_token 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 --hf_username Soberrrrrr