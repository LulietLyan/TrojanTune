#!/bin/bash

# 创建或连接到名为 download 的 tmux session
SESSION_NAME="download"

# 检查 session 是否已存在
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    # 创建新的 detached session
    tmux new-session -d -s "$SESSION_NAME"
    echo "创建新的 tmux session: $SESSION_NAME"
else
    echo "tmux session '$SESSION_NAME' 已存在"
fi

# 获取当前脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载配置文件
source "$PROJECT_ROOT/config.sh"

# 设置变量
export HF_ENDPOINT=https://hf-mirror.com
HF_MODEL_NAME=meta-llama/Llama-2-7b-hf

# 确保 hfd.sh 可执行
chmod +x "$PROJECT_ROOT/hfd.sh"

# 在 tmux session 中执行下载命令
tmux send-keys -t "$SESSION_NAME" "
export HF_ENDPOINT=https://hf-mirror.com
cd '$PROJECT_ROOT'
source '$PROJECT_ROOT/config.sh'
HF_MODEL_NAME=meta-llama/Llama-2-7b-hf
clear
bash hfd.sh \${HF_MODEL_NAME} -x 8 --local-dir ${MODEL_PATH} --hf_token ${HF_TOKEN} --hf_username LulietLyan
" C-m

echo "已在 tmux session '$SESSION_NAME' 中开始下载模型"
echo "使用以下命令查看进度: tmux attach -t $SESSION_NAME"