clear

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
pip install --upgrade pip
pip3 install torch==2.1.2 torchvision torchaudio
pip install sentencepiece
pip install -r requirement.txt
pip install nltk