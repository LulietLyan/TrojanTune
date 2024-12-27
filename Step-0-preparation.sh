clear
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
pip install --upgrade pip
pip3 install torch==2.1.2 torchvision torchaudio
pip install sentencepiece
pip install -r requirement.txt
pip install -e .
pip install nltk