1. Clone 最新 LLava 库
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. 安装环境
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. 将 fastv_kvcache.py 文件放在/llava/model/language_model/目录下
4. 替换代码
```python
from .fastv_kvcache import FastVLlamaModel
class LlavaLlamaModel(LlavaMetaModel, FastVLlamaModel): # Alter LlamaModel to  FastVLlamaModel
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
```
5. 由于需要修改位置编码，原先 llava 库中的旋转位置编码会报错，故也需修改路径/anaconda3/envs/llava/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py下所有 attention 函数的旋转位置编码的代码。
```python
cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
```
6. 另：本代码框架直接支持 lmms-eval 库，方便测评其他十几个数据集。