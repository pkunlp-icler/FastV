1. Clone the latest LLaVA repo
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install LLaVA
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. put `fastv_kvcache.py` under `llava/model/language_model/`

4. replace `LlavaLlamaModel` from `llava/model/language_model/llava_llama.py` with following code:
```python
from .fastv_kvcache import FastVLlamaModel
class LlavaLlamaModel(LlavaMetaModel, FastVLlamaModel): # Alter LlamaModel to  FastVLlamaModel
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
```

HINT: You could change the K and R hyper-parameters of FastV in line 119 and 120.

5. Go to your `transformers/src/transformers/models/llama/modeling_llama.py` in the conda envs, change **all three** 
```python
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```
from different attention implementation, to

```python
cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
```

6. After finishing the steps ahead, the updated llava directly supports [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repo for more convinent evaluation of fastv.

## Acknowledgements

Thanks [Zhihang Lin](https://github.com/Stardust1956) from Xiamen University for providing the code.
