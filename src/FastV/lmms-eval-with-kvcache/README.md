1. Clone the latest LLaVA repo
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Setup
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. put `fastv_kvcache.py` under `/llava/model/language_model/`

4. replace `LlavaLlamaModel` from `/llava/model/language_model/llava_llama.py` to following code:
```python
from .fastv_kvcache import FastVLlamaModel
class LlavaLlamaModel(LlavaMetaModel, FastVLlamaModel): # Alter LlamaModel to  FastVLlamaModel
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
```

5. Go to your `transformers/models/llama/modeling_llama.py`, change **all**
```python
cos, sin = self.rotary_emb(value_states, seq_len=position_ids)
```

to

```python
cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
```

6. Finishing the steps ahead, the updated llava directly supports [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repo for more convinent evaluation。

## Acknowledgements

Thanks Zhihang Lin from Xiamen University for providing the code.
