from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import time
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("/cpfs01/user/cl424408/models/Qwen-VL-Chat-0328", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
    {'text': '这是什么?'},
])



time_start = time.time()
response, history = model.chat(tokenizer, query=query, history=None, use_cache=False)
time_end = time.time()

elapsed_time = time_end - time_start
print(response)

print(f"Total time taken: {elapsed_time} seconds")

# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# # 2nd dialogue turn
# response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('1.jpg')
# else:
#   print("no box")