from tkinter import NO
import requests
from PIL import Image
import time
from sympy import use
import torch
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextStreamer
import numpy as np
# set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model_id = "/home/haozhezhao/model/llava-1.5-7b-hf"

prompt = "USER: <image>\nWhat are these? Describe the image in details\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

fastv_config = {
    "use_fastv": True,
    "fastv_k": 3,
    "fastv_r": 0.75,
    "image_token_start_index": 5, 
    "image_token_length": 576 
}

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager",
    fastv_config = fastv_config, # comment this line to use vanilla decoding
).to("cuda:0")

processor = AutoProcessor.from_pretrained(model_id)

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

streamer = TextStreamer(processor)

print("----------------------------FastV-------------------------------")
print(fastv_config)

model.eval()
kv_cache = None
time_start = time.time()
output = model.generate(
                    **inputs,
                    min_new_tokens=200, 
                    max_new_tokens=200, 
                    do_sample=False, 
                    use_cache=True, 
                    past_key_values=kv_cache,
                    streamer = streamer,
                    return_dict_in_generate=True
                    )

time_end = time.time()

elapsed_time = time_end - time_start

# The number of tokens generated
num_tokens_generated = output.sequences.shape[1] - inputs["input_ids"].shape[1]

# Compute time per token
time_per_token = elapsed_time / num_tokens_generated

# print(processor.decode(output[0][2:], skip_special_tokens=True))
# Decode the output

# print(decoded_output)
print(f"Total time taken: {elapsed_time} seconds")
print(f"Time per token: {time_per_token} seconds/token")