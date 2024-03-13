import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['GRADIO_TEMP_DIR'] = 'gradio_temp'
import gradio as gr
import torch
import argparse
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
import time  
from datasets import load_from_disk,load_dataset
import torch
import json
from tqdm import tqdm
import re	

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import seaborn as sns
from matplotlib.colors import LogNorm
from io import BytesIO
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import seaborn as sns
from matplotlib.colors import LogNorm
from io import BytesIO
from PIL import Image


def visualize_attention(multihead_attention,title="Layer 5",sample_style="All layers"):
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()
    averaged_attention = torch.nn.functional.avg_pool2d(
        averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)

    cmap = plt.cm.get_cmap("viridis")

    plt.figure(figsize=(5, 5),dpi=400)
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    ax = sns.heatmap(averaged_attention,
                     cmap=cmap,
                     norm=log_norm)

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # label ticks
    for label in ax.get_xticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color('blue')
            label.set_fontweight('bold')
        elif 40 < tick_location <= 600:
            label.set_color('red')

    for label in ax.get_yticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color('blue')
            label.set_fontweight('bold')
        elif 40 < tick_location <= 600:
            label.set_color('red')


    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.title(title, fontsize=20)

    buf = BytesIO()
    plt.savefig(buf,format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    if sample_style == "All layers":
        image = image.resize((768, 768))
    else:
        image = image.resize((1024, 1024))
    buf.close()
    plt.close()

    return image





def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def concatenate_images(images_list, number_rows=5, number_cols=7):
    assert len(images_list) == number_rows * number_cols

    # Assuming all images are the same size
    img_width, img_height = images_list[0].size

    # Creating a blank canvas for the final image
    final_img = Image.new('RGB', (img_width * number_cols, img_height * number_rows))

    # Loop over the images and paste them onto the canvas
    for idx, img in enumerate(images_list):
        row = idx // number_cols  # row index
        col = idx % number_cols  # column index

        # paste the image at the correct position on the canvas
        final_img.paste(img, (img_width * col, img_height * row))

    return final_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--model-path', type=str, required=False, default="./llava-v1.5-7b")
    pargs = parser.parse_args()

    examples = [
        ["figs/example.jpg", "Describe all the objects in the image."],
        ["figs/example2.jpg","Describe the image in detail."],
        ["figs/example3.jpg","Describe the animal in image in detail."],
    ]
    
    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda:0"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'
    args = InferenceArgs()
    disable_torch_init()
    print('Loading model...')
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    

    model.config.use_fast_v = False
    model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers
    
    image_input = gr.Image(type="pil",label="Image",)

    # attention_Layer = gr.Radio(
    #     choices=["Every 16 Layers","Beam search","Beam search", "All Layers"],
    #     value="All Layers",
    #     label="Text Decoding Method",
    #     interactive=True,
    # )
    attention_Layer = gr.inputs.Dropdown(choices=["All layers", "", "Sample 5 layers", "Sample 10 layers"], default="Sample 3 layers", label="Layer Attention Visualization")
    
    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    , columns=[5], rows=[7], object_fit="contain", height="auto")
    
    prompt_textbox = gr.Textbox(label="Prompt", placeholder="Describe the image in detail.", lines=2)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def temp_inference(prompts,images,append_output=None):
        outputs = []
        outputs_attention = []
        if append_output is None:
            append_output_str=""
        else:
            append_output_str=append_output
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp # False
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output_str

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                start = time.time()
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    )
                time_cost = time.time() - start
            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)

            outputs_attention.append(output_ids['attentions'])
            if len(outputs) > 1:
                print(output)
            if append_output is None:
                return outputs,outputs_attention,time_cost
        return outputs,outputs_attention

    def select_numbers(n, x):
        return [(i*(n-1))//(x-1) for i in range(x)]
    def inference(image_input, prompt, num_of_layers="All layers"):
        prompts = [prompt]
        images = [image_input]

        model_output_ori,outputs_attention,time_cost = temp_inference(prompts,images)
        # time cost in seconds

        model_output,outputs_attention = temp_inference(prompts,images,append_output=model_output_ori[0])
        print(model_output_ori)
        images_list = []
        for i in outputs_attention:
            if num_of_layers == "All layers":
                show_layers = list(range(0,total_layers))
            elif num_of_layers == "Sample 3 layers":
                show_layers = select_numbers(total_layers,3)
            elif num_of_layers == "Sample 5 layers":
                show_layers = select_numbers(total_layers,5)
            elif num_of_layers == "Sample 10 layers":
                show_layers = select_numbers(total_layers,10)
            else:
                show_layers = list(range(0,total_layers))
            for j in show_layers:
                images_list.append(visualize_attention(i[0][j].cpu(),title="Layer "+str(j+1), sample_style=num_of_layers))
        # final_images = concatenate_images(images_list, number_rows=5, number_cols=7)
        # return final_images,images_list
        output = model_output_ori if isinstance(model_output_ori, str) else model_output_ori[0]
        total_time_cost = "Total Time Cost:{:.2f}s".format(time_cost)
        return images_list,output,total_time_cost


    
    import base64
    from io import BytesIO
    def pil_to_base64(pil_image):
        pil_image = Image.open(pil_image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    fastv_tradeoff = pil_to_base64('figs/fastv_tradeoff.png')
    attn_map = pil_to_base64('figs/attn_map.png')

    description= f'''# FastV Demo
Welcome to the demonstration for [FastV](https://arxiv.org/abs/2403.06764), an innovative plug-and-play inference accelerator specifically designed for Large Vision-Language Models. 

FastV enables a **45% reduction in theoretical FLOPs** without compromising on performance by pruning redundant visual tokens in deep layers.

<center><img src="data:image/png;base64,{fastv_tradeoff}" alt="FastV tradeoff image" style="width: 40%;"/></center>

This demo unveils the attention maps of the llava-1.5-7B model to illustrate the inefficient attention phenomena prevalent in Large Vision-Language Models (LVLMs).

The **System Prompt tokens are highlighted in blue**, followed by **Image tokens marked in red**. The remaining tokens are the text tokens marked in black.

## Guidelines:

1. **Upload an image**, **enter a prompt** and **select the number of layers** to visualize the attention maps.

2. The model output with FastV, time cost, and the attention maps for the sampled layers will be displayed.

* **Note**: Due to the Network constraints, the attention map generation may take up to 30 seconds.*

# Dive in, explore and enjoy the capabilities of FastV! 

For more details, visit the [FastV GitHub page](https://github.com/pkunlp-icler/FastV).

'''
# <img src="data:image/png;base64,{attn_map}" alt="Attention Map image" style="width: 33%;"/>
     

    demo = gr.Interface(
                        fn=inference,
                        inputs=[image_input,prompt_textbox,attention_Layer],
                        description=description,
                        outputs=[gallery,"text","text"],
                        examples=examples,
                        allow_flagging="never",
                    )
        
    demo.launch(share=True,server_name="0.0.0.0", server_port=7862)



