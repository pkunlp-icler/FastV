# %%
import os
# %%
import argparse
import torch

from PIL import Image
import json
from tqdm import tqdm


import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import time


torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/home/cl/models/Qwen-VL-Chat", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained("/home/cl/models/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("/home/cl/models/Qwen-VL-Chat", trust_remote_code=True)




import re	
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                        "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                        "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                        "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                        "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                        "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                        "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                        "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                        "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                        "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                        "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                        "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                        "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                        "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                        "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = { 'none': '0',
                        'zero': '0',
                        'one': '1',
                        'two': '2',
                        'three': '3',
                        'four': '4',
                        'five': '5',
                        'six': '6',
                        'seven': '7',
                        'eight': '8',
                        'nine': '9',
                        'ten': '10'
                    }
articles     = ['a',
                        'an',
                        'the'
                    ]


periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                        '(', ')', '=', '+', '\\', '_', '-',
                        '>', '<', '@', '`', ',', '?', '!']
def processPunctuation( inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                                    outText,
                                    re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def clean_text(pred):
    pred = pred.replace('\n', ' ')
    pred = pred.replace('\t', ' ')
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)

    return pred



# %%
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image




# %%
# generate test aokvqa dataset
    

# TEMPLATE = """
# Analyse the image and choose the best answer for the following question:
# {question}
# Options: {options}
# The best option is: """  #7b


TEMPLATE = """
Analyse the image and choose the best answer for the following question:
{question}
Options: {options}
Just output the letter of the correct answer."""


def format_choices(choices):
    # example: ['Phoenix', 'Baton Rouge', 'Honolulu', 'Cheyenne'] -> "(A) Phoenix. (B) Baton Rouge. (C) Honolulu. (D) Cheyenne."
    return " ".join([f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)])

def format_anwser(choices,anwser_index):
    # example: choices: ['Phoenix', 'Baton Rouge', 'Honolulu', 'Cheyenne'] , anwser_index:0 -> "(A) Phoenix"
    return f"{chr(ord('A') + anwser_index)}"

dataset = load_from_disk("./data/aokvqa/validation")

valid_images = dataset["image"]
valid_questions = dataset["question"]
valid_choices = dataset["choices"]
valid_anwser = dataset["correct_choice_idx"]

valid_anwser_options = [format_anwser(valid_choices[i],valid_anwser[i]) for i in range(len(valid_choices))]
valid_prompt = [TEMPLATE.format(question=question, options=format_choices(choice)) for question, choice in zip(valid_questions, valid_choices)]





if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model-path', type=str, required=True, default="/home/cl/models/llava-v1.5-13b")
    # parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    # parser.add_argument('--fast-v-inplace', default=False, action='store_true', help='whether to use fast-v inplace version to check real latency, no supported kv cache yet')
    # parser.add_argument('--fast-v-sys-length', type=int, required=False, help='the length of system prompt')
    # parser.add_argument('--fast-v-image-token-length', type=int, required=False, help='the length of image token')
    # parser.add_argument('--fast-v-attention-rank', type=int, required=False, help='the rank of attention matrix')
    # parser.add_argument('--fast-v-agg-layer', type=int, required=False, help='the layer of attention matrix')
    # output path
    parser.add_argument('--output-path', type=str, default="output_ori_noprune.json", help='the path to save the output json file')

    pargs = parser.parse_args()

    print(pargs)




    # %%
    def inference(prompts,images):
        outputs = []
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            image = image.convert('RGB')
            # save image to a temp file with random name
            image_path = "temp_{}.jpg".format(torch.randint(0,1000000,(1,)).item())
            # save image
            image.save(image_path)
            
            # 1st dialogue turn
            query = tokenizer.from_list_format([
                {'image': image_path}, # Either a local path or an url
                {'text': prompt},
            ])
            
            response, history = model.chat(tokenizer, query=query, history=None, use_cache=False)
            
            output = response
            outputs.append(output)
            print(output)

            # delete temp image file
            os.remove(image_path)
            

        return outputs
    

    # %%
    # inference and compute cider scores
    oakvqa_val_inference_outputs = inference(valid_prompt,valid_images)



    # %%
    # compute acc

    def compute_acc(model_output,correct_anwser):
        correct = 0
        for i in range(len(model_output)):
            if correct_anwser[i] in model_output[i]:
                correct += 1
        return correct/len(model_output)
    # %%


    # %%
    acc = compute_acc(oakvqa_val_inference_outputs,valid_anwser_options)

    output_path = pargs.output_path

    with open(output_path,"w") as f:
        # json dumps
        json.dump({"acc":str(acc),"output": oakvqa_val_inference_outputs, "labels":valid_anwser_options},f,indent=4)
