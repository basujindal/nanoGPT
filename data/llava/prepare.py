import numpy as np
import sys
import os
import json
from tqdm import trange
from datasets import load_dataset
import pickle

pth = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname("__file__"))))

print(pth)

sys.path.append(pth)
from llamaTokenizer import LLaMAtokenizer
###################################################################################################                          

## LLaVA-Instruct-150K
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")

data_cleaned = []

for i in trange(len(dataset['train'])):
    con = dataset['train'][i]['conversations']
    con = "###User: " + con[0]['value'] + "\n###Bot:" + con[1]['value']
    con = con.replace('<image>\n', '')
    data_cleaned.append(con)

tokenizer_path = os.path.join(os.path.dirname(pth), "cptData/lit-llama/tokenizer.model")

train_frac = 0.9
seq_len = 2048

tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
enc = lambda s: tokenizer.encode(s, bos=False, eos=True)
dec = lambda s: tokenizer.decode(s)

encoded = [enc(data_cleaned[i]) for i in trange(len(data_cleaned))]
assert len (encoded) == len(data_cleaned)

encoded = [encoded[i] for i in range(len(encoded)) if len(encoded[i]) < seq_len]

# save encoded as pickle
with open(os.path.join(pth, '../cptData/llava/encoded.pkl'), 'wb') as f:
    pickle.dump(encoded, f)
    
    
pth = '../cptData/llava/train2014'
dir_imgs = os.listdir(pth)

img_idxs = []
for i in trange(len(dataset['train'])):
    img = 'COCO_train2014_' + dataset['train'][i]['image']
    img_idxs.append(dir_imgs.index(img))

assert len(img_idxs) == len(dataset['train'])
    
pickle.dump(img_idxs, open('../cptData/llava/img_idxs.pkl', 'wb'))

