import numpy as np
import sys
import os
from datasets import load_dataset
from tqdm import trange
import json

pth = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(pth)
sys.path.append('/root/data/nanoGPT_LB/gemma/')

from utils import get_tokenizer

tokenizer_path = os.path.join(os.path.dirname(pth), "gemma/tokenizer.model")
eos_token_id = 1
pad_token_id = 0

train_frac = 0.9
seq_len = 2048
dataset = 'shareGPT'

enc, dec = get_tokenizer("gemma")

with open(os.path.join(pth, 'data/shareGPT/sharegpt.json')) as f:
    data = f.readlines()
    
    
data_cleaned = [json.loads(line) for line in data][0]
data_cleaned = [data_cleaned[i][str(i)] for i in range(len(data_cleaned))]

# data_cleaned = [json.loads(line) for line in data]
# data_cleaned  = ["\n###User: " + instruct['instruction'] + "\n###Bot: " + instruct['output'] for instruct in data]

encoded = []
for sentence in data_cleaned:
    encoded.append(enc(sentence))
assert len (encoded) == len(data_cleaned)

encoded = [encoded[i][:seq_len] for i in range(len(encoded))]
comb = np.ones((len(encoded), seq_len), dtype=np.int32)*pad_token_id ## pad with pad_token_id

j, k = 0, 0
sen_lens, l = [], []

for i in encoded:
    if k + len(i) > seq_len:
        sen_lens.append(l)
        l = [len(i)]
        j+=1
        k=len(i)
    else:
        k+=len(i)
        l.append(len(i))
    comb[j, k-len(i):k] = i

for i in trange(len(comb)):
    if np.all(comb[i] == pad_token_id):
        num_sen = i-1
        break

comb = comb[:num_sen]

print(dec(comb[0].tolist()))
max_len = max([len(i) for i in sen_lens])

sen_lens = [i + [0]*(max_len - len(i)) for i in sen_lens]

inp_shape_train = sen_lens[:int(train_frac*len(comb))]
inp_shape_val = sen_lens[int(train_frac*len(comb)):]

train_ids = comb[:int(train_frac*len(comb))]
val_ids = comb[int(train_frac*len(comb)):]

assert train_ids.shape[0] == len(inp_shape_train)
assert val_ids.shape[0] == len(inp_shape_val)

print("train_ids.shape: ", train_ids.shape)
print("val_ids.shape: ", val_ids.shape)

inp_shape_train = [item for sublist in inp_shape_train for item in sublist]
inp_shape_val = [item for sublist in inp_shape_val for item in sublist]

# export to bin files
train_ids = np.array(train_ids, dtype=np.int32)
val_ids = np.array(val_ids, dtype=np.int32)
inp_shape_train = np.array(inp_shape_train, dtype=np.int32)
inp_shape_val = np.array(inp_shape_val, dtype=np.int32)

print(dec(train_ids[0][:100].tolist()))
train_ids.tofile(os.path.join(pth, 'data/{dataset}/train_gemma.bin'.format(dataset=dataset)))
val_ids.tofile( os.path.join(pth, 'data/{dataset}/val_gemma.bin'.format(dataset=dataset)))
inp_shape_train.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_train_gemma.bin'.format(dataset=dataset)))
inp_shape_val.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_val_gemma.bin'.format(dataset=dataset)))