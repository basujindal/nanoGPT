import numpy as np
import sys
import os
import json
from tqdm import trange

pth = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(pth)
from llamaTokenizer import LLaMAtokenizer

tokenizer_path = os.path.join(os.path.dirname(pth), "lit-llama/tokenizer.model")

train_frac = 0.9
seq_len = 2048

dataset = 'alpaca'
with open(os.path.join(pth, 'data/alpaca/alpaca_data_cleaned.json')) as f:
    data = json.load(f)

tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
enc = lambda s: tokenizer.encode(s, bos=False, eos=True)
dec = lambda s: tokenizer.decode(s)

data_cleaned  = ["###User: " + instruct['input'] + "\n" + instruct['instruction'] + "\n###Bot: " + instruct['output'] for instruct in data]

encoded = [enc(data_cleaned[i]) for i in trange(len(data_cleaned))]
assert len (encoded) == len(data_cleaned)

encoded = [encoded[i] for i in range(len(encoded)) if len(encoded[i]) < seq_len]

comb = np.ones((len(encoded), seq_len), dtype=np.int32)*2 ## pad with eos_token_id
j, k = 0, 0
sen_lens,l = [], []

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

for i in range(len(comb)):
    if np.all(comb[i] == 2):
        num_sen = i-1
        break

comb = comb[:num_sen]

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
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
inp_shape_train = np.array(inp_shape_train, dtype=np.uint16)
inp_shape_val = np.array(inp_shape_val, dtype=np.uint16)

train_ids.tofile(os.path.join(pth, 'data/{dataset}/train.bin'.format(dataset=dataset)))
val_ids.tofile(os.path.join(pth, 'data/{dataset}/val.bin'.format(dataset=dataset)))
inp_shape_train.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_train.bin'.format(dataset=dataset)))
inp_shape_val.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_val.bin'.format(dataset=dataset)))