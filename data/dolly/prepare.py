import numpy as np
import sys
import os
import json

pth = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(pth)
from llamaTokenizer import LLaMAtokenizer
tokenizer_path = os.path.join(os.path.dirname(pth), "cptData/llama/tokenizer.model")
train_frac = 0.9
seq_len = 2048.


tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
enc = lambda s: tokenizer.encode(s, bos=False, eos=True)
dec = lambda s: tokenizer.decode(s)

with open('/home/li/basu_workspace/nanoGPT/data/dolly/databricks-dolly-15k.jsonl') as f:
    data = f.readlines()
    
data = [json.loads(line) for line in data]
data_cleaned  = ["User: " + instruct['instruction'] + "\nBot: " + instruct['response'] for instruct in data]

encoded = []
for sentence in data_cleaned:
    encoded.append(enc(sentence))
assert len (encoded) == len(data_cleaned)

encoded = [encoded[i] for i in range(len(encoded)) if len(encoded[i]) < seq_len]

comb = np.ones((len(encoded), seq_len), dtype=np.int32)*2 ## pad with eos_token_id
j = 0
k = 0
sen_lens = []
l = []
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

train_ids.tofile('/home/li/basu_workspace/nanoGPT/data/dolly/train.bin')
val_ids.tofile( '/home/li/basu_workspace/nanoGPT/data/dolly/val.bin')
inp_shape_train.tofile('/home/li/basu_workspace/nanoGPT/data/dolly/inp_shape_train.bin')
inp_shape_val.tofile('/home/li/basu_workspace/nanoGPT/data/dolly/inp_shape_val.bin')