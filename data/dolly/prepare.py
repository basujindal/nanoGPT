import numpy as np
import sys
import os
import json
sys.path.append('/home/li/basu_workspace/nanoGPT')
from llamaTokenizer import LLaMAtokenizer

tokenizer_path = "/home/li/basu_workspace/llama/tokenizer.model"
train_frac = 0.9
seq_len = 2048


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

comb = np.zeros((len(encoded), seq_len), dtype=np.int32)
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
    if np.sum(comb[i]) == 0:
        num_sen = i-1
        break

comb = comb[:num_sen]

max_len = max([len(i) for i in sen_lens])
## pad sen_lens with zeros
sen_lens = [i + [0]*(max_len - len(i)) for i in sen_lens]
sen_lens = [item for sublist in sen_lens for item in sublist]

train_ids = comb[:int(train_frac*len(encoded))]
val_ids = comb[int(train_frac*len(encoded)):]

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
sen_lens = np.array(sen_lens, dtype=np.uint16)
train_ids.tofile('/home/li/basu_workspace/nanoGPT/data/dolly/train.bin')
val_ids.tofile( '/home/li/basu_workspace/nanoGPT/data/dolly/val.bin')
sen_lens.tofile('/home/li/basu_workspace/nanoGPT/data/dolly/sen_lens.bin')