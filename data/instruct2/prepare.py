import numpy as np
import sys
import os
import json
from tqdm import trange

pth = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

print(pth)
###################################################################################################                          

## alpaca cleaned
with open(os.path.join(pth, 'data/alpaca/alpaca_data_cleaned.json')) as f:
    data = json.load(f)

data_cleaned = []

for instruct in data:
    if instruct['input'] == "":
        data_cleaned.append("###User: " + instruct['instruction'] + "\n###Bot: " + instruct['output'])
    else:
        data_cleaned.append("###User: " + instruct['input'] + " " + instruct['instruction'] + "\n###Bot: " + instruct['output'])

print("Number of examples in alpaca: ", len(data_cleaned))

## dolly
with open(os.path.join(pth, 'data/dolly/databricks-dolly-15k.jsonl')) as f:
    data = f.readlines()
    
data = [json.loads(line) for line in data]
data_cleaned_dolly = []

for instruct in data:
    if instruct['context'] == "":
        data_cleaned_dolly.append("###User: " + instruct['instruction'] + "\n###Bot: " + instruct['response'])
    else:
        data_cleaned_dolly.append("###User: " + instruct['context'] + " " + instruct['instruction'] + "\n###Bot: " + instruct['response'])
print("Number of examples in dolly: ", len(data_cleaned_dolly))

data_cleaned += data_cleaned_dolly

# ## gpt4all
# from datasets import load_dataset

# data = load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision='v1.3-groovy')
# data_cleaned_gpt4all  = ["###User: " + instruct['prompt'] + "\n###Bot: " + instruct['response'] for instruct in data['train']]

# print("Number of examples in gpt4all: ", len(data_cleaned_gpt4all))

# data_cleaned += data_cleaned_gpt4all

# print("Total number of examples: ", len(data_cleaned))
############################################################################################################

# ## save data_cleaned into a file
# with open(os.path.join(pth, 'data/instruct2/data_cleaned.json'), 'w') as f:
#     # convert into JSON:
#     dc_json = json.dumps(data_cleaned)
#     json.dump(dc_json, f)


sys.path.append(pth)
from llamaTokenizer import LLaMAtokenizer

tokenizer_path = os.path.join(os.path.dirname(pth), "lit-llama/tokenizer.model")

train_frac = 0.9
seq_len = 2048
dataset = 'instruct2'

tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
enc = lambda s: tokenizer.encode(s, bos=False, eos=True)
dec = lambda s: tokenizer.decode(s)

encoded = [enc(data_cleaned[i]) for i in trange(len(data_cleaned))]
assert len (encoded) == len(data_cleaned)

encoded = [encoded[i] for i in range(len(encoded)) if len(encoded[i]) < seq_len]

## shuffle encoded
np.random.seed(42)
np.random.shuffle(encoded)

totalTokens=0
for i in encoded:
    totalTokens+=len(i)
print("totalTokens: {:.2f}".format(totalTokens/1e6), "M")


num_train = totalTokens//seq_len + 1

### converting 1d tokens to 2d batch of tokens 
comb = np.ones((num_train, seq_len), dtype=np.int32)*2 ## pad with eos_token_id

j, k = 0, 0

for i in trange(len(encoded)):
    sen = encoded[i]
    
    if k + len(sen) > seq_len:

        remlen = seq_len - k
        comb[j, -remlen:] = sen[:remlen]
        k = len(sen) - remlen
        l = [k]
        j+=1
        comb[j, :k] = sen[-k:]
    else:
        comb[j, k:len(sen)+k] = sen
        k+=len(sen)
    if k == seq_len:
        k = 0
        j+=1
        
sen_lens = []
for i in comb:
    idx_old = 0
    l = []
    for idx, j in enumerate(i):
        if j == 2:
            l.append(idx-idx_old + 1)
            idx_old = idx + 1
    l.append(len(i)-idx_old)
    sen_lens.append(l)


### check if all the tokens are accounted for (may remove this later)
all_lens = []
for i in sen_lens:
    all_lens+=i

j = 0
for i in trange(len(encoded)):
    if len(encoded[i]) == all_lens[j]:
        j+=1
    else:
        if len(encoded[i]) != all_lens[j] + all_lens[j+1]:
            assert False
        else:
            j+=2
            
all_lens = all_lens[:j]
            

ans = 0
for i in trange(len(encoded)):
    ans += len(encoded[i])
for j in all_lens:
    ans-=j
assert ans == 0


### remove empty sentences lengths from last batch
for i in trange(len(sen_lens[-1])):
    if sen_lens[-1][i] == 1:
        ans = i
        break
sen_lens[-1] = sen_lens[-1][:ans]

sen_lens = [i + [0]*(seq_len - len(i)) for i in sen_lens]

train_lens = sen_lens[:int(train_frac*len(comb))]
val_lens = sen_lens[int(train_frac*len(comb)):]

train_ids = comb[:int(train_frac*len(comb))]
val_ids = comb[int(train_frac*len(comb)):]

assert train_ids.shape[0] == len(train_lens) and val_ids.shape[0] == len(val_lens)

print("train_ids.shape: ", train_ids.shape)
print("val_ids.shape: ", val_ids.shape)

## flatten the seq lengths although not necessary as tofile flattens the array
train_lens = [item for sublist in train_lens for item in sublist]
val_lens = [item for sublist in val_lens for item in sublist]

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_lens = np.array(train_lens, dtype=np.uint16)
val_lens = np.array(val_lens, dtype=np.uint16)

train_ids.tofile(os.path.join(pth, 'data/{dataset}/train.bin'.format(dataset=dataset)))
val_ids.tofile(os.path.join(pth, 'data/{dataset}/val.bin'.format(dataset=dataset)))
train_lens.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_train.bin'.format(dataset=dataset)))
val_lens.tofile(os.path.join(pth, 'data/{dataset}/inp_shape_val.bin'.format(dataset=dataset)))