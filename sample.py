"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from utils import load_model
from llamaTokenizer import LLaMAtokenizer

# -----------------------------------------------------------------------------
init_from = ['resume', 'llama', 'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'][1] # or 'resume' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
out_dir = '/home/li/basu_workspace/nanoGPT/harrypotter-learning-block_1684388718.5518227' # ignored if init_from is not 'resume'
start = "User: Capital of France? \n Bot: Paris \n User: Capital of India \n Bot:"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples =  3  # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# learning block
learning_block = True
influence = 0
# -----------------------------------------------------------------------------

model_type = 'llama' if 'llama' in init_from else 'gpt2'

# sampling = "continuous"
sampling = "discrete"

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
model, model_args = load_model(model_type, out_dir, device, learning_block, influence, init_from)

model.eval()
model.to(device)
print(model)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:

    if model_type == 'gpt2':
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    elif model_type == 'llama':
        tokenizer_path = "/home/li/basu_workspace/llama/tokenizer.model"
        tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
        encode = lambda s: tokenizer.encode(s, bos=True, eos=False)
        print(encode)
        decode = lambda l: tokenizer.decode(l)

if sampling == "discrete":
    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print("generating sample", k+1, "of", num_samples)

                start_ids = encode(start)
                tkns = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                y = model.generate(tkns, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')


if sampling == "continuous":

    while True:
        ## take input
        print("Enter a sentence to continue:")
        start = str(input())
        start_ids = encode(start)
        tkns = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    print("Sample", k+1, "------------------------------------")
                    
                    y = model.generate(tkns, max_new_tokens, temperature=temperature, top_k=top_k)
                    print(decode(y[0].tolist()))
                    print('---------------')
                    
