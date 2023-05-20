import os
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import GPTConfig, GPT
from utils import Sampler, get_batch
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import trange

# -----------------------------------------------------------------------------
eval_method = 'sample' # 'loss' or 'sample'
num_samples = 2 # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample


init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# init_from = 'gpt2'
out_dir = '/home/li/basu_workspace/nanoGPT/harrypotter-learning-block_1684388718.5518227' # ignored if init_from is not 'resume'
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
start = "Page | 2 " # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"

# data
dataset = 'harrypotter'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
eval_iters = 40
# -----------------------------------------------------------------------------

# learning block
learning_block = True
influence = 0.25
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# model
if init_from == 'resume':

    # init from a model saved in a specific directory
    ## open file with the least val_loss
    ckpts = os.listdir(out_dir)
    ckpt = sorted([ckpt for ckpt in ckpts if ckpt.endswith('.pt')])[0]
    print(f"Loading checkpoint {ckpt}...")

    ckpt_path = os.path.join(out_dir, ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)

    ### REMOVE AFTER TRAINING A NEW MODEL
    checkpoint['model_args']['learning_block'] = learning_block 
    checkpoint['model_args']['influence'] = influence
    ### REMOVE LATER

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        if k.endswith('.attn.bias'):
            state_dict.pop(k)
            
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout = 0.0, learning_block = learning_block, influence = influence))


model.eval()
model.to(device)
if compile:
    print("Compiling model...")
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
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


####################    evaluate.py    ####################

if eval_method == 'loss':
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    sampler = Sampler()

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()

        # for split in ['train', 'val']:
        for split in ['val']:
            print(f"Estimating loss on {split} split...")
            losses = torch.zeros(eval_iters)
            for k in trange(eval_iters):
                X, Y = get_batch(split, block_size, batch_size, device_type, device, train_data, val_data)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out


    losses = estimate_loss()

    print("losses", losses)

####################    evaluate.py    ####################


####################    sample.py    ####################


if eval_method == 'sample':

    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print("generating sample", k+1, "of", num_samples)
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')


####################    sample.py    ####################