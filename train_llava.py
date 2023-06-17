import os
import time
import math
from pathlib import Path
import json
import torch
import pickle
from contextlib import nullcontext
import numpy as np
from tqdm import trange
import torch
import random
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from utils import load_model, Sampler, get_batch, configure_optimizers, time_gpu, get_pred_idxs

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
sample_start = "The king exclaimed thou"
max_new_tokens=100
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # or 'resume' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl' or 'eval_llama' or 'llama'
# wandb logging
wandb_log = False # disabled by default
wandb_project = "transformers"
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'llava'
gradient_accumulation_steps = 32 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 2048
# model
n_layers = 12
n_heads = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
seed_offset = 0
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
torch.set_default_dtype(ptdtype)

# learning block
learning_block = False
influence = 0.5
## instruct
data_type = None
break_at_eos=False
eos_token_id=2
train_on_user_only = False

import time
eval_interval = 5
eval_iters = 40
wandb_log = True
wandb_project = 'learning-block'
wandb_run_name = 'llava' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset


sample_start = "###User: Explain the image.\n###Bot:"
max_new_tokens = 100

dataset = 'llava'
init_from = 'llava'

data_type = 'llava'
out_dir = '../cptData/out/' + wandb_run_name 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 500//batch_size

learning_block = True
device = 'cuda'

learning_rate = 3e-4
lr_decay_iters = 300
decay_lr = True
warmup_iters = 20

compile = False

break_at_eos = False
eos_token_id = 2

train_on_user_only = False



# -----------------------------------------------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

model_type = 'llava'


print("Using Learning Block", learning_block)
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

## load encoded from pickle

encoded = pickle.load(open('../cptData/llava/encoded.pkl', 'rb'))
img_idxs = pickle.load(open('../cptData/llava/img_idxs.pkl', 'rb'))
encoded_imgs = torch.load('../cptData/llava/all_features.pt', map_location = device)

train_data = encoded[:int(len(encoded)*0.9)]
val_data = encoded[int(len(encoded)*0.9):]
train_idxs = img_idxs[:int(len(img_idxs)*0.9)]
val_idxs = img_idxs[int(len(img_idxs)*0.9):]


def get_batch(split, device):
    if split == 'train':
        i = random.randint(0, len(train_data)-1)
        tkns = torch.tensor(train_data[i], device=device).unsqueeze(0)
        
        return tkns[:, :-1], tkns[:, 1:], encoded_imgs[train_idxs[i]].to(device).unsqueeze(0)
    else:
        i = random.randint(0, len(val_data)-1)
        tkns = torch.tensor(val_data[i], device=device).unsqueeze(0)
        return tkns[:, :-1], tkns[:, 1:],  encoded_imgs[val_idxs[i]].to(device).unsqueeze(0)

from llava import MultiLLaMA, LLaMAConf

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

## get current file path
file_path = os.path.dirname(os.path.realpath("__file__"))

ckpt_dir = os.path.join(file_path, "../llama/7B")
ckpt_path = os.path.join(file_path,"../cptData/lit-llama/7B/lit-llama.pth")


# print(f"Initializing from OG weights: {ckpt_path}")
# with open(Path(ckpt_dir) / "params.json", "r") as f:
#     params = json.loads(f.read())
        
# start_time = time.time()

# model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embd=n_embd, 
#                         learning_block=learning_block, influence=influence) # start with model_args from command line

# model_args['n_layers'] = params['n_layers']
# model_args['n_heads'] = params['n_heads']
# model_args['n_embd'] = params['dim']

# conf = LLaMAConf(**model_args)
# with time_gpu(device, "Creating model"):
#     model = MultiLLaMA(conf)
    
# with time_gpu(device, "Loading state dict"):
#     weights = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(weights, strict=False)


# print("Total time to load model: ", time.time() - start_time)
    
# with time_gpu(device, 'model to GPU'):
#     model.to(device)
# torch.save(model, 'model_llava.pt')

    
with time_gpu(device, 'load directly'):
    model = torch.load('model_llava.pt', map_location = device)


## set requires grad to false for all layers except learning block and img block

if learning_block:
    print("setting requires_grad=False for all layers except learning block")
    for name, param in model.named_parameters():
        if "learning_block" in name:
            pass
        elif "img_" in name:
            pass
        else:
            param.requires_grad = False


## total number of parameters that requires grad
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total params with requires grad", total_params/1e6, "M")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=False)

# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory
# sampler = Sampler(model_name = model_type, start = sample_start, device = device)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    print("Sampling from trained model")
    # sampler.generate(model, max_new_tokens=max_new_tokens, break_at_eos = break_at_eos,eos_token_id = eos_token_id) 

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # X, Y, mask, pred_idxs = get_batch(split, block_size, batch_size, device_type, device, train_data, val_data, 
            #                        data_type = data_type, mask_train = mask_train, mask_val = mask_val, train_on_user_only = train_on_user_only)
            
            X, Y, enc = get_batch(split, device)                
            with ctx:
                logits = model(X, img_enc = enc)

            # perplexity.update(logits, Y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
        # print(f"perplexity on {split} split: {perplexity.compute():.3f}")
        # perplexity.reset()
        
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb_api_key = ""
    wandb.init(project=wandb_project, name=wandb_run_name, entity='basujindal123',config=config)

X, Y, enc = get_batch('train', device)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
iter_num_resume = iter_num
        
        
print("Training")
for iter_num in range(iter_num_resume, max_iters+1):

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        
        with time_gpu(device,'Ealuate'):
            losses = estimate_loss()
            
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir,'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    with time_gpu(device, 'Training Step'):
        
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits = model(X, img_enc = enc)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            X, Y, enc = get_batch('train', device)
            
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1