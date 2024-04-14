import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils import load_model, Sampler, get_batch, configure_optimizers, time_gpu, get_pred_idxs

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
sample_start = "The king exclaimed thou"
max_new_tokens=100
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # or 'resume' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl' or 'eval_llama' or 'llama'
# wandb logging
wandb_log = False # disabled by default
wandb_project = "transformers"
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
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
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
torch.set_default_dtype(ptdtype)
data_store_type = np.uint16

# learning block
learning_block = False
influence = 0.5

## instruct
data_type = None
break_at_eos=True
eos_token_id=1
train_on_user_only = False

## eval
eval_only = False # if True, script exits right after the first eval
calc_perplexity = False # if True, calculate perplexity
num_samples = 1

# -----------------------------------------------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if 'llama' in init_from:    
    model_type = 'llama'
elif init_from == "gemma":
    model_type = "gemma"
else:
    model_type = "gpt2"

## if torch version < 2 set compile to False
if torch.__version__[0] == '1' and compile:
    print("PyTorch version < 2.0, disabling compilation")
    compile = False

print("Using Learning Block", learning_block)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train_gemma.bin'), dtype=data_store_type, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val_gemma.bin'), dtype=data_store_type, mode='r')

print("Iterations per epoch:", train_data.shape[0] // tokens_per_iter)

print("Train data shape:", train_data.shape, "Val data shape:", val_data.shape)

if data_type == 'instruct':
    mask_train = np.memmap(os.path.join(data_dir, 'inp_shape_train_gemma.bin'), dtype=data_store_type, mode='r')
    mask_val = np.memmap(os.path.join(data_dir, 'inp_shape_val_gemma.bin'), dtype=data_store_type, mode='r')
    train_data = train_data.reshape(-1, block_size)
    val_data = val_data.reshape(-1, block_size)    
    mask_train = mask_train.reshape(train_data.shape[0], -1)
    mask_val = mask_val.reshape(val_data.shape[0], -1)

    print("Train data shape:", train_data.shape, "Mask train:", mask_train.shape, "Val data shape:", val_data.shape, "Mask val:", mask_val.shape)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, learning_block=learning_block) # start with model_args from command line


model, model_args = load_model(model_type, out_dir, device, learning_block, influence, init_from)

with time_gpu(device, 'model to GPU'):
    model.to(device)

## set requires grad to false for all layers except learning block
if learning_block:
    print("setting requires_grad=False for all layers except learning block")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "learning_block" not in name:
                param.requires_grad = False


## total number of parameters that requires grad
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total params with requires grad", total_params/1e6, "M")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    with time_gpu(device, 'compiling model'):
        model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

sampler = Sampler(model_name = model_type, start = sample_start, device = device)

if calc_perplexity:
    from torcheval.metrics.text import Perplexity
    perplexity = Perplexity()
    perplexity.to(device)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    print("Sampling from model")
    sampler.generate(model, max_new_tokens=max_new_tokens, break_at_eos = break_at_eos,eos_token_id = eos_token_id, num_samples = num_samples)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask, pred_idxs = get_batch(split, block_size, batch_size, device_type, device, train_data, val_data, 
                                   data_type = data_type, mask_train = mask_train, mask_val = mask_val, train_on_user_only = train_on_user_only)
                
            with ctx:
                logits = model(X, mask = mask)
                                
            if train_on_user_only:
                logits = logits.gather(1, torch.tensor(pred_idxs, device=device).unsqueeze(2).repeat(1,1,logits.size(-1))).squeeze(2)
                Y = Y.gather(1, torch.tensor(pred_idxs, device=device)).squeeze(1)

            if calc_perplexity:
                perplexity.update(logits, Y)
                
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
        if calc_perplexity:
            print(f"perplexity on {split} split: {perplexity.compute():.3f}")
            perplexity.reset()
        
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
if wandb_log and master_process:
    import wandb
    wandb_api_key = ""
    wandb.init(project=wandb_project, name=wandb_run_name, entity='basujindal123',config=config)

# training loop
X, Y, mask, pred_idxs = get_batch('train', block_size, batch_size, device_type, device, train_data, val_data,
                       data_type = data_type, mask_train = mask_train, mask_val = mask_val, train_on_user_only = train_on_user_only)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
iter_num_resume = iter_num
        
        
print("Training")
for iter_num in range(iter_num_resume, max_iters+1):

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        model.eval()
        # print("Sampling from model")
        # sampler.generate(model, max_new_tokens=max_new_tokens, break_at_eos = break_at_eos,eos_token_id = eos_token_id)
        # model.train() 
        
        with time_gpu(device,'Ealuate'):
            losses = estimate_loss()

        # losses = {"train":0, "val":0}
            
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
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
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits = model(X, mask = mask)
                
            if train_on_user_only:
                logits = logits.gather(1, torch.tensor(pred_idxs, device=device).unsqueeze(2).repeat(1,1,logits.size(-1))).squeeze(2)
                Y = Y.gather(1, torch.tensor(pred_idxs, device=device)).squeeze(1)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, mask, pred_idxs = get_batch('train', block_size, batch_size, device_type, device, train_data, val_data,
                                   data_type = data_type, mask_train = mask_train, mask_val = mask_val, train_on_user_only = train_on_user_only)
            
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
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()

