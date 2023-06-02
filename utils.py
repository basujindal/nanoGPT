from contextlib import nullcontext
import os
import time
import torch
import tiktoken
import inspect
from dataclasses import dataclass

import torch
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from model import GPTConfig, GPT
from llamaModel import LLaMAConf, LLaMA
from llamaTokenizer import LLaMAtokenizer

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used/1024**3


class time_gpu():
    def __init__(self, device, statement="Time taken: "):
        self.statement = statement
        
        self.device = 0
        if device == 'cpu':
            self.device = 'cpu'
        elif ':' in device:
            self.device = int(device.split(':')[-1]) 
        
    def __enter__(self):
        self.t = time.time()

        if self.device != 'cpu':
        
            self.gpu = print_gpu_utilization()
            torch.cuda.max_memory_allocated(self.device)
            

    def __exit__(self, exc_type, exc_value, traceback):
        
        if self.device == 'cpu':
            print(self.statement, time.time() - self.t)
            return
        
        print(self.statement, time.time() - self.t)
        print("GPU memory used:", print_gpu_utilization() - self.gpu)
        t = torch.cuda.get_device_properties(self.device).total_memory
        r = torch.cuda.memory_reserved(self.device)
        a = torch.cuda.memory_allocated(self.device)
        max_mem = torch.cuda.max_memory_allocated(self.device)
        
        max_spaces = 10
        print_space = lambda x: ((max_spaces - len(x))//2)*" " +  x + ((max_spaces - len(x))//2)*" "
        
        print("|".join([print_space(i) for i in ["Total", "Reserved", "Allocated", "Max"]]))
        print("|".join([ print_space(i) for i in [str(round(x/1024**3, 2)) + 'GB' for x in [t, r, a, max_mem]]]))  

class Sampler():
    def __init__(self, model_name = "gpt2", start="\n", seed = 1337, device='cuda', dtype='bfloat16'):

        """
        Sample from a trained model
        """

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        ## tokenizer
        self.encode, self.decode = get_tokenizer(model_name)

        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = self.encode(start)
        self.x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    def generate(self,model, num_samples=2, max_new_tokens=200, temperature=0.7, top_k=200):
        # run generation
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    y = model.generate(self.x, max_new_tokens, temperature=temperature, top_k=top_k)
                    print(self.decode(y[0].tolist()))
                    print('---------------')


def get_batch(split, block_size, batch_size, device_type, device, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_time_str():
    now = datetime.now()
    return now.strftime("%d/%m/%Y/%H:%M")

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def load_model(model_type, out_dir, device, learning_block, influence, init_from,
               n_layers=None,n_heads=None,n_embd=None,block_size=None,bias=None,
            dropout=None, meta_vocab_size=None):

    print("Creating and loading model")
    start_time = time.time()

    if model_type == 'gpt2':
        model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout, learning_block=learning_block) # start with model_args from command line


        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

        elif init_from == 'resume':
            
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layers', 'n_heads', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)


        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout, learning_block=learning_block, influence=influence)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layers', 'n_heads', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = getattr(model.config, k)
                
        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            model_args['block_size'] = block_size # so that the checkpoint will have the right value

    elif model_type == 'llama':
        
        model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embd=n_embd, 
                        learning_block=learning_block, influence=influence) # start with model_args from command line
                        
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            conf = LLaMAConf(**model_args)
            
            with time_gpu(device, "Creating model"):
                model = LLaMA(conf)

        elif init_from == 'resume_llama':

            with time_gpu(device, "Loading checkpoint"):
                checkpoint = torch.load(out_dir, map_location=device)

            model_args = checkpoint['model_args']
            conf = LLaMAConf(**model_args)

            with time_gpu(device, "Creating model"):
                model = LLaMA(conf)
            
            state_dict = checkpoint['model']
            
            with time_gpu(device, "Loading state dict"):
                model.load_state_dict(state_dict)

        elif init_from.startswith('llama'):

            ## get current file path
            file_path = os.path.dirname(os.path.realpath(__file__))

            ckpt_dir = os.path.join(file_path, "../llama/7B")
            ckpt_path = os.path.join(file_path,"../cptData/lit-llama/7B/lit-llama.pth")
            
            print(f"Initializing from OG weights: {ckpt_path}")

            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())

            model_args['n_layers'] = params['n_layers']
            model_args['n_heads'] = params['n_heads']
            model_args['n_embd'] = params['dim']
            
            conf = LLaMAConf(**model_args)
            with time_gpu(device, "Creating model"):
                model = LLaMA(conf)
                
            with time_gpu(device, "Loading state dict"):
                weights = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(weights, strict=False)


    print("Time to load model: ", time.time() - start_time)
    return model, model_args


def get_tokenizer(model_type):
    
    if model_type == 'gpt2':
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    elif model_type == 'llama':

        file_path = os.path.dirname(os.path.realpath(__file__))
        tokenizer_path = os.path.join(file_path, "../llama/tokenizer.model")
        tokenizer = LLaMAtokenizer(model_path=tokenizer_path)
        encode = lambda s: tokenizer.encode(s, bos=True, eos=False)
        decode = lambda l: tokenizer.decode(l)

    return encode, decode

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    print(param_dict.keys())    
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer