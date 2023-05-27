from contextlib import nullcontext
import os
import torch
import tiktoken
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from model import GPTConfig, GPT
from llamaModel import LLaMAConf, LLaMA

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
# from fairscale.nn.model_parallel.initialize import initialize_model_parallel

# def setup_model_parallel() -> Tuple[int, int]:
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     world_size = int(os.environ.get("WORLD_SIZE", -1))

#     torch.distributed.init_process_group("nccl")
#     initialize_model_parallel(world_size)
#     torch.cuda.set_device(local_rank)

#     # seed must be the same in all processes
#     torch.manual_seed(1)


class Sampler():
    def __init__(self, start="\n", seed = 1337, device='cuda', dtype='bfloat16'):

        """
        Sample from a trained model
        """

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: enc.decode(l)

        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = encode(start)
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
               n_layer=None,n_head=None,n_embd=None,block_size=None,bias=None, dropout=None, meta_vocab_size=None):

    if model_type == 'gpt2':
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
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
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout, learning_block=learning_block, influence=influence)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = getattr(model.config, k)
                
        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            model_args['block_size'] = block_size # so that the checkpoint will have the right value

    elif model_type == 'llama':

        # setup_model_parallel()

        model_args = dict(n_layers=n_layer, n_heads=n_head,learning_block=learning_block, influence=influence,
                        vocab_size = -1, max_seq_len=2048)
                        
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            conf = LLaMAConf(**model_args)
            model = LLaMA(conf)

        elif init_from == 'resume':
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            conf = LLaMAConf(**model_args)
            model = LLaMA(conf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

        elif init_from.startswith('llama'):

            ckpt_dir = "/home/li/basu_workspace/llama/7B"
            ckpt_path = "/home/li/basu_workspace/nanoGPT/lit-llama/7B/lit-llama.pth"
            print(f"Initializing from OG weights: {ckpt_path}")
            # initialize from Meta OG weights
            override_args = dict(dropout=dropout, learning_block=learning_block, influence=influence)

            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())

            model_args['vocab_size'] = 32000
            model_args['n_layers'] = params['n_layers']
            model_args['n_heads'] = params['n_heads']
            model_args['n_embed'] = params['dim']

            print(model_args)

            conf = LLaMAConf(**model_args)
            model = LLaMA(conf)
            weights = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(weights)


    return model, model_args