from contextlib import nullcontext
import os
import time
import torch
import tiktoken
import inspect
from dataclasses import dataclass
from torch.nn import functional as F

import torch
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# from model import GPTConfig, GPT
from llama.llamaModel import LLaMAConf, LLaMA
from llama.llamaTokenizer import LLaMAtokenizer
from gemma import gemma_config, gemma_model
from gemma.tokenizer import Tokenizer as GemmaTokenizer

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from pynvml import *

import time
from prettytable import PrettyTable

class Timer:
    def __init__(self, start_msg = "", end_msg = ""):
    
        self.start_msg = start_msg
        self.end_msg = end_msg
        
    def __enter__(self):
        if self.start_msg != "":
            print(self.start_msg)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(self.end_msg, f"{elapsed_time:.3f} sec")


def count_parameters(model, print_table = False):
    
    total_params = 0
    
    if(print_table):
        table = PrettyTable(["Modules", "Parameters", "dtype", "Required Grad", "Device"]) 
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        
        if(print_table):
            table.add_row([name, parameter.shape, parameter.dtype, parameter.requires_grad, parameter.device ])
            
        total_params += params
        
    if(print_table):
        print(table)
        
    if total_params/1e9 > 1:
        print(f"Total Trainable Params: {total_params/1e9} B")
    else:
        print(f"Total Trainable Params: {total_params/1e6} M")
        
    return total_params


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

def get_pred_idxs(sen, sep = [13, 29933, 327, 29901]):
    idxs = find_sub_list(sep, sen)

    pred_idxs = []
    for i in range(len(idxs)):
        for j in range(idxs[i][1], len(sen)):
            if sen[j] == 2:
                pred_idxs += [p for p in range(idxs[i][1], j)]     
                break
            
    return pred_idxs


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
        self.encode, self.decode = get_tokenizer(model_name, eos = False)

        self.idxs = []
        start_ids = self.encode(start)
        self.idxs.append(torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        start_ids = self.encode("\n###User: Write a few words on Einstein\n###Bot:")
        self.idxs.append(torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    def generate(self,model, num_samples=2, max_new_tokens=200, temperature=0.7, top_k=200, break_at_eos=True, eos_token_id=None, block_size=2048):
        # run generation
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    # idx = self.idxs
                    for idx in self.idxs:
                        for ii in range(max_new_tokens):
                            # if the sequence context is growing too long we must crop it at block_size
                            idx = idx if idx.size(1) <= block_size else idx[:, -block_size:]

                            mask = torch.tril(torch.ones(idx.shape[0], idx.shape[1],idx.shape[1],dtype=idx.dtype)).to(idx.device)
                            logits = model(idx, mask)
                            logits = logits[:, -1, :] / temperature
                            if top_k is not None:
                                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                                logits[logits < v[:, [-1]]] = -float('Inf')
                            probs = F.softmax(logits, dim=-1)
                            idx_next = torch.multinomial(probs, num_samples=1)
                            idx = torch.cat((idx, idx_next), dim=1)
                            
                            if break_at_eos and idx_next.item() == eos_token_id:
                                print("breaking at eos")
                                break

                        print(self.decode(idx[0].tolist()))
                        print('---------------')


def make_mask(inp, seq_len):
    mask = np.eye((seq_len))
    ptr = 0
    for len in inp:
        if len == 0:
            break
        mask[ptr:ptr+len, ptr:ptr+len] = np.tril(np.ones((len, len)))
        ptr += len
    return mask

def get_batch(split, block_size, batch_size, device_type, device, train_data, val_data, data_type=None, mask_train = None, mask_val = None, train_on_user_only=False):
    
    if data_type == None:
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if 'cuda' in device_type:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y, None, None
    
    elif data_type == 'instruct':

        pred_idxs = None
        data = train_data if split == 'train' else val_data
        mask_data = mask_train if split == 'train' else mask_val
        
        ix = torch.randint(data.shape[0] - 1, (batch_size,))
        
        x = torch.stack([torch.from_numpy((data[i][:-1]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i][1:]).astype(np.int64)) for i in ix])
        mask = torch.stack([torch.from_numpy(make_mask(mask_data[i], block_size)[:-1, :-1]) for i in ix])
        ## conver mask to bool
        mask = mask.bool()
        
        pred_idxs = [get_pred_idxs(xx.tolist()) for xx in x]
        
        if 'cuda' in device_type:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y, mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), mask.pin_memory().to(device, non_blocking=True)
        else:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
        return x, y, mask, pred_idxs

    
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
            dropout=None, meta_vocab_size=None, ckpt_path=None):

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

        elif init_from == 'eval_llama':

            with time_gpu(device, "Loading checkpoint"):
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=device)

            model_args = checkpoint['model_args']
            conf = LLaMAConf(**model_args)

            with time_gpu(device, "Creating model"):
                model = LLaMA(conf)
            
            state_dict = checkpoint['model']
            
            with time_gpu(device, "Loading state dict"):
                model.load_state_dict(state_dict)

        elif init_from == "llama7B":

            ## get current file path
            file_path = os.path.dirname(os.path.realpath(__file__))
            ckpt_path = os.path.join(file_path,"../lit-llama/7B/model.pt")
            
            print(f"Initializing from OG weights: {ckpt_path}")

            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())

            model_args['n_layers'] = params['n_layers']
            model_args['n_heads'] = params['n_heads']
            model_args['n_embd'] = params['dim']
                
            with time_gpu(device, "Loading state dict"):
                model = torch.load(ckpt_path, map_location=device)


        elif init_from.startswith('llama'):

            ## get current file path
            file_path = os.path.dirname(os.path.realpath(__file__))

            ckpt_dir = os.path.join(file_path, "../llama/7B")
            ckpt_path = os.path.join(file_path,"../lit-llama/7B/lit-llama.pth")
            
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
                weights = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(weights, strict=False)

    elif model_type == 'gemma':

        file_path = os.path.dirname(os.path.realpath(__file__))

        if ckpt_path is None:
            ckpt_path = "../gemma/gemma-2b.ckpt"
        
        print(f"Initializing Gemma weights: {ckpt_path}")

        model_args = gemma_config.get_model_config(variant="2b")
        model_args.dtype = "bfloat16"
        # model_config.quant = args.quant

        with time_gpu(device, "Creating model"):
            model = gemma_model.GemmaForCausalLM(model_args)
            model.load_weights(ckpt_path, device=device)
            
        # with time_gpu(device, "Loading state dict"):
        #     model = model.to(device)


    print("Total time to load model: ", time.time() - start_time)
    return model, model_args


def get_tokenizer(model_type, eos = True, bos= False):
    
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
        encode = lambda s: tokenizer.encode(s, bos=eos, eos=bos)
        decode = lambda l: tokenizer.decode(l)

    elif model_type == 'gemma':
        
        tokenizer_path = "../gemma/tokenizer.model"
        tokenizer = GemmaTokenizer(model_path=tokenizer_path)
        encode = lambda s: tokenizer.encode(s, bos=bos, eos=eos)
        decode = lambda l: tokenizer.decode(l)

    return encode, decode

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} 
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