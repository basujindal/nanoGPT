"""
Sample from a trained model
"""

from contextlib import nullcontext
import torch
import pickle
from torch.nn import functional as F
import random
from utils import load_model, get_tokenizer, time_gpu

# -----------------------------------------------------------------------------
init_from = 'llava'
out_dir = "/root/data/cptData/out/llava_0615-0745/ckpt.pt"
start = "###User: Explain the image.\n###Bot:" ## Can also specify a file, use as: "FILE:prompt.txt"
num_samples =  3  # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# learning block
learning_block = True
influence = 0.333

## sampling
break_at_eos = True
eos_token_id = 2
# -----------------------------------------------------------------------------

model_type = 'llava'
sampling = "format" # "discrete" or "continuous" or "format"


exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

torch.set_default_dtype(ptdtype)

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
model, model_args = load_model(model_type, out_dir, device, learning_block, influence, init_from)

model.eval()

    
with time_gpu(device, 'load directly'):
    model = torch.load('model_llava.pt', map_location = device)
    
    
with time_gpu(device, 'load directly'):
    model.load_state_dict(torch.load(out_dir, map_location = device))

encoded = pickle.load(open('../cptData/llava/encoded.pkl', 'rb'))
img_idxs = pickle.load(open('../cptData/llava/img_idxs.pkl', 'rb'))
encoded_imgs = torch.load('../cptData/llava/all_features.pt', map_location = device)


def get_batch(device):
    i = random.randint(0, len(encoded)-1)
    tkns = torch.tensor(encoded[i], device=device).unsqueeze(0)
    return tkns[:, :-1], encoded_imgs[img_idxs[i]].to(device).unsqueeze(0), img_idxs[i]
    

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


# tokenizer
encode, decode = get_tokenizer(model_type)

@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0, top_k=None, break_at_eos=False, eos_token_id=None, img_enc=None):

    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= 2048 else idx[:, -2048:]
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond, img_enc=img_enc)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        if break_at_eos and idx_next.item() == eos_token_id:
            print("breaking at eos")
            break

    
    return idx

def sample(start):
    start_ids = encode(start)
    tkns = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                with time_gpu('Time to generate'):
                    print("Sample", k+1, "------------------------------------")
                    y = generate(tkns, max_new_tokens, temperature=temperature, top_k=top_k,
                                       break_at_eos=break_at_eos, eos_token_id=eos_token_id, 
                                       enc_img = enc_img) 
                                       
                    print(decode(y[0].tolist()))
                    print('---------------')
                  
tkns, enc_img, img_name = get_batch(device)  

print(img_name, decode(tkns[0].tolist()))
                        
if sampling == "discrete":
    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
            sample(start)


elif sampling == "continuous":
    while True:
        ## take input
        print("Enter a sentence to continue:")
        start = str(input())
        sample(start)
        
elif sampling == "format":
    while True:
        ## take input
        print("Enter a sentence to continue:")
        start = str(input())
        start = "###User: " + start + "\n###Bot: "
        sample(start)
    
                    
