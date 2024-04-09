"""
Sample from a trained model
"""

from contextlib import nullcontext
import torch
from utils import load_model, get_tokenizer, print_gpu_utilization, time_gpu

# -----------------------------------------------------------------------------
init_from = ['resume', 'eval_llama', 'llama', 'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'][1] # or 'resume' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
out_dir = "/data1/out/lb2_llama_instruct_0613-1226"
start = "###User: Write a few words on Einstein.\n###Bot:" ## Can also specify a file, use as: "FILE:prompt.txt"
num_samples =  3  # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# learning block
learning_block = True
influence = 0.5

## sampling
break_at_eos = True
eos_token_id = 2
# -----------------------------------------------------------------------------

model_type = 'llama' if 'llama' in init_from else 'gpt2'
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

with time_gpu('move model to device'):
    model.to(device)

print_gpu_utilization()

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


# tokenizer
encode, decode = get_tokenizer(model_type)

def sample(start):
    start_ids = encode(start)
    tkns = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                with time_gpu('Time to generate'):
                    print("Sample", k+1, "------------------------------------")
                    y = model.generate(tkns, max_new_tokens, temperature=temperature, top_k=top_k,
                                       break_at_eos=break_at_eos, eos_token_id=eos_token_id)
                                       
                    print(decode(y[0].tolist()))
                    print('---------------')
                        
                        
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
    
                    
