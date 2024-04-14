import argparse
import contextlib
import random
import numpy as np
import torch
import sys
from tqdm import trange

sys.path.append("/root/data/nanoGPT_LB")
from gemma import gemma_config as config
import gemma.gemma_model_infer as gemma_model
from utils import count_parameters


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

model_config = config.get_model_config("2b")
model_config.dtype = "float16"
model_config.quant = False

seed = 1337
device = "cuda"
ckpt = "/root/data/gemma/gemma-2b.ckpt"
# ckpt = "../cptData/out/ft_gemma_dolly_0414-0328/ckpt.pt"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

output_len = 100
prompt = "\n###User: Write a few words on Einstein.\n###Bot:"

# Create the model and load the weights.
device = torch.device(device)
with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma_model.GemmaForCausalLM(model_config)
    model.load_weights(ckpt, device = device)
print("Model loading done")

model_config = config.get_model_config("2b")
model_config.dtype = "float16"
model_config.quant = True

with _set_default_tensor_type(model_config.get_dtype()):
    model_quant = gemma_model.GemmaForCausalLM(model_config).to(device)
    count_parameters(model_quant, print_table=False)    
    model_quant.load_weights(ckpt, device = device)

print("Model loaded")

def quantize(weights):

    scale = torch.max(torch.abs(weights), dim = 1).values/127
    scaled_weights = torch.div(weights,(scale.unsqueeze(1))).to(torch.float32)
    weights_int = torch.round(scaled_weights).to(torch.int8)

    return weights_int, scale

if __name__ == "__main__":

    with torch.no_grad():

        iter_quant = model_quant.named_parameters()
        iter = model.named_parameters()
    
        num_layers = 0
        for _ in model.named_parameters():
            num_layers+=1

        print("Quantizing model")
        for i in trange(num_layers):
            with torch.no_grad():
                name, params = next(iter)
                name_quant, params_quant = next(iter_quant) 
                
                if "norm" not in name:
                    name_scale, params_scale = next(iter_quant) 
                    # print("quantizing", name)

                    qp = quantize(params)
                    params_quant.copy_(qp[0])                    
                    params_scale.copy_(qp[1])

                iter_quant = model_quant.named_parameters()
        iter = model.named_parameters()

        for i in trange(num_layers):
            with torch.no_grad():
                name, params = next(iter)
                name_quant, params_quant = next(iter_quant) 
                
                if "norm" not in name:
                    name_scale, params_scale = next(iter_quant) 

                    print(name,params_quant, params_scale )

    # print("Saving quantized model")
    # torch.save({"model_state_dict":model_quant.state_dict()}, "/root/data/gemma/gemma-2b-quant.ckpt")

    # # Samole from quantized model.
    # result = model_quant.generate(prompt, device, output_len=output_len)

    # # Print the prompts and results.
    # print('======================================')
    # print(f'PROMPT: {prompt}')
    # print(f'RESULT: {result}')
    # print('======================================')

