import argparse
import contextlib
import random
import numpy as np
import torch
import sys

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
# Seed random.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

output_len = 100
prompt = "The meaning of life is"

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

print("Model loading done")

def quantize(weights):

    scale = torch.max(torch.abs(weights), dim = 1).values/127
    scaled_weights = torch.div(weights,(scale.unsqueeze(1))).to(torch.float32)
    weights_int = torch.round(scaled_weights).to(torch.int8)

    return weights_int, scale

if __name__ == "__main__":
    
    with torch.no_grad():
        for name_quant, params_quant in model_quant.named_parameters():
            if "weight_scaler" in name_quant:
                print("copying ", name_quant)
                params_quant.copy_(qp[1])
            if "norm" not in name_quant:
                for name, params in model.named_parameters():
                    if (name_quant == name):
                        qp = quantize(params)
                        params_quant.copy_(qp[0])
                        # print("copying ", name_quant)

    print("Saving quantized model")
    torch.save({"model_state_dict":model_quant.state_dict()}, "/root/data/gemma/gemma-2b-quant.ckpt")

    # Samole from quantized model.
    result = model_quant.generate(prompt, device, output_len=output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {prompt}')
    print(f'RESULT: {result}')
    print('======================================')