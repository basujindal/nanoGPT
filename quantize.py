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

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    model_quant = gemma_model.GemmaForCausalLM(model_config)
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
                        print("copying ", name_quant)

    torch.save({"model_state_dict":model_quant.state_dict()}, "/root/data/gemma/gemma-2b-quant2.ckpt")

