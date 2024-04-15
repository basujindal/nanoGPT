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


def main(args):
    # Construct the model config.

    
    print(args)
    dtype = args.dtype
    model_config = config.get_model_config(args.variant)
    model_config.dtype = dtype
    if args.device == "cpu":
        model_config.dtype = "float32" 
    model_config.quant = args.quant

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config).to(device)
        model.load_weights(args.ckpt, device = args.device)
        count_parameters(model, print_table=True)
        model = model.to(device).eval()
    print("Model loading done")

    # Generate the response.
    result = model.generate(args.prompt, device, output_len=args.output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {args.prompt}')
    print(f'RESULT: {result}')
    print('======================================')

    prompt = "\n###User: Write a few words on Einstein.\n###Bot:"
    result = model.generate(prompt, device, output_len=args.output_len)

    print('======================================')
    print(f'PROMPT: {prompt}')
    print(f'RESULT: {result}')
    print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    main(args)
