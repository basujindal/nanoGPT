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

def quantize(weights):

    scale = torch.max(torch.abs(weights), dim = 1).values/127
    scaled_weights = torch.div(weights,(scale.unsqueeze(1))).to(torch.float32)
    weights_int = torch.round(scaled_weights).to(torch.int8)

    return weights_int, scale
       
def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = "bfloat16"

    model_config = config.get_model_config(args.variant)
    model_config.dtype = dtype
    model_config.quant = False

    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(args.ckpt, device = device)

    count_parameters(model, 1)

    model_config = config.get_model_config("2b")
    model_config.dtype = dtype
    model_config.quant = True

    with _set_default_tensor_type(model_config.get_dtype()):
        model_quant = gemma_model.GemmaForCausalLM(model_config).to(device)
        count_parameters(model_quant, print_table=False)    

    print("Model loaded")

    with torch.no_grad():

        num_layers = 0
        for _ in model.named_parameters():
            num_layers+=1

        iter_quant = model_quant.named_parameters()
        iter = model.named_parameters()

        print("Quantizing model")
        for i in trange(num_layers):
            with torch.no_grad():
                name, params = next(iter)
                name_quant, params_quant = next(iter_quant) 
                
                if "norm" not in name:
                    name_scale, params_scale = next(iter_quant) 
                    qp = quantize(params)
                    params_quant.copy_(qp[0])                    
                    params_scale.copy_(qp[1])
                else:
                    params_quant.copy_(params)

        print("Sampling from quantized model")
        result = model_quant.generate(args.prompt, device, output_len=args.output_len)

        print('======================================')
        print(f'PROMPT: {args.prompt}')
        print(f'RESULT: {result}')
        print('======================================')

        prompt = "\n###User: Write a few words on Einstein.\n###Bot:"
        result = model_quant.generate(prompt, device, output_len=args.output_len)

        print('======================================')
        print(f'PROMPT: {prompt}')
        print(f'RESULT: {result}')
        print('======================================')

    print("Saving quantized model")
    # torch.save({"model_state_dict":model_quant.state_dict()}, "/root/data/gemma/gemma-2b-quant-" + args.ckpt.split('/')[-2] + ".ckpt")
    torch.save({"model_state_dict":model_quant.state_dict()}, "/root/data/gemma/gemma-2b-quant.ckpt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/root/data/gemma/gemma-2b.ckpt")
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    args = parser.parse_args()

    main(args)