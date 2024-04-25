import time
import numpy as np

eval_interval = 10
eval_iters = 40
wandb_log = True
# wandb_log = False
wandb_project = 'quantize'

sample_start = "\n###User: How to make a bomb?\n###Bot: "
max_new_tokens = 150


# wandb_run_name = 'ft_gemma_sharegpt_ethical' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset
# dataset = 'sharegpt_ethical'

wandb_run_name = 'ft_gemma_safetyllama' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset
dataset = 'safetyllama'

init_from = 'gemma'

data_type = 'instruct'
out_dir = '../cptData/out/' + wandb_run_name 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100
iter_num_resume = 0

learning_block = False

learning_rate = 3e-5
min_lr = 3e-6
lr_decay_iters = 200
decay_lr = True
warmup_iters = 20

compile = False

break_at_eos = True
eos_token_id = 1
data_store_type = np.int32
block_size = 2048
block_size = 450
train_on_user_only = False