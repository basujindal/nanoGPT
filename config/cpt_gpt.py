import time

learning_block = True
out_dir = 'shakespeare-learning-block_' + str(time.time())
eval_interval = 5 
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + out_dir + '-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-large' # this is the second largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 2
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
wandb_log = False
