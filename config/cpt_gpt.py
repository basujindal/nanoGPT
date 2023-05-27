from utils import get_time_str

device = 'cuda:1'

learning_block = True
influence = 0.5

init_from = 'gpt2-xl' # this is the second largest GPT-2 model
dataset = 'llama'
sample_start = "The paper is about llama, a LLM with "
out_dir = dataset + '-learning-block_' + get_time_str()
eval_interval = 5 
eval_iters = 40

## logging
wandb_log = True # feel free to turn on
wandb_project = 'learning-block'
wandb_run_name = out_dir

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4
gradient_accumulation_steps = 32
max_iters = 100

# Decay LR
learning_rate = 3e-4
decay_lr = True
warmup_iters = 1 # how many steps to warm up for
lr_decay_iters = 100 # should be ~= max_iters per Chinchilla
min_lr = 3e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla