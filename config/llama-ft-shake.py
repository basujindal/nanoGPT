import time

eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'learning-block'


wandb_run_name = 'ft_llama_shakespeare' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset
dataset = 'llama_shakespeare'
init_from = 'llama'


out_dir = '../cptData/out/' + wandb_run_name 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
learning_block = False

compile = False