import time
eval_interval = 5
eval_iters = 40
wandb_log = True
wandb_project = 'learning-block'
wandb_run_name = 'llava' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset


sample_start = "###User: Explain the image.\n###Bot:"
max_new_tokens = 100

dataset = 'llava'
init_from = 'llava'

data_type = 'llava'
out_dir = '../cptData/out/' + wandb_run_name 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 500//batch_size

learning_block = True
device = 'cuda'

learning_rate = 3e-4
lr_decay_iters = 300
decay_lr = True
warmup_iters = 20

compile = False

break_at_eos = False
eos_token_id = 2

train_on_user_only = False
