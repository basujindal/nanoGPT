eval_interval = 5
eval_iters = 40
wandb_log = True
wandb_project = 'learning-block'
wandb_run_name = 'lb2_llama_instruct' + '_' + time.strftime("%m%d-%H%M") ## train_type,  model , dataset


sample_start = "###User: Write a few words on Einstein.\n\n###Bot:"
max_new_tokens = 100

dataset = 'instruct2'
init_from = 'llama'

data_type = 'instruct'
out_dir = '../cptData/out/' + wandb_run_name 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 7500//batch_size

learning_block = True
device = 'cuda:1'

learning_rate = 3e-4
lr_decay_iters = 300
decay_lr = True
warmup_iters = 20

compile = False

break_at_eos = False
eos_token_id = 2

train_on_user_only = False
