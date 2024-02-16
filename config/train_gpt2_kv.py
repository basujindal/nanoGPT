# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'K=V'
wandb_run_name='k_v_equal'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 40 gradaccum = 655360
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 40

# this makes total number of tokens be ~ 655M
max_iters = 1000
lr_decay_iters = 1000

# eval stuff
eval_interval = 10
eval_iters = 40
log_interval = 5

# weight decay
weight_decay = 1e-1
