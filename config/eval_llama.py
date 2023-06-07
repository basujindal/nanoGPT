dataset = 'dolly'
init_from = 'eval_llama'
learning_block = True


device = 'cuda:1'
data_type = 'instruct'
out_dir = "/home/li/basu_workspace/cptData/out/lb2_llama_dolly_0605-2223/ckpt.pt"


batch_size = 1
compile = False


break_at_eos = False
eos_token_id = 2
train_on_user_only = False
