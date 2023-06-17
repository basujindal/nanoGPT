dataset = 'instruct'
init_from = 'eval_llama'

device = 'cuda:1'
data_type = 'instruct'
sample_start = "###User: Write a few words on Einstein.\n###Bot:"

# out_dir = "/home/li/basu_workspace/cptData/out/lb2_llama_dolly_0605-2223"
# learning_block = True

out_dir = "/data1/li/cptData/out/lb2_llama_instruct_0613-1226"
learning_block = True

eval_only = True

batch_size = 1
compile = False

break_at_eos = False
eos_token_id = 2
train_on_user_only = False

calc_perplexity = True