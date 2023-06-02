export WANDB_API_KEY=84742742b66deb0de22b5dfec52ec1f23a539d9b
. activate basu
# python train.py config/wandb_finetune_shakespeare.py 
python all_train.py config/llama-ft-shake.py
