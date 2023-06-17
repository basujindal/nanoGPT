export WANDB_API_KEY=
. activate basu
# python train.py config/wandb_finetune_shakespeare.py 
python all_train.py config/llama-ft-shake.py
