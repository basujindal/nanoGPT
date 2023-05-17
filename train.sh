export WANDB_API_KEY=84742742b66deb0de22b5dfec52ec1f23a539d9b
. activate basu
pip install tiktoken
pip install torch -U
cd data/nanoGPT
# python train.py config/wandb_finetune_shakespeare.py 
python train.py config/cpt_gpt.py 
