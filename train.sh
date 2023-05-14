. activate myenv
pip install tiktoken
pip install torch -U
cd data/nanoGPT
# python train.py config/wandb_finetune_shakespeare.py 
python train.py config/cpt_gpt_shakespeare.py 
