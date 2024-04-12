export WANDB_API_KEY=
. activate myenv
pip install -U "huggingface_hub[cli]" sentencepiece
python all_train.py config/gemma-ft-dolly.py
