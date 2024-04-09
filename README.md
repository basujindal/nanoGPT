# Control Pretrained Transformers

## Push changes

`chmod 600 ~/.ssh/id_rsa_basu`

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_basu
```

## Train

```bash
cd ~/basu_workspace/nanoGPT
export WANDB_API_KEY=
. activate myenv
pip install sentencepiece
python all_train.py config/llama-lb-instruct.py
```

## Datasets:

### Harry potter books:

wget blob:https://download-directory.github.io/4f3436ac-7be1-479c-afdb-9a9888857520

## Download data

### cleaned Alpaca

`wget https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json`

## Prepare dataset

```
python python data/{dataset name}/prepare.py

python data/dolly/prepare.py
```


## Ideas

- [ ] Try to various influence values (0.1 - 0.9)
- [ ] Try with K, Q, V
- [ ] Use same network for K or Q or V or any combination