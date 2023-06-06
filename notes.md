## Push changes

`chmod 600 ~/.ssh/id_rsa_basu`

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_basu
```

## Train

```bash
cd ~/basu_workspace/nanoGPT
export WANDB_API_KEY=84742742b66deb0de22b5dfec52ec1f23a539d9b
. activate basu
python all_train.py config/llama-lb-dolly.py
```

`scp -r li@ecepxiegpu1.ucsd.edu:basu_workspace/nanoGPT .`



## Datasets:

### Harry potter books:

wget blob:https://download-directory.github.io/4f3436ac-7be1-479c-afdb-9a9888857520


## Ideas

- [ ] Try to various influence values (0.1 - 0.9)
- [ ] Try with K, Q, V
- [ ] Use same network for K or Q or V or any combination