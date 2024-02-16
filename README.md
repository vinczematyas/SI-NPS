# NPS_SocialImplicit

1. Download the data from [kaggle](https://www.kaggle.com/datasets/matyasvincze/eth-ucy-pkls) and place it into the folder _eth_ucy_pkls_
2. To train from stratch run | `bash base/run.sh`

# Master Thesis merging [NPS](https://huggingface.co/papers/2103.01937) with [Social Implicit](https://huggingface.co/papers/2203.03057) + some tweaks

- no 2D convolution
- Gumbel-Softmax for module and context selection
- <5k parameters, modular, non-autoregressive, structured sparsity
