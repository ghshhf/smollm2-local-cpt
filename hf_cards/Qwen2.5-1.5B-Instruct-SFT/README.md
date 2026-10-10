---
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets: sft_verl_0811-1
library_name: transformers
model_name: Qwen2.5-1.5B-Instruct-SFT
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for Qwen2.5-1.5B-Instruct-SFT

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the [sft_verl_0811-1](https://huggingface.co/datasets/sft_verl_0811-1) dataset.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="Bearrr310/Qwen2.5-1.5B-Instruct-SFT", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://app.bandw.top/xinyue-peng-intel/huggingface/runs/2jrr7o1p) 


This model was trained with SFT.

### Framework versions

- TRL: 0.18.2
- Transformers: 4.52.4
- Pytorch: 2.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```