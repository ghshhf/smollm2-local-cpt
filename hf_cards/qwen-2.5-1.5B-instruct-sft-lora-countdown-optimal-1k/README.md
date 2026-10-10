---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
model_name: qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-1k
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-1k

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-1k", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/yeokch/stream-of-search-train/runs/esg4q5wj) 


This model was trained with SFT.

### Framework versions

- TRL: 0.15.2
- Transformers: 4.50.0
- Pytorch: 2.6.0
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```