---
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets: MelinaLaimon/stream-of-search
library_name: transformers
model_name: Qwen/Qwen2.5-1.5B-Instruct
tags:
- generated_from_trainer
- alignment-handbook
- trl
- sft
licence: license
---

# Model Card for Qwen/Qwen2.5-1.5B-Instruct

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the [MelinaLaimon/stream-of-search](https://huggingface.co/datasets/MelinaLaimon/stream-of-search) dataset.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-1k", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/yeokch/stream-of-search/runs/2z12lxa1) 


This model was trained with SFT.

### Framework versions

- TRL: 0.15.2
- Transformers: 4.50.0
- Pytorch: 2.6.0
- Datasets: 3.4.1
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