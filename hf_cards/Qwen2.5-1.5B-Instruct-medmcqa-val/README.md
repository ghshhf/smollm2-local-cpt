---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: Qwen2.5-1.5B-Instruct-medmcqa-val
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Qwen2.5-1.5B-Instruct-medmcqa-val

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 3

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 4.56.0
- Pytorch 2.10.0+cu130
- Datasets 4.5.0
- Tokenizers 0.22.2