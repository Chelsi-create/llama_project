#!/bin/bash

# ✅ Limit to one GPU to avoid tensor parallelism issues with quantized models
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# ✅ Run training on single GPU with LLaMA 3.1 + LoRA
python train_text.py