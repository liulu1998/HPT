#!/bin/bash
export TRANSFORMERS_CACHE="/home/liulu/data/.cache/huggingface"
python train.py --data yixinli --name testSoftPrompt --batch 16
