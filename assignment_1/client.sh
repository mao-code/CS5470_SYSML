#!/bin/bash

python3 benchmark.py --backend vllm \
    --model meta-llama/Llama-3.1-8B \
    --request-rate 10 \
    --num-prompts 100 \
    --dataset-name dummy \
    --long-prompts 0 \
    --long-prompt-len 32000