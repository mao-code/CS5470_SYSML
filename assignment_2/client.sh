#!/bin/bash

python3 benchmark.py --backend openai --model meta-llama/CodeLlama-34b-hf --request-rate 3 --num-prompts 30 --dataset-name dummy --long-prompts 0 --long-prompt-len 32000