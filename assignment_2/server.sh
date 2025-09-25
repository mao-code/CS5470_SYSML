#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 VLLM_LOGGING_CONFIG_PATH=./logging_config.json python3 -m vllm.entrypoints.openai.api_server --model meta-llama/CodeLlama-34b-hf --swap-space 16 --disable-log-requests --enforce-eager --max-num-seqs 512 --disable-sliding-window  --load-format dummy --preemption-mode swap --tensor-parallel-size 2 --max_model_len 16368