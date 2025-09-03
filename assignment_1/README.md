# CS5470 - HW1: vLLM Benchmarking and Profiling
**Due Date:** [9/17/2025] 

Please start this assignment as early as possible!

## Overview

This homework focuses on setting up and benchmarking the vLLM inference server ([GitHub repository](https://github.com/vllm-project/vllm/)). You will:

- Set up the inference server
- Benchmark generation performance
- Understand the breakdown of GPU execution time during LLM inference

You might find this [blog](https://www.aleksagordic.com/blog/vllm) helpful.

## Perequisites

### Hardware Access
You will be provided access to the **Perlmutter HPC** where you can reserve a server with 4 A100 GPUs, each having 40 GB of memory.

### Setup Instructions

#### Step 1: Setup Conda and VLLM

Set up the conda environment and clone the vLLM repository with the specified commit. 

<!-- Install conda if it is not available on the GPU server using the [Linux terminal installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer). -->

**Setup Commands:**
```bash
# On login node, load conda
module load conda
conda create --name sysml python=3.10.12
conda activate sysml

git clone https://github.com/vllm-project/vllm/
cd vllm

# Use the version we are using for this homework
git reset --hard 2f13319f47eb9a78b471c5ced0fcf90862cd16a2

# Install vLLM with precompiled binaries
VLLM_USE_PRECOMPILED=1 python3 -m pip install -e .
```

<!-- If you run into disk quota issue, you could try 

https://docs.nersc.gov/filesystems/perlmutter-scratch/

/pscratch/sd/FirstLetterOfUserName/YourUserName

```
conda create -p /pscratch/sd/e/$USER/sysml python=3.10.12


``` -->
#### Step 2: Huggingface access
1. Create Huggingface account and request for model access.
- `meta-llama/Llama-3.1-8B`
2. Log in your account from command line
   ```bash
   huggingface-cli login
   # Create token through the url and copy-paste
   ```
#### Step 3: Download model and tokenizer
This is required as vLLM could not download models on-the-fly on Perlmutter GPU nodes due to restrictions on large file download.

```
python download.py --model-id meta-llama/Llama-3.1-8B ----cache-dir /pscratch/sd/<first_letter_of_usrname>/<usrname>/huggingface
```

**Note**: Due to disk space limitation on `home` file system, we encourage using `pscratch` for storing model weights. More information could be found [here](https://docs.nersc.gov/filesystems/perlmutter-scratch/).

**Note**: remember to set environment variable `HF_HOME=/pscratch/sd/<first_letter_of_usrname>/<usrname>/huggingface` in your running environment such that HF library could select locally-cached model weights instead of pulling from remote repository.

<!-- TODO Do we need this 

Copy the homework zip file into the vLLM source's root directory and uncompress it. -->

## Homework Tasks

This homework is designed to teach how to serve LLMs on modern GPUs. You will learn how to serve models that fit on a single GPU and how to serve models using tensor parallelism on multiple GPUs.

### Task 1: Single GPU Benchmarking

Within the homework folder, we have provided `server.sh`, a script that starts a `meta-llama/Llama-3.1-8B` instance on GPU 0 and exposes a vLLM API.

**Requirements:**
1. Allocate a node with 1 GPU.
2. Start the serving workload on 1 GPU using the command provided in `server.sh`
3. Execute the benchmark script using the command in `client.sh`
4. Store the TTFT (Time To First Token) and TPOT (Time Per Output Token) for each prompt from the output in a txt file.

**Hint:**
1. Allocate interactive GPU node using `salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account <projectID>`.
2. Remember to make those scripts executable.
3. Spawning two terminals would be useful, one for server and one for client. They should be on the same GPU node. Access the allocated node using `ssh <GPU_node_ID>` if the other terminal is on the login node.
4. You need to login to Huggingface again on GPU nodes, and set `HF_HOME` to `HF_HOME=/pscratch/sd/<first_letter_of_usrname>/<usrname>/huggingface` for both server and client terminal.
5. You might wait for several minutes until the serving engine is up and running.

### Task 2: Multi-GPU Benchmarking

**Requirements:**
1. Serve `meta-llama/Llama-3.1-8B` on 2 GPUs.
2. Serve `meta-llama/Llama-3.1-8B` on 4 GPUs.
3. Modify parameters in `server.sh` to execute these configurations.
   - Review the vLLM documentation to understand what CLI arguments you need to add or modify for multi-GPU inference
4. Store the TTFT and TPOT metrics for both configurations

### Task 3: Performance Visualization

Create two plots showing:
1. Sorted TTFTs of prompts in the benchmark from all three configurations: serving with 1, 2, and 4 GPUs.
2. Sorted TPOTs of prompts in the benchmark from all three configurations

X-axis is the prompt index (not the original index of prompt) and the y-axis is TTFT or TTFT, so the plots will be monotonically increasing. Add labels to indicate the configuration.

### Task 4: NVIDIA Nsight Profiling
**Requirements:**
1. Install NVIDIA Nsight Systems if not available from the [official website](https://developer.nvidia.com/nsight-systems/get-started)
   - Use the CLI-only deb installer for Linux
   - The current version on Perlmutter may not work for us.
2. Generate an Nsight trace using the `nsys` profiler for Llama-3.1-8B on 2 GPUs
3. Open the generated `.nsys-rep` file in Nsight desktop client on your laptop
4. Use the Stats System View to identify the top 3 kernels that consumed the most time
5. Identify one AllReduce kernel and take a screenshot.

**Hint**
1. Use interactive profiling mode.
2. Store the `nsys-rep` file under `pscratch`.

## Deliverables

### Submission Requirements

Submit the following components in a zip folder to Canvas: <netid\>_a1.zip:

#### 1. Report (PDF format) (80%)
- All requested plots (TTFT and TPOT visualizations) and a brief analysis of what's going on. (40%)
- A table with Nsight profiling results showing top 3 kernel names and times. (15%)
- Identification (Screenshot) of the All-reduce kernel responsible for tensor parallel communication during multi-GPU inference (15%)
- One paragraph discussing the difficulties or surprising results you have in this assignment (10%)

#### 2. Code (15%)
- Modified `server.sh` file for the 2 multi-GPU scenarios (10%)
- Python scripts for data parsing and visualization (5%)

#### 3. Data Files (5%)
- Benchmark results
- Profiling outputs

## Chatting with the LLM

One of the main use cases of LLMs is the ability to provide a chat user interface to end-users. However, LLMs are trained to generate the next token given a series of tokens in the prompt without any inherent attribution to each token's source (user or LLM).

To generate meaningful responses, it is critical to annotate which parts of the prompts are generated by which entity. LLMs are fine-tuned to follow a specific template to distinguish between user and assistant messages. Conversations need to follow this template for generating prompts for every user's new messages (see [Llama 3.1 documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)).

We have provided a simple Python script called `complete.py` to test this functionality. We encourage you to:
- Go through the file and test custom messages
- Build on top of it to manage conversation history
- Understand how LLM applications like ChatGPT and Claude manage conversation history

---

