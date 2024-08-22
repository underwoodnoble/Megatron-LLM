#! /usr/bin/env bash

# TORCH DEBUG
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# NCCL DEBUG
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=15


export VLLM_WORKER_MULTIPROC_METHOD=fork
REPO_DIR=/cfs/hadoop-mtai-llms/users/TianhaoHu/code/Megatron-LLM
python3 ${REPO_DIR}/src/vllm_inference.py --config_file_path ${REPO_DIR}/tasks/vllm_inference/inference_config.yaml