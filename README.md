# Megatron-LLM
A codebase for llm training based on Megatron-LM


### LSP PYTHONPATH setting
1. Create an .env file in the project directory.
2. Add the following configuration to the .env file.
```shell 
PYTHONPATH=3rdparty/Megatron-LM:3rdparty/VLLM:${PYTHONPATH}
```
