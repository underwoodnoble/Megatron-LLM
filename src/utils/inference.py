import yaml
from pathlib import Path
from argparse import ArgumentParser

from typing import Dict
from pydantic import BaseModel

import ray


class InferenceConfig(BaseModel):
    model_name_or_path: str
    data_path: Path
    save_path: Path
    tensor_parallel_size: int
    num_instances: int
    sampling_params: Dict
    batch_size: int


def load_inference_config(file_path: str) -> InferenceConfig:
    with open(file_path, 'r') as f:
        return InferenceConfig(**yaml.safe_load(f))

        
def load_inference_dataset(file_path: Path):
    if file_path.suffix in ['.json', '.jsonl']:
        return ray.data.read_json(file_path)
    else:
        raise('Only support json and jsonl format.')

        
def get_inference_parser():
    parser = ArgumentParser()
    parser.add_argument('--config_file_path', type=str)
    return parser