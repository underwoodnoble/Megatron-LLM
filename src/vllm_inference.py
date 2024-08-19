import json
import os
from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from utils.inference import load_inference_config, load_inference_dataset, get_inference_parser


assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"


def LLMPredictorWrapper(model_name_or_path, tensor_parallel_size, sampling_params: SamplingParams):
    # Create a class to do batch inference.
    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(model=model_name_or_path,
                        tensor_parallel_size=tensor_parallel_size)

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            outputs = self.llm.generate(batch["prompt"], sampling_params)
            prompt: List[str] = []
            generated_text: List[str] = []
            for output in outputs:
                prompt.append(output.prompt)
                generated_text.append(' '.join([o.text for o in output.outputs]))
            return {
                "prompt": prompt,
                "generated_text": generated_text,
            }

    return LLMPredictor


def main(config_file_path):
    config = load_inference_config(config_file_path)

    # Create a sampling params object.
    sampling_params = SamplingParams(**config.sampling_params)

    
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 1
            }] * config.tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))
    
    
    # load dataset
    ds = load_inference_dataset(config.data_path)
    resources_kwarg: Dict[str, Any] = {}
    if config.tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
    
    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictorWrapper(config.model_name_or_path, config.tensor_parallel_size, sampling_params),
        concurrency=config.num_instances,
        batch_size=config.batch_size,
        **resources_kwarg
    )

    os.makedirs(config.save_path.parent, exist_ok=True)
    with open(config.save_path, 'a') as f:
        for output in ds.iter_rows():
            prompt = output["prompt"]
            generated_text = output["generated_text"]
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            f.write(json.dumps({"prompt": prompt, "responses": generated_text}) + '\n')


if __name__ == '__main__':
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args.config_file_path)