from typing import Tuple, Dict, Any
from argparse import ArgumentParser
from yaml import safe_load

from configs.model_configs import BaseModelConfig


from .data_configs import PretrainDataConfig, SFTDataConfig, GenericDataConfig
from .training_configs import PretrainTrainingConfig, SFTTrainingConfig, GenericTrainingConfig



ALGORITHM_CONFIG_DICT: Dict[str, Any] = {
    "pretrain": {
        "data": PretrainDataConfig,
        "train": PretrainTrainingConfig
    },
    "sft": {
        "data": SFTDataConfig,
        "train": SFTTrainingConfig
    }
}


def get_configs() -> Tuple[GenericDataConfig, GenericTrainingConfig]:
    parser = ArgumentParser()
    parser.add_argument(
        '--algorithm', type=str, required=True, help="training type"
    )
    parser.add_argument(
        '--config_file_path', type=str, required=True, help="path to yaml config file"
    )
    args =  parser.parse_args()

    with open(args.config_file_path, 'r') as f:
        arg_dict = safe_load(f)

    data_config: GenericDataConfig = ALGORITHM_CONFIG_DICT[args.algorithm].load_from_dict(arg_dict)
    training_config: GenericTrainingConfig = ALGORITHM_CONFIG_DICT[args.algorithm].load_from_dict(arg_dict)

    return (data_config, training_config)


