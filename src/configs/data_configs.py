from typing import List, Dict, Any, TypeVar
from pathlib import Path
from pydantic import BaseModel, Field


class BaseDataConfig(BaseModel):
    train_data_list: List[Path] = Field(default_factory=lambda:[], title="training set paths.")
    eval_data_list: List[Path] = Field(default_factory=lambda:[], title="evaluate data paths.")

    def load_from_dict(self, arg_dict: Dict[str, Any]):
        self.train_data_list = [Path(path) for path in arg_dict['train_data_list']]
        self.eval_data_list = [Path(path) for path in arg_dict['eval_data_list']]


class PretrainDataConfig(BaseDataConfig):
    def load_from_dict(self, arg_dict: Dict[str, Any]):
        super().load_from_dict(arg_dict)


class SFTDataConfig(BaseDataConfig):
    def load_from_dict(self, arg_dict: Dict[str, Any]):
        super().load_from_dict(arg_dict)


GenericDataConfig = TypeVar(
    "GenericDataConfig",
    PretrainDataConfig,
    SFTDataConfig,
)
