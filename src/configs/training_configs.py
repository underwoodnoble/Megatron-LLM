from typing import Dict, Any, TypeVar
from pydantic import BaseModel, Field


class BaseTrainingConfig(BaseModel):
    training_type: str = Field(default='pretrain', examples=['pretrain', 'sft'])

    def load_from_dict(self, arg_dict: Dict[str, Any]):
        self.training_type = arg_dict.get('training_type', 'pretrain')


class PretrainTrainingConfig(BaseTrainingConfig):
    def load_from_dict(self, arg_dict: Dict[str, Any]):
        super().load_from_dict(arg_dict)


class SFTTrainingConfig(BaseTrainingConfig):
    def load_from_dict(self, arg_dict: Dict[str, Any]):
        super().load_from_dict(arg_dict)


GenericTrainingConfig = TypeVar(
    "GenericTrainingConfig",
    PretrainTrainingConfig,
    SFTTrainingConfig,
)
