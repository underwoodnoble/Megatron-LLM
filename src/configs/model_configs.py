from typing import Dict, Any
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    model_path: str = Field(title="Model Path.")
    model_type: str = Field(title="Model Type.")
    tp: int = Field(title="Tensor Parallel Size.")
    pp: int = Field(title="Pipeline Parallel Size.")
    vp: int = Field(title="Virtual Pipeline Parallel Size.")
    ep: int = Field(title="Expert Parallel Size.")
    cp: int = Field(title="Context Parallel Size.")
    
    def load_from_dict(self, arg_dict: Dict[str, Any]):
        self.model_path = arg_dict['model_path']
        self.model_type = arg_dict['model_type']
