import os.path as osp
import torch
from abc import ABC, abstractmethod
from typing import Optional
from mlcbase import Registry, load_json


MODELS = Registry("models")
SCHEDULER = Registry("scheduler")
INVERSION = Registry("inversion")


class Text2ImageModels(ABC):
    def __init__(self, 
                 path: str, 
                 dtype: torch.dtype, 
                 variant: Optional[str] = None, 
                 device: str = "cuda"):
        super().__init__()
        self.model_path = path
        self.dtype = dtype
        self.variant = variant
        self.device = torch.device(device)
        
        # check model type
        model_index = load_json(osp.join(path, "model_index.json"))
        if model_index._class_name.startswith("StableDiffusion"):
            if model_index._class_name == "StableDiffusionXLPipeline":
                self.model_type = "SDXL"
            else:
                self.model_type = "SD"
        elif model_index._class_name.startswith("LatentConsistencyModel"):
            self.model_type = "LCM"
        else:
            raise TypeError(f"Model type '{model_index._class_name}' not supported.")

    @abstractmethod
    def text2image(self):
        pass


class CustomScheduler(ABC):
    
    @abstractmethod
    def denoise_step(self):
        pass
