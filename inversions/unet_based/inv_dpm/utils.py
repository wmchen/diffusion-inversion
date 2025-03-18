from typing import Optional, Union, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import BaseOutput


class InversionPipelineOutput(BaseOutput):
    zT: torch.Tensor
    ori_image: Image.Image


class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))


def load_image_from_path(path: str, 
                         size: Optional[Sequence] = None, 
                         offsets: Sequence = (0, 0, 0, 0), 
                         center_crop: bool = True,
                         color_mode: str = "RGB") -> Image.Image:
    """load image from local path and perform center crop if necessary

    Args:
        path (str): local path
        size (Optional[Sequence]): target image size (width, height). Defaults to None.
        offsets (Sequence, optional): offsets before center crop, (left, top, right, bottom). 
                                      Defaults to (0, 0, 0, 0).
        center_crop (bool, optional): whether to perform center crop. Defaults to True.
        color_mode (str, optional): color mode. Defaults to "RGB".

    Returns:
        pil.Image: image
    """
    image = np.array(Image.open(path).convert(color_mode))

    # perform cropping using offsets
    h, w, _ = image.shape
    left = min(offsets[0], w - 1)
    top = min(offsets[1], h - 1)
    right = min(w - 1, w - offsets[2])
    bottom = min(h - 1, h - offsets[3])
    image = image[top:bottom, left:right, :]

    # perform center crop
    if center_crop:
        h, w, _ = image.shape
        if h > w:
            image = image[(h - w) // 2: (h - w) // 2 + w, :, :]
        elif h < w:
            image = image[:, (w - h) // 2: (w - h) // 2 + h, :]
        
    image = Image.fromarray(image)
    if size is not None:
        image = image.resize(size)
    return image


def image2latents(pipe: DiffusionPipeline, 
                  image: Image.Image, 
                  requires_grad: bool = False,
                  autocast: bool = True) -> torch.Tensor:
    device = pipe._execution_device
    if isinstance(pipe, StableDiffusionPipeline):
        dtype = pipe.text_encoder.dtype
    elif isinstance(pipe, StableDiffusionXLPipeline):
        if pipe.vae.config.force_upcast and autocast:
            dtype = torch.float32
            pipe.vae.to(dtype=torch.float32)
        else:
            dtype = pipe.text_encoder.dtype
    
    image = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
    image = image.to(dtype=dtype, device=device)
    if requires_grad:
        latents = pipe.vae.encode(image).latent_dist.sample()
    else:
        with torch.no_grad():
            latents = pipe.vae.encode(image).latent_dist.sample()
    
    if isinstance(pipe, StableDiffusionPipeline):
        latents = pipe.vae.config.scaling_factor * latents
    elif isinstance(pipe, StableDiffusionXLPipeline):
        latents = pipe.vae.config.scaling_factor * latents
        if pipe.vae.config.force_upcast and autocast:
            pipe.vae.to(dtype=pipe.text_encoder.dtype)
            latents = latents.to(dtype=pipe.text_encoder.dtype)

    return latents


def latents2image(pipe: DiffusionPipeline, 
                  latents: torch.Tensor, 
                  output_type: str = "pil", 
                  requires_grad: bool = False, 
                  autocast: bool = True,
                  **kwargs) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    if isinstance(pipe, StableDiffusionPipeline):
        latents = latents / pipe.vae.config.scaling_factor
    elif isinstance(pipe, StableDiffusionXLPipeline):
        latents = latents / pipe.vae.config.scaling_factor
        if pipe.vae.config.force_upcast and autocast:
            latents = latents.to(dtype=torch.float32)
            pipe.vae.to(dtype=torch.float32)
    
    if requires_grad:
        image = pipe.vae.decode(latents, return_dict=False)[0]
    else:
        with torch.no_grad():
            image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type=output_type, **kwargs)
    
    if isinstance(pipe, StableDiffusionXLPipeline):
        if pipe.vae.config.force_upcast and autocast:
            pipe.vae.to(dtype=pipe.text_encoder.dtype)
            if output_type == "pt" or output_type == "latent":
                image = image.to(dtype=pipe.text_encoder.dtype)
    
    if output_type == "pil":
        image = image[0]
    
    return image


def pil2tensor(image: Image.Image, normalize: bool = False) -> torch.Tensor:
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    if normalize:
        image = 2.0 * image - 1.0  # normalize to [-1, 1]
    return image
