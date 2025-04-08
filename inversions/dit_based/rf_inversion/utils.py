import os.path as osp
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from mlcbase import Logger


class ImageLoader:
    def __init__(self, logger: Optional[Logger] = None, quiet: bool = False):
        if logger is None:
            logger = Logger()
            logger.init_logger()
        if quiet:
            logger.set_quiet()
        else:
            logger.set_activate()
        self.logger = logger

        self._image = None
        self._path = None
        self._mode = None

    @property
    def image(self):
        if self._image is None:
            self.logger.error("Image not loaded yet.")
            raise ValueError("Image not loaded yet.")
        return self._image

    @property
    def path(self):
        return self._path

    def load_image_from_path(self, path: str, color_mode: str = "RGB"):
        self._image = Image.open(path).convert(color_mode)
        self._path = path
        self._mode = color_mode
        self.logger.info(f"[Load] {osp.basename(path)} | original size: {self.image.size}")
        return self.image
    
    def direct_resize_image(self, size: Sequence[int]):
        assert len(size) == 2, "size must be a sequence of two integers"
        self._image = self._image.resize(size)
        self.logger.info(f"[Direct Resize] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image
    
    def scale_image(
        self, 
        match_long_size: Optional[int] = None, 
        match_short_size: Optional[int] = None, 
        scale_ratio: Optional[float] = None
    ):
        assert match_long_size is None or match_short_size is None or scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"

        w, h = self.image.size

        if match_long_size is not None:
            assert match_short_size is None and scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            if w > h:
                ratio = match_long_size / w
            else:
                ratio = match_long_size / h
        
        if match_short_size is not None:
            assert match_long_size is None and scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            if w > h:
                ratio = match_short_size / h
            else:
                ratio = match_short_size / w

        if scale_ratio is not None:
            assert match_long_size is None and match_short_size is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            ratio = scale_ratio

        self._image = self._image.resize((int(w * ratio), int(h * ratio)))
        self.logger.info(f"[Scaling] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image
    
    def adjust_to_scale(
        self,
        scale_factor: int = 16,
        method: str = "center_crop",
        offset: Optional[Sequence[int]] = None,
    ):
        assert method in ["center_crop", "offset", "resize"], "method must be one of ['center_crop', 'offset', 'resize']"
        if method == "offset":
            assert offset is not None, "offset must be provided when method is 'offset'"
            assert len(offset) == 4, "offset must be a sequence of four integers"

        w, h = self.image.size
        if w % scale_factor == 0 and h % scale_factor == 0:
            return self.image
        
        image = np.array(self.image)

        if h % scale_factor != 0:
            if method == "resize":
                new_h = h - h % scale_factor
                image = np.array(Image.fromarray(image, self._mode).resize((w, new_h)))
            
            if method == "center_crop":
                start = (h % scale_factor) // 2
                end = h - (h % scale_factor - start)
                if len(image.shape) == 3:
                    image = image[start:end, :, :]
                else:
                    image = image[start:end]

            if method == "offset":
                assert offset[1] + offset[3] == (h % scale_factor)
                if len(image.shape) == 3:
                    image = image[offset[1]:h-offset[3], :, :]
                else:
                    image = image[offset[1]:h-offset[3]]

        if w % scale_factor != 0:
            if method == "resize":
                new_w = w - w % scale_factor
                image = np.array(Image.fromarray(image, self._mode).resize((new_w, h)))
            
            if method == "center_crop":
                start = (w % scale_factor) // 2
                end = w - (w % scale_factor - start)
                if len(image.shape) == 3:
                    image = image[:, start:end, :]
                else:
                    image = image[:, start:end]
            
            if method == "offset":
                assert offset[0] + offset[2] == (w % scale_factor)
                if len(image.shape) == 3:
                    image = image[:, offset[0]:w-offset[2], :]
                else:
                    image = image[:, offset[0]:w-offset[2]]

        self._image = Image.fromarray(image, self._mode)
        self.logger.info(f"[Adjust2Scale] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image


def image2latent(
    pipe, 
    image, 
    generator: Optional[torch.Generator] = None,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None, 
    device: Optional[str] = None, 
):
    if dtype is None:
        dtype = pipe.transformer.dtype
    if device is None:
        device = pipe._execution_device
    
    image = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
    image = image.to(dtype=dtype, device=device)

    if requires_grad:
        latents = pipe.vae.encode(image).latent_dist.sample(generator)
    else:
        with torch.no_grad():
            latents = pipe.vae.encode(image).latent_dist.sample(generator)

    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    
    return latents


def latent2image(
    pipe, 
    latents, 
    output_type="pil", 
    requires_grad: bool = False
):
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    if requires_grad:
        image = pipe.vae.decode(latents, return_dict=False)[0]
    else:
        with torch.no_grad():
            image = pipe.vae.decode(latents, return_dict=False)[0]

    image = pipe.image_processor.postprocess(image, output_type=output_type)
    if output_type == "pil":
        image = image[0]
        
    return image
