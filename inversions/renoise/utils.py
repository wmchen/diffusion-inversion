from typing import Optional, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import BaseOutput


class InversionPipelineOutput(BaseOutput):
    zT: torch.Tensor
    ori_image: Image.Image


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


def latents_kl_divergence(x0, x1):
    EPSILON = 1e-6
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
        torch.log((var1 + EPSILON) / (var0 + EPSILON))
        + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON)
        - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl


def patchify_latents_kl_divergence(x0, x1, patch_size=4, num_channels=4):

    def patchify_tensor(input_tensor):
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches

    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl


def auto_corr_loss(x, random_shift=True, generator=None):
    B, C, H, W = x.shape
    assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(0, noise.shape[2] // 2, (1,), generator=generator).item()
            else:
                roll_amount = 1
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=2)
            ).mean() ** 2
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=3)
            ).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def noise_regularization(
    e_t, 
    noise_pred_optimal, 
    lambda_kl, 
    lambda_ac, 
    num_reg_steps, 
    num_ac_rolls, 
    generator=None
):
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_ac = auto_corr_loss(_var, generator=generator)
                l_ac.backward()
                _grad = _var.grad.detach() / num_ac_rolls
                e_t = e_t - lambda_ac * _grad
        e_t = e_t.detach()

    return e_t
