import random
import PIL.Image as pil
import torch
from typing import Union, Optional, Sequence
from tqdm import tqdm
from mlcbase import ConfigDict, Logger, is_str, is_dict
from .utils import *


@INVERSION.register_module("EDICT")
@torch.no_grad()
def exact_diffusion_inversion(image: Union[str, pil.Image],
                              models: Union[str, StableDiffusionT2IModels],
                              prompt: str,
                              z0: Optional[torch.Tensor] = None,
                              mix_weight: float = 0.93,
                              width: int = 512,
                              height: int = 512,
                              num_inference_steps: int = 50,
                              guidance_scale: float = 7.5,
                              scheduler: Optional[CustomScheduler] = None,
                              seed: Optional[int] = None,
                              device: str = "cuda",
                              dtype: torch.dtype = torch.float32,
                              variant: Optional[str] = None, 
                              image_offsets: Sequence = (0, 0, 0, 0), 
                              logger: Optional[Logger] = None):
    """Exact Diffusion Inversion via Coupled Transformations (EDICT): https://arxiv.org/abs/2211.12446
    
    CVPR 2023 Poster paper
    
    Official Implementation: https://github.com/salesforce/EDICT

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        prompt (str): textual prompt
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        mix_weight (float): mixing weight for each diffusion step. Defaults to 0.93.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
        guidance_scale (float): guidance scale for CFG. Defaults to 7.5.
        scheduler (Optional[CustomScheduler]): Defaults to None.
        seed (Optional[int]): Defaults to None.
        device (str): Defaults to "cuda".
        dtype (torch.dtype): Defaults to torch.float32.
        variant (Optional[str]): Defaults to None.
        image_offsets (Sequence): image offsets before center crop, (left, top, right, bottom). 
                                  Defaults to (0, 0, 0, 0).
        logger (Optional[Logger]): Defaults to None.

    Returns:
        dict: outputs
    """
    if logger is None:
        logger = Logger()
        logger.init_logger()
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    outputs = ConfigDict()

    # prepare models
    if is_str(models):
        models = StableDiffusionT2IModels(models, dtype, variant, device)
    if scheduler is None:
        scheduler = CustomDDIMScheduler(beta_start=0.00085, 
                                        beta_end=0.012, 
                                        beta_schedule="scaled_linear", 
                                        clip_sample=False, 
                                        set_alpha_to_one=False, 
                                        steps_offset=1)
    elif is_str(scheduler):
        scheduler = CustomDDIMScheduler.from_pretrained(scheduler)
    elif is_dict(scheduler):
        scheduler = SCHEDULER.build(scheduler)
    assert isinstance(scheduler, CustomDDIMScheduler)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # prepare latent
    if is_str(image):
        image = load_image_from_path(image, (width, height), image_offsets)
    outputs.ori_image = image
    if z0 is None:
        latent = image2latent(image=image, models=models, generator=generator)
        outputs.z0_vae = latent.clone()  # image latent representation from VAE
    else:
        latent = z0

    # prepare prompt
    cond_prompt_emb = encode_prompt(prompt, models, logger)
    uncond_prompt_emb = encode_prompt("", models, logger)
    prompt_emb = torch.cat([uncond_prompt_emb, cond_prompt_emb])
    
    # inverse
    latent_x = latent.clone()
    latent_y = latent.clone()
    with tqdm(total=num_inference_steps, desc="EDICT") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            
            latent_y_inter = (latent_y - (1 - mix_weight) * latent_x) / mix_weight
            latent_x_inter = (latent_x - (1 - mix_weight) * latent_y_inter) / mix_weight
            
            latent_x_input = scheduler.scale_model_input(torch.cat([latent_x_inter]*2), t)
            noise_pred_x = models.unet(latent_x_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_cond = noise_pred_x.chunk(2)
            noise_pred_x = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latent_y = scheduler.approx_inverse_step(noise_pred_x, t, latent_y_inter)
            
            latent_y_input = scheduler.scale_model_input(torch.cat([latent_y]*2), t)
            noise_pred_y = models.unet(latent_y_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_cond = noise_pred_y.chunk(2)
            noise_pred_y = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latent_x = scheduler.approx_inverse_step(noise_pred_y, t, latent_x_inter)
            
            pbar.update(1)
    outputs.zT = [latent_x.clone(), latent_y.clone()]
    
    # denoise
    with tqdm(total=num_inference_steps, desc="denoise") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[i]
            
            latent_y_input = scheduler.scale_model_input(torch.cat([latent_y]*2), t)
            noise_pred_y = models.unet(latent_y_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_cond = noise_pred_y.chunk(2)
            noise_pred_y = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latent_x_inter = scheduler.denoise_step(noise_pred_y, t, latent_x)
            
            latent_x_input = scheduler.scale_model_input(torch.cat([latent_x_inter]*2), t)
            noise_pred_x = models.unet(latent_x_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_cond = noise_pred_x.chunk(2)
            noise_pred_x = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latent_y_inter = scheduler.denoise_step(noise_pred_x, t, latent_y)
            
            latent_x = mix_weight * latent_x_inter + (1 - mix_weight) * latent_y_inter
            latent_y = mix_weight * latent_y_inter + (1 - mix_weight) * latent_x

            pbar.update(1)
    outputs.z0 = [latent_x.clone(), latent_y.clone()]
    
    # latent to image
    x = latent2image(latent_x, models)
    y = latent2image(latent_y, models)
    outputs.recon_image = [x, y]
    
    return outputs
