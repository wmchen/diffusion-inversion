import random
import PIL.Image as pil
import torch
from typing import Union, Optional, Sequence
from tqdm import tqdm
from mlcbase import ConfigDict, Logger, is_str, is_dict
from .utils import *



@INVERSION.register_module("FPI")
@torch.no_grad()
def fixed_point_inversion(image: Union[str, pil.Image],
                          models: Union[str, StableDiffusionT2IModels],
                          prompt: str,
                          z0: Optional[torch.Tensor] = None,
                          num_iter_steps: int = 20,
                          width: int = 512,
                          height: int = 512,
                          num_inference_steps: int = 50,
                          guidance_scale: float = 2.0,
                          scheduler: Optional[Union[CustomScheduler, str, dict]] = None,
                          seed: Optional[int] = None,
                          device: str = "cuda",
                          dtype: torch.dtype = torch.float32,
                          variant: Optional[str] = None, 
                          image_offsets: Sequence = (0, 0, 0, 0), 
                          logger: Optional[Logger] = None):
    """Fixed-Point Inversion (FPI): https://arxiv.org/abs/2312.12540v1

    Official Implementation: https://github.com/dvirsamuel/FPI

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        prompt (str): textual prompt
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        num_iter_steps (int): the number of iterative steps. Defaults to 20.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
        guidance_scale (float): guidance scale for CFG. Defaults to 2.0.
        scheduler (Optional[Union[CustomScheduler, str, dict]]): Defaults to None.
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
    do_classifier_free_guidance = guidance_scale > 1
    if do_classifier_free_guidance:
        uncond_prompt_emb = encode_prompt("", models, logger)

    # inverse
    inverse_latents = [latent]
    with tqdm(total=num_inference_steps*num_iter_steps, desc="FPI") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            zt = latent.clone()
            for _ in range(num_iter_steps):
                latent_model_input = torch.cat([zt] * 2) if do_classifier_free_guidance else zt
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                prompt_emb = torch.cat([uncond_prompt_emb, cond_prompt_emb]) if do_classifier_free_guidance else cond_prompt_emb
                noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=prompt_emb).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                zt = scheduler.ddim_inverse_step(noise_pred, t, latent)
                pbar.update(1)
            latent = zt.clone()
            inverse_latents.append(latent)
    outputs.zT = latent.clone()
    outputs.latents = inverse_latents

    # denoise
    with tqdm(total=num_inference_steps, desc="denoise") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[i]
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            prompt_emb = torch.cat([uncond_prompt_emb, cond_prompt_emb]) if do_classifier_free_guidance else cond_prompt_emb
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=prompt_emb).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latent = scheduler.denoise_step(noise_pred, t, latent)
            pbar.update(1)
    outputs.z0 = latent.clone()  # image latent representation from denoising

    # latent to image
    recon_image = latent2image(latent, models)
    outputs.recon_image = recon_image

    return outputs
