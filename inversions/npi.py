import random
import PIL.Image as pil
import torch
from typing import Union, Optional, Sequence
from tqdm import tqdm
from mlcbase import ConfigDict, Logger, is_str, is_dict
from .utils import *


@INVERSION.register_module("NPI")
@torch.no_grad()
def negative_prompt_inversion(image: Union[str, pil.Image],
                              models: Union[str, StableDiffusionT2IModels],
                              prompt: str,
                              z0: Optional[torch.Tensor] = None,
                              width: int = 512,
                              height: int = 512,
                              num_inference_steps: int = 50,
                              scheduler: Optional[Union[CustomScheduler, str, dict]] = None,
                              seed: Optional[int] = None,
                              device: str = "cuda",
                              dtype: torch.dtype = torch.float32,
                              variant: Optional[str] = None, 
                              image_offsets: Sequence = (0, 0, 0, 0), 
                              logger: Optional[Logger] = None):
    """Negative Prompt Inversion (NPI): http://arxiv.org/abs/2305.16807

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        prompt (str): textual prompt
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
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

    # inverse
    inverse_latents = [latent]
    with tqdm(total=num_inference_steps, desc="NPI") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            latent_model_input = scheduler.scale_model_input(latent, t)
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=cond_prompt_emb).sample
            latent = scheduler.ddim_inverse_step(noise_pred, t, latent)
            inverse_latents.append(latent)
            pbar.update(1)
    outputs.zT = latent.clone()
    outputs.latents = inverse_latents

    # denoise
    with tqdm(total=num_inference_steps, desc="denoise") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[i]
            latent_model_input = scheduler.scale_model_input(latent, t)
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=cond_prompt_emb).sample
            latent = scheduler.denoise_step(noise_pred, t, latent)
            pbar.update(1)
    outputs.z0 = latent.clone()  # image latent representation from denoising

    # latent to image
    recon_image = latent2image(latent, models)
    outputs.recon_image = recon_image

    return outputs
