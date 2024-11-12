import random
import PIL.Image as pil
import torch
from typing import Union, Optional, Sequence
from mlcbase import ConfigDict, Logger, EmojiProgressBar, is_str, is_dict
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from .utils import *


def f_func(prev_sample: torch.Tensor, 
           cur_sample: torch.Tensor, 
           timestep: torch.Tensor, 
           prompt_emb: torch.Tensor, 
           guidance_scale: float, 
           models: Text2ImageModels, 
           scheduler: SchedulerMixin):
    """The function of the fixed-point solution. Copy to latex to check the following equation:

    z_t^i = f(z_t^{i-1}) 
          = \phi_t * z_{t-1} + \psi_t * \epsilon_\theta (z_t^{i-1}, t, p)
    
    when i = 0, z_t^0 = z_{t-1}

    Args:
        prev_sample (torch.Tensor): z_{t-1}
        cur_sample (torch.Tensor): z_t^{i-1}
        timestep (torch.Tensor): current timestep
        prompt_emb (torch.Tensor): prompt embedding
        guidance_scale (float): guidance scale for CFG
        models (Text2ImageModels)
        scheduler (SchedulerMixin)

    Returns:
        torch.Tensor
    """
    do_classifier_free_guidance = guidance_scale > 1
    latent_model_input = torch.cat([cur_sample] * 2) if do_classifier_free_guidance else cur_sample
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    noise_pred = models.unet(latent_model_input, timestep, encoder_hidden_states=prompt_emb).sample
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    new_cur_latent = scheduler.ddim_inverse_step(noise_pred, timestep, prev_sample)
    return new_cur_latent


@INVERSION.register_module("AIDI")
@torch.no_grad()
def accelerated_iterative_diffusion_inversion(
    image: Union[str, pil.Image],
    models: Union[str, StableDiffusionT2IModels],
    prompt: str,
    num_iter_steps: int = 5,
    m: int = 3,
    accelerate_method: str = "Anderson",
    z0: Optional[torch.Tensor] = None,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 0.0,
    scheduler: Optional[Union[CustomScheduler, str, dict]] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    variant: Optional[str] = None, 
    image_offsets: Sequence = (0, 0, 0, 0), 
    logger: Optional[Logger] = None
):
    """Accelerated Iterative Diffusion Inversion (AIDI): https://arxiv.org/abs/2309.04907

    ICCV 2023 Oral paper

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        prompt (str): textual prompt
        num_iter_steps (int): the number of iterative steps. Defaults to 5.
        accelerate_method (str): the type of accelerate method, options include: Anderson and Empirical. 
                                 Defaults to "Anderson".
        m (int): the window size in Anderson Acceleration. Defaults to 3.
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
        guidance_scale (float): guidance scale for CFG. Defaults to 0.0.
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
    assert accelerate_method in ["Anderson", "Empirical"], "accelerate method must be Anderson or Empirical"
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
    with EmojiProgressBar(total=num_inference_steps*num_iter_steps, desc="AIDI") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            prompt_emb = torch.cat([uncond_prompt_emb, cond_prompt_emb]) if do_classifier_free_guidance else cond_prompt_emb
            zt_0 = latent.clone()  # z_t^0 <-- z_{t-1}
            zt_1 = f_func(latent, zt_0, t, prompt_emb, guidance_scale, models, scheduler)  # z_t^1 <-- f(z_t^0)
            if accelerate_method == "Anderson":
                aa_solver = AndersonAcceleration(window_size=m)
                for _ in range(1, num_iter_steps+1):
                    zt_1 = zt_1.view(-1).cpu().numpy()
                    zt = aa_solver.apply(zt_1)
                    zt_1 = torch.from_numpy(zt).to(device=device, dtype=dtype)
                    zt_1 = zt_1.view(latent.shape)
                    zt_1 = f_func(latent, zt_1, t, prompt_emb, guidance_scale, models, scheduler)
                    pbar.update(1)
            else:
                for _ in range(num_iter_steps):
                    item_0 = f_func(latent, zt_0, t, prompt_emb, guidance_scale, models, scheduler)
                    item_1 = f_func(latent, zt_1, t, prompt_emb, guidance_scale, models, scheduler)
                    zt = 0.5 * item_0 + 0.5 * item_1
                    zt_0 = zt_1.clone()
                    zt_1 = zt.clone()
                    pbar.update(1)
            latent = zt_1.clone()
            inverse_latents.append(latent)
    outputs.zT = latent.clone()
    outputs.latents = inverse_latents

    # denoise
    with EmojiProgressBar(total=num_inference_steps, desc="denoise") as pbar:
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
