import random
import PIL.Image as pil
import torch
import torch.nn.functional as F
from typing import Union, Optional, Sequence
from mlcbase import ConfigDict, Logger, EmojiProgressBar, is_str, is_dict
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from .tic import AttentionQKVHook
from .utils import *


class AttentionSelfAttnKVInjector(AttentionQKVHook):
    def __init__(self, unet: UNet2DConditionModel, t_align: int):
        self.unet = unet
        self.t_align = t_align
        self.self_attn = None
        self.cur_timestep = None

    def custom_forward(self, attn: Attention):
        
        def forward(hidden_states: torch.Tensor, 
                    encoder_hidden_states: Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None,
                    temb: Optional[torch.Tensor] = None):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
                
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            # -----------------------------------------------------------------------
            # ---------- custom code: Cross-Image Self-Attention Injection ----------
            # -----------------------------------------------------------------------
            if not attn.is_cross_attention:
                _, inv_key, inv_value = self.self_attn.pop(0)
                if self.cur_timestep > self.t_align:
                    # replace
                    key[1, :] = inv_key
                    value[1, :] = inv_value
                else:
                    # concatenation in the spatial dimension
                    key_cond = torch.cat([key[1, :].unsqueeze(0), inv_key], dim=1)
                    key_uncond = torch.cat([key[0, :].unsqueeze(0)] * 2, dim=1)
                    key = torch.cat([key_uncond, key_cond], dim=0)
                    value_cond = torch.cat([value[1, :].unsqueeze(0), inv_value], dim=1)
                    value_uncond = torch.cat([value[0, :].unsqueeze(0)] * 2, dim=1)
                    value = torch.cat([value_uncond, value_cond], dim=0)
            # -----------------------------------------------------------------------
            # ---------- custom code: Cross-Image Self-Attention Injection ----------
            # -----------------------------------------------------------------------
                
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        return forward


@INVERSION.register_module("RIVAL")
@torch.no_grad()
def real_world_image_variation_by_alignment(
    image: Union[str, pil.Image],
    models: Union[str, StableDiffusionT2IModels],
    prompt: str,
    z0: Optional[torch.Tensor] = None,
    t_align: int = 600,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    scheduler: Optional[Union[CustomDDIMScheduler, str, dict]] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    variant: Optional[str] = None, 
    image_offsets: Sequence = (0, 0, 0, 0), 
    logger: Optional[Logger] = None
):
    """Real-world ImageVariation by ALignment (RIVAL): https://arxiv.org/abs/2305.18729
    
    NeurlPS 2023 paper

    Official Implementation: https://github.com/dvlab-research/RIVAL

    To employ RIVAL in diffusion inversion, we directly clone the inversed seed noise as the 
    starting point of generation chain, and apply Cross-Image Self-Attention Injection to perform 
    noise alignment between inversion and generation chains.

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        prompt (str): textual prompt
        negative_prompt (str): textual negative prompt
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        t_align (int): the timestep of early step boundary to perform Cross-Image Self-Attention
                       Injection. Defaults to 600.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
        guidance_scale (float): guidance scale for CFG. Defaults to 5.0.
        scheduler (Optional[Union[CustomDDIMScheduler, str, dict]]): Defaults to None.
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
    do_classifier_free_guidance = guidance_scale > 1
    if do_classifier_free_guidance:
        uncond_prompt_emb = encode_prompt("", models, logger)

    # ddim inversion
    inverse_latents = [latent]
    self_attn = []
    attn_hook = AttentionQKVHook(models.unet)
    attn_hook.parse_module()
    with EmojiProgressBar(total=num_inference_steps, desc="DDIM Inversion") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            latent_model_input = scheduler.scale_model_input(latent, t)
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=cond_prompt_emb).sample
            latent = scheduler.ddim_inverse_step(noise_pred, t, latent)
            inverse_latents.append(latent)
            
            # save q, k, v in self-attention layers
            self_attn.append(attn_hook.self_attn)
            attn_hook.clear_attn()
            
            pbar.update(1)
    outputs.zT = latent.clone()
    outputs.latents = inverse_latents

    # denoise
    injector = AttentionSelfAttnKVInjector(models.unet, t_align)
    injector.parse_module()
    with EmojiProgressBar(total=num_inference_steps, desc="denoise") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[i]
            injector.self_attn = self_attn[num_inference_steps-i-1]
            injector.cur_timestep = t.item()
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
