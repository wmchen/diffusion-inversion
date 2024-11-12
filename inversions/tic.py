import random
import PIL.Image as pil
import torch
import torch.nn.functional as F
from typing import Union, Optional, Sequence
from mlcbase import ConfigDict, Logger, EmojiProgressBar, is_str, is_dict
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from .utils import *


class AttentionQKVHook:
    def __init__(self, unet: UNet2DConditionModel):
        assert isinstance(unet, UNet2DConditionModel)
        self.unet = unet
        self.cross_attn = []
        self.self_attn = []
        
    def parse_module(self):
        def parse(net, block_type):
            if net.__class__.__name__ == "Attention":
                net.forward = self.custom_forward(net)
            elif hasattr(net, "children"):
                for net in net.children():
                    parse(net, block_type)
        
        for name, module in self.unet.named_children():
            if "down" in name:
                parse(module, "down")
            elif "mid" in name:
                parse(module, "mid")
            elif "up" in name:
                parse(module, "up")
        
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
            
            # -----------------------------------------------
            # ---------- custom code: save q, k, v ----------
            # -----------------------------------------------
            if attn.is_cross_attention:
                self.cross_attn.append((query, key, value))
            else:
                self.self_attn.append((query, key, value))
            # -----------------------------------------------
            # ---------- custom code: save q, k, v ----------
            # -----------------------------------------------
                
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
        
    def clear_attn(self):
        self.cross_attn = []
        self.self_attn = []
        
        
class AttentionQKVInverter:
    def __init__(self, 
                 unet, 
                 cross_attn: Optional[list] = None, 
                 self_attn: Optional[list] = None,
                 cross_start_idx: int = 0,
                 self_start_idx: int = 0,
                 invert_q: bool = True,
                 invert_k: bool = True,
                 invert_v: bool = True):
        self.unet = unet
        self.cross_attn = cross_attn
        self.self_attn = self_attn
        self.cross_start_idx = cross_start_idx
        self.self_start_idx = self_start_idx
        self.invert_q = invert_q
        self.invert_k = invert_k
        self.invert_v = invert_v
        
        self.cross_cursor = 0
        self.self_cursor = 0
        
    def parse_module(self):
        def parse(net, block_type):
            if net.__class__.__name__ == "Attention":
                net.forward = self.custom_forward(net)
            elif hasattr(net, "children"):
                for net in net.children():
                    parse(net, block_type)
        
        for name, module in self.unet.named_children():
            if "down" in name:
                parse(module, "down")
            elif "mid" in name:
                parse(module, "mid")
            elif "up" in name:
                parse(module, "up")
                
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
                
            # -----------------------------------------------
            # --------- custom code: invert q, k, v ---------
            # -----------------------------------------------
            if attn.is_cross_attention:
                if self.cross_attn is not None:
                    if self.cross_cursor >= self.cross_start_idx:
                        query_, key_, value_ = self.cross_attn[0]
                    else:
                        query_, key_, value_ = None, None, None
                    del self.cross_attn[0]
                else:
                    query_, key_, value_ = None, None, None
                self.cross_cursor += 1
            else:
                if self.self_attn is not None:
                    if self.self_cursor >= self.self_start_idx:
                        query_, key_, value_ = self.self_attn[0]
                    else:
                        query_, key_, value_ = None, None, None
                    del self.self_attn[0]
                else:
                    query_, key_, value_ = None, None, None
                self.self_cursor += 1
            
            if self.invert_q and query_ is not None:
                query = query_
            if self.invert_k and key_ is not None:
                key = key_
            if self.invert_v and value_ is not None:
                value = value_
            # -----------------------------------------------
            # --------- custom code: invert q, k, v ---------
            # -----------------------------------------------
                
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


@INVERSION.register_module("TIC")
@torch.no_grad()
def tuning_free_inversion_enhanced_control(
    image: Union[str, pil.Image],
    models: Union[str, StableDiffusionT2IModels],
    z0: Optional[torch.Tensor] = None,
    start_timestep: int = 4,
    start_layer_index: int = 10,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    scheduler: Optional[Union[CustomDDIMScheduler, str, dict]] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    variant: Optional[str] = None, 
    image_offsets: Sequence = (0, 0, 0, 0), 
    logger: Optional[Logger] = None
):
    """Tuning-free Inversion-enhanced Control (TIC): https://arxiv.org/abs/2312.14611
    
    AAAI 2024 paper

    Args:
        image (Union[str, pil.Image]): image path or image object
        models (Union[str, StableDiffusionT2IModels]): model path or model object
        z0 (Optional[torch.Tensor]): initial latent code. Defaults to None.
        start_timestep (int): the timestep to start performing TIC. Defaults to 4.
        start_layer_index (int): the layer index to start performing TIC. Defaults to 10.
        width (int): image width. Defaults to 512.
        height (int): image height. Defaults to 512.
        num_inference_steps (int): diffusion inference steps. Defaults to 50.
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
    prompt = ""  # for reconstruction, the prompt is null text
    prompt_emb = encode_prompt(prompt, models, logger)
    
    # inverse
    inverse_latents = [latent]
    self_attn = []
    attn_hook = AttentionQKVHook(models.unet)
    attn_hook.parse_module()
    with EmojiProgressBar(total=num_inference_steps, desc="TIC") as pbar:
        for i in range(num_inference_steps):
            t = timesteps[num_inference_steps-i-1]
            latent_model_input = scheduler.scale_model_input(latent, t)
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=prompt_emb).sample
            latent = scheduler.ddim_inverse_step(noise_pred, t, latent)
            inverse_latents.append(latent)
            
            # save q, k, v in self-attention layers
            self_attn.append(attn_hook.self_attn)
            attn_hook.clear_attn()
            
            pbar.update(1)
    outputs.zT = latent.clone()
    outputs.latents = inverse_latents
    
    # denoise
    with EmojiProgressBar(total=num_inference_steps, desc="denoise") as pbar:
        for i in range(num_inference_steps):
            if i > start_timestep:
                attn_inverter = AttentionQKVInverter(models.unet, 
                                                     self_attn=self_attn[num_inference_steps-i-1],
                                                     self_start_idx=start_layer_index,
                                                     invert_q=False,
                                                     invert_k=True,
                                                     invert_v=True)
                attn_inverter.parse_module()
            
            t = timesteps[i]
            latent_model_input = scheduler.scale_model_input(latent, t)
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=prompt_emb).sample
            latent = scheduler.denoise_step(noise_pred, t, latent)
            pbar.update(1)
    outputs.z0 = latent.clone()  # image latent representation from denoising

    # latent to image
    recon_image = latent2image(latent, models)
    outputs.recon_image = recon_image

    return outputs
