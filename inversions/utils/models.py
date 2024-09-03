import random
import torch
from typing import Optional, Union
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from mlcbase import is_str, is_dict
from .scheduler import SCHEDULER, CustomDDIMScheduler
from .misc import encode_prompt, latent2image
from .utils import MODELS, Text2ImageModels


@MODELS.register_module()
class StableDiffusionT2IModels(Text2ImageModels):
    def __init__(self, 
                 path: str, 
                 dtype: torch.dtype = torch.float32, 
                 variant: Optional[str] = None, 
                 device: str = "cuda"):
        super().__init__(path, dtype, variant, device)
        self.text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2") \
              if self.model_type == "SDXL" else None
        self.tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2") \
              if self.model_type == "SDXL" else None
        self.unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype, variant=self.variant)
        self.vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype, variant=self.variant)
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder_2 = self.text_encoder_2.to(self.device) if self.text_encoder_2 is not None else None
        self.unet = self.unet.to(self.device)
        self.vae = self.vae.to(self.device)
    
    @torch.no_grad()
    def text2image(self, 
                   prompt: str, 
                   negative_prompt: str = "",
                   width: int = 512,
                   height: int = 512,
                   num_inference_steps: int = 50,
                   guidance_scale: float = 7.5,
                   scheduler: Optional[Union[CustomDDIMScheduler, str, dict]] = None,
                   seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # prepare scheduler
        if scheduler is None:
            # default to use DDIM scheduler
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
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        
        # prepare prompt
        cond_prompt_emb = encode_prompt(prompt, self)
        do_cfg = guidance_scale > 1
        if do_cfg:
            uncond_prompt_emb = encode_prompt(negative_prompt, self)

        # prepare latent
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (1, self.unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        latent = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype, layout=torch.strided)
        
        # denoise
        with tqdm(total=num_inference_steps, desc="denoise") as pbar:
            for i in range(num_inference_steps):
                t = timesteps[i]
                latent_model_input = torch.cat([latent] * 2) if do_cfg else latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                prompt_emb = torch.cat([uncond_prompt_emb, cond_prompt_emb]) if do_cfg else cond_prompt_emb
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_emb).sample
                if do_cfg:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = scheduler.denoise_step(noise_pred, t, latent)
                pbar.update(1)
                
        # latent to image
        image = latent2image(latent, self)
        
        return image
