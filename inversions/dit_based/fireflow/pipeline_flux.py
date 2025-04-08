from typing import Union, Optional, Callable, Dict, List, Any

import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from mlcbase import Logger, EmojiProgressBar

from .utils import InversionOutput, ImageLoader, image2latent


class FireFlowFluxPipeline(FluxPipeline):
    @torch.no_grad()
    def inverse(
        self,
        image: Union[str, Image.Image],
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        logger: Optional[Logger] = None,
        quiet: bool = False
    ):
        assert isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)

        if isinstance(image, str):
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor
            im_loader = ImageLoader(logger, quiet)
            im_loader.load_image_from_path(image)
            im_loader.scale_image(match_long_size=height)
            image = im_loader.adjust_to_scale(scale_factor=16)
        elif isinstance(image, Image.Image):
            height = image.height
            width = image.width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = image2latent(
            self, 
            image, 
            generator=generator, 
            dtype=prompt_embeds.dtype, 
            device=device
        )
        latents = self._pack_latents(
            latents,
            batch_size,
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor)
        )
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu, is_inverse=True)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        cached_v = None
        with EmojiProgressBar(total=num_inference_steps, desc="FireFlow (inversion)") as pbar:
            for t in timesteps:
                if self.interrupt:
                    continue

                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                sigma_mid = 0.5 * (sigma_cur + sigma_next)

                if cached_v is None:
                    timestep_cur = t.expand(latents.shape[0]).to(latents.dtype)
                    v_cur = self.transformer(
                        hidden_states=latents,
                        timestep=timestep_cur / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    v_cur = cached_v

                sample = latents.to(torch.float32)
                latents_mid = sample + (sigma_next - sigma_cur) / 2 * v_cur
                latents_mid = latents_mid.to(v_cur.dtype)

                t_mid = sigma_mid * self.scheduler.config.num_train_timesteps
                timestep_mid = t_mid.expand(latents.shape[0]).to(latents.dtype)
                v_mid = self.transformer(
                    hidden_states=latents_mid,
                    timestep=timestep_mid / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                cached_v = v_mid

                latents = sample + (sigma_next - sigma_cur) * v_mid
                latents = latents.to(v_mid.dtype)
                
                self.scheduler._step_index += 1

                pbar.update(1)

        # Offload all models
        self.maybe_free_model_hooks()

        return InversionOutput(init_noise=latents, ori_image=image)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        cached_v = None
        with EmojiProgressBar(total=num_inference_steps, desc="FireFlow (denoise)") as pbar:
            for t in timesteps:
                if self.interrupt:
                    continue

                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                sigma_mid = 0.5 * (sigma_cur + sigma_next)

                if cached_v is None:
                    timestep_cur = t.expand(latents.shape[0]).to(latents.dtype)
                    v_cur = self.transformer(
                        hidden_states=latents,
                        timestep=timestep_cur / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    v_cur = cached_v

                sample = latents.to(torch.float32)
                latents_mid = sample + (sigma_next - sigma_cur) / 2 * v_cur
                latents_mid = latents_mid.to(v_cur.dtype)

                t_mid = sigma_mid * self.scheduler.config.num_train_timesteps
                timestep_mid = t_mid.expand(latents.shape[0]).to(latents.dtype)
                v_mid = self.transformer(
                    hidden_states=latents_mid,
                    timestep=timestep_mid / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                cached_v = v_mid

                latents = sample + (sigma_next - sigma_cur) * v_mid
                latents = latents.to(v_mid.dtype)
                
                self.scheduler._step_index += 1

                pbar.update(1)

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
