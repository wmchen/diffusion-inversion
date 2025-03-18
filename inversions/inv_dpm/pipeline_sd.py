from typing import Union, Optional, Callable, List, Dict, Any

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import deprecate
from transformers import get_cosine_schedule_with_warmup
from mlcbase import EmojiProgressBar

from .utils import InversionPipelineOutput, StepScheduler, load_image_from_path, image2latents, latents2image, pil2tensor


class SDInversionPipeline(StableDiffusionPipeline):
    @torch.inference_mode()
    def fixedpoint_correction(
        self, 
        x, 
        t, 
        x_t, 
        order=1, 
        n_iter=500, 
        no_improve_break=30,
        step_size=0.1, 
        th=1e-3, 
        text_embeddings=None, 
        guidance_scale=3.0, 
        scheduler=False, 
        factor=0.5, 
        patience=20, 
        warmup=True, 
        warmup_time=20
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        if order==1:
            input = x.clone()
            original_step_size = step_size
            
            # step size scheduler, reduce when not improved
            if scheduler:
                step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)
            
            sigma_t = self.scheduler_denoise.sigmas[self.scheduler_denoise.step_index + 1]
            sigma_s = self.scheduler_denoise.sigmas[self.scheduler_denoise.step_index]
            alpha_t, sigma_t = self.scheduler_denoise._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s, sigma_s = self.scheduler_denoise._sigma_to_alpha_sigma_t(sigma_s)
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
            h = lambda_t - lambda_s

            min_loss = float("inf")
            no_improve_num = 0
            best_input = None
            for i in range(n_iter):
                # step size warmup
                if warmup:
                    if i < warmup_time:
                        step_size = original_step_size * (i+1)/(warmup_time)
                
                latent_model_input = (torch.cat([input] * 2) if do_classifier_free_guidance else input)
                latent_model_input = self.scheduler_denoise.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                model_output = self.scheduler_denoise.convert_model_output(noise_pred, sample=input)

                x_t_pred = (sigma_t / sigma_s) * input - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

                loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction='sum')
                
                if loss.item() < th:
                    break
                
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_input = input.detach().clone()
                    no_improve_num = 0
                else:
                    no_improve_num += 1
                
                if no_improve_num > no_improve_break:
                    break

                # forward step method
                input = input - step_size * (x_t_pred- x_t)

                if scheduler:
                    step_size = step_scheduler.step(loss)

            return input, i+1, loss.item()
        else:
            raise NotImplementedError
        
    def decoder_inversion(
        self, 
        image: Image.Image,
        num_warmup_steps: int = 10,
        num_training_steps: int = 100,
        lr: float = 0.1,
    ):
        image_latent = pil2tensor(image, normalize=True).to(dtype=torch.float32, device=self._execution_device)
        z = image2latents(self, image).to(dtype=torch.float32)
        z = z.requires_grad_()

        loss_function = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam([z], lr=lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )

        self.vae.to(torch.float32)
        
        with EmojiProgressBar(total=num_training_steps, desc="Decoder inversion") as pbar:
            for _ in range(num_training_steps):
                x_pred = latents2image(self, z, output_type="latent", requires_grad=True)

                loss = loss_function(x_pred, image_latent)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                pbar.set_postfix(dict(loss=loss.item()))
                pbar.update(1)
        
        self.vae.to(self.text_encoder.dtype)

        return z.detach().to(self.text_encoder.dtype)

    def inverse(
        self,
        image: Union[str, Image.Image],
        solver_order: int = 1,
        inv_order: Optional[int] = None,
        step_size: float = 0.1,
        perform_decoder_inversion: bool = True,
        num_warmup_steps: int = 10,
        num_training_steps: int = 100,
        lr: float = 0.1,
        prediction_type: str = "epsilon",
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        assert solver_order in [1, 2], "Only support DPM solver order 1, 2"
        if solver_order == 2:
            raise NotImplementedError("DPM solver order 2 is not implemented yet")

        print("[Inv-DPM] Force to use DPMSolverMultistepScheduler")
        self.scheduler_denoise = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type=prediction_type,
            steps_offset=1, 
            trained_betas=None,
            solver_order=solver_order,
        )
        self.scheduler_inverse = DPMSolverMultistepInverseScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type=prediction_type,
            steps_offset=1, 
            trained_betas=None,
            solver_order=solver_order,
        )
        if inv_order is None:
            inv_order = solver_order
        assert inv_order <= 2, "Only support inversion order 0, 1, 2"

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if isinstance(image, str):
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            offsets = kwargs.get("offsets", (0, 0, 0, 0))
            center_crop = kwargs.get("center_crop", True)
            color_mode = kwargs.get("color_mode", "RGB")
            image = load_image_from_path(image, (width, height), offsets, center_crop, color_mode)
        elif isinstance(image, Image.Image):
            height = image.height
            width = image.width
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps_denoise, num_inference_steps = retrieve_timesteps(
            self.scheduler_denoise, num_inference_steps, device, timesteps, sigmas
        )
        timesteps_inverse, num_inference_steps = retrieve_timesteps(
            self.scheduler_inverse, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        if perform_decoder_inversion:
            latents = self.decoder_inversion(image, num_warmup_steps, num_training_steps, lr)
        else:
            latents = image2latents(self, image)
        latents = latents * self.scheduler_inverse.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        with EmojiProgressBar(total=num_inference_steps, desc="Inv-DPM") as pbar:
            for i in range(num_inference_steps):
                # t = timesteps[num_inference_steps-i-1]
                t_prev = timesteps_inverse[i]
                t = timesteps_denoise[num_inference_steps-i-1]

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler_inverse.scale_model_input(latent_model_input, t_prev)
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t_prev,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                
                x_t = latents.clone()
                latents = self.scheduler_inverse.step(noise_pred, t_prev, latents, return_dict=False)[0]
                
                if self.scheduler_denoise.step_index is None:
                    self.scheduler_denoise._init_step_index(t)

                latents, index, loss = self.fixedpoint_correction(
                    latents, 
                    t, 
                    x_t, 
                    order=1, 
                    text_embeddings=prompt_embeds, 
                    guidance_scale=guidance_scale,
                    step_size=step_size, 
                    scheduler=True
                )

                self.scheduler_denoise._step_index -= 1

                pbar.set_postfix(dict(index=index, loss=loss))
                pbar.update(1)

        # Offload all models
        self.maybe_free_model_hooks()

        return InversionPipelineOutput(zT=latents/self.scheduler_inverse.init_noise_sigma, ori_image=image)
    
    @torch.no_grad()
    def __call__(
        self,
        solver_order: int = 1,
        prediction_type: str = "epsilon",
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        print("[Inv-DPM] Force to use DPMSolverMultistepScheduler")
        self.scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type=prediction_type,
            steps_offset=1, 
            trained_betas=None,
            solver_order=solver_order,
        )
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = latents2image(self, latents)

        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=[image], nsfw_content_detected=None)
