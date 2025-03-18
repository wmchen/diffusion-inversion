import torch
from diffusers import DDIMScheduler


class CustomInversionScheduler:
    pass


class CustomDDIMInversionScheduler(DDIMScheduler, CustomInversionScheduler):
    def ddim_inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        """inversion process based on the reversible assumption of ODE processes: z_{t-1} -> z_t

        Args:
            noise_pred (torch.Tensor): noise prediction from the diffusion UNet model
            timestep (int): current timestep t-1
            sample (torch.Tensor): latent code z_{t-1}

        Returns:
            torch.Tensor: latent code at next timestep z_t
        """
        prev_timestep = timestep
        timestep = min(
            timestep - self.config.num_train_timesteps // self.num_inference_steps, self.config.num_train_timesteps - 1
        )
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_epsilon = noise_pred
        elif self.config.prediction_type == "sample":
            pred_original_sample = noise_pred
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * noise_pred
            pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * sample
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample
