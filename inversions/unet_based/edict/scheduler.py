import torch
from diffusers import DDIMScheduler


class CustomInversionScheduler:
    pass


class CustomDDIMInversionScheduler(DDIMScheduler, CustomInversionScheduler):
    def inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            a_t = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)
            b_t = (1 - alpha_prod_t_prev) ** (0.5) - (alpha_prod_t_prev * beta_prod_t / alpha_prod_t) ** (0.5)
        elif self.config.prediction_type == "sample":
            a_t = alpha_prod_t_prev ** (0.5) + ((1 - alpha_prod_t_prev) / beta_prod_t) ** (0.5)
            b_t = - ((1 - alpha_prod_t_prev) * alpha_prod_t / beta_prod_t) ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            a_t = (alpha_prod_t_prev * alpha_prod_t) ** (0.5) + ((1 - alpha_prod_t_prev) * beta_prod_t) ** (0.5)
            b_t = ((1 - alpha_prod_t_prev) * alpha_prod_t) ** (0.5) - (alpha_prod_t_prev * beta_prod_t) ** (0.5)

        next_sample = (sample - b_t * noise_pred) / a_t

        return next_sample

    def denoise_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            a_t = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)
            b_t = (1 - alpha_prod_t_prev) ** (0.5) - (alpha_prod_t_prev * beta_prod_t / alpha_prod_t) ** (0.5)
        elif self.config.prediction_type == "sample":
            a_t = alpha_prod_t_prev ** (0.5) + ((1 - alpha_prod_t_prev) / beta_prod_t) ** (0.5)
            b_t = - ((1 - alpha_prod_t_prev) * alpha_prod_t / beta_prod_t) ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            a_t = (alpha_prod_t_prev * alpha_prod_t) ** (0.5) + ((1 - alpha_prod_t_prev) * beta_prod_t) ** (0.5)
            b_t = ((1 - alpha_prod_t_prev) * alpha_prod_t) ** (0.5) - (alpha_prod_t_prev * beta_prod_t) ** (0.5)
        
        prev_sample = a_t * sample + b_t * noise_pred

        return prev_sample
