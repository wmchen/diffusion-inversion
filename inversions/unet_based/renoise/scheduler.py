import torch
from diffusers import DDIMScheduler


class CustomInversionScheduler:
    pass


class CustomDDIMInversionScheduler(DDIMScheduler, CustomInversionScheduler):
    def approx_inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            phi_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
            psi_t = (1 - alpha_prod_t_prev) ** 0.5 - (alpha_prod_t_prev * beta_prod_t / alpha_prod_t) ** 0.5
        elif self.config.prediction_type == "sample":
            phi_t = ((1-alpha_prod_t_prev) / beta_prod_t) ** 0.5
            psi_t = alpha_prod_t_prev ** 0.5 - ((1-alpha_prod_t_prev) * alpha_prod_t / beta_prod_t) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            phi_t = (alpha_prod_t_prev * alpha_prod_t) ** 0.5 + ((1-alpha_prod_t_prev) * beta_prod_t) ** 0.5
            psi_t = ((1-alpha_prod_t_prev) * alpha_prod_t) ** 0.5 - (alpha_prod_t_prev * beta_prod_t) ** 0.5

        # zt = (zt-1 - ψ_t * noise_pred) / φ_t
        next_sample = (sample - psi_t * noise_pred) / phi_t
        return next_sample
