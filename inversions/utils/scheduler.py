import torch
from diffusers import DDIMScheduler
from .utils import SCHEDULER, CustomScheduler


@SCHEDULER.register_module()
class CustomDDIMScheduler(CustomScheduler, DDIMScheduler):
    def denoise_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        """denoising process: zt -> z_{t-1}

        Args:
            noise_pred (torch.Tensor): noise prediction from the diffusion UNet model
            timestep (int): current timestep t
            sample (torch.Tensor): latent code z_t

        Returns:
            torch.Tensor: latent code at previous timestep z_{t-1}
        """
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
        
        prev_sample = phi_t * sample + psi_t * noise_pred

        return prev_sample
    
    def ddim_inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        """inversion process based on the reversible assumption of ODE processes: z_t -> z_{t+1}

        Args:
            noise_pred (torch.Tensor): noise prediction from the diffusion UNet model
            timestep (int): current timestep t
            sample (torch.Tensor): latent code z_t

        Returns:
            torch.Tensor: latent code at next timestep z_{t+1}
        """
        timestep, next_timestep = min(timestep - self.config.num_train_timesteps // self.num_inference_steps, 999), timestep
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            phi_t = (alpha_prod_t_next / alpha_prod_t) ** 0.5
            psi_t = (1 - alpha_prod_t_next) ** 0.5 - (alpha_prod_t_next * beta_prod_t / alpha_prod_t) ** 0.5
        elif self.config.prediction_type == "sample":
            phi_t = ((1-alpha_prod_t_next) / beta_prod_t) ** 0.5
            psi_t = alpha_prod_t_next ** 0.5 - ((1-alpha_prod_t_next) * alpha_prod_t / beta_prod_t) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            phi_t = (alpha_prod_t_next * alpha_prod_t) ** 0.5 + ((1-alpha_prod_t_next) * beta_prod_t) ** 0.5
            psi_t = ((1-alpha_prod_t_next) * alpha_prod_t) ** 0.5 - (alpha_prod_t_next * beta_prod_t) ** 0.5

        # zt+1 = φ_t' * zt + ψ_t' * noise_pred
        next_sample = phi_t * sample + psi_t * noise_pred

        return next_sample
    
    def approx_inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        """inversion process based on the approximation of $z_t \approx z_{t-1}$: z_{t-1} -> z_t

        Copy to latex to check the following equation:

        z_t = \frac{z_{t-1} - \psi_t \epsilon_\theta (z_t, t, p)}{\phi_t}

        Args:
            noise_pred (torch.Tensor): noise prediction from the diffusion UNet model
            timestep (int): current timestep t
            sample (torch.Tensor): previous latent code z_{t-1}

        Returns:
            torch.Tensor: latent code at current timestep z_t
        """
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
        cur_sample = (sample - psi_t * noise_pred) / phi_t
        return cur_sample
