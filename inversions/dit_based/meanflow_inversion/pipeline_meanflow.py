import os
from typing import Union, List, Optional

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from einops import rearrange
from torchvision import transforms
from tqdm import tqdm




class MeanFlowInversionPipeline(DiffusionPipeline):


    def __init__(self, dit_model, vae: AutoencoderKL):

        super().__init__()
        self.register_modules(dit_model=dit_model, vae=vae)
        self.dit_model = dit_model
        self.vae = vae

    @torch.no_grad()
    def inverse(
        self,
        image: Union[str, Image.Image],
        class_label: int,
        num_inference_steps: int = 5,
        image_size: int = 256,
        generator: Optional[torch.Generator] = None,
    ) -> dict:

        # 1. 图像预处理
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)

        # 2. 编码到潜在空间
        z = self.vae.encode(image_tensor).latent_dist.sample(generator=generator) * 0.18215

        # 3. 准备类别标签
        y = torch.tensor([class_label], device=self.device)

        # 4. 反演循环
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float32)
        with tqdm(total=num_inference_steps, desc="Inversion") as pbar:
            for i in range(num_inference_steps):
                t = torch.full((1,), timesteps[i], device=self.device)
                r = torch.full((1,), timesteps[i + 1], device=self.device)
                t_ = rearrange(t, "b -> b 1 1 1")
                r_ = rearrange(r, "b -> b 1 1 1")

                # u(z_t, r, t, y) -> 估算从 t 到 r 的速度
                u = self.dit_model(z, r, t, y)
                # 近似反解公式: z_{t+1} = z_t + (r - t) * u
                z = z + (r_ - t_) * u
                pbar.update(1)

        return {"inverted_latent": z, "original_image": image}

    @torch.no_grad()
    def __call__(
        self,
        class_label: int,
        latents: torch.FloatTensor,
        num_inference_steps: int = 5,
        output_type: str = "pil",
    ) -> dict:

        z = latents.clone().to(self.device)
        y = torch.tensor([class_label], device=self.device)

        # 1. 去噪循环
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
        with tqdm(total=num_inference_steps, desc="Generation") as pbar:
            for i in range(num_inference_steps):
                t = torch.full((1,), timesteps[i], device=self.device)
                r = torch.full((1,), timesteps[i + 1], device=self.device)
                t_ = rearrange(t, "b -> b 1 1 1")
                r_ = rearrange(r, "b -> b 1 1 1")

                # 正向公式: z_{t-1} = z_t - (t - r) * u
                u = self.dit_model(z, r, t, y)
                z = z - (t_ - r_) * u
                pbar.update(1)

        if output_type == "latent":
            return {"image": z}

        # 2. 解码到图像空间
        image = self.vae.decode(z / 0.18215).sample[0]
        image = (image.clamp(-1, 1) + 1) / 2 * 255
        image = image.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(image)

        return {"image": image}
