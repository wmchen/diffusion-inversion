import sys
sys.path.append("../../")

import os.path as osp
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from accelerate.utils import set_seed
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from inversions.unet_based.rival import (SDInversionPipeline, SDXLInversionPipeline, CustomDDIMInversionScheduler, 
                                         image2latents, latents2image)
from inversions.utils import pil2tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["SDv1.5", "SDv2.1", "SDXLv1.0"],  default="SDv1.5")
    parser.add_argument("--model_id", type=str, default=None)

    parser.add_argument("--image", type=str, default="../../demo/alley.jpg")
    parser.add_argument("--prompt", type=str, default="A narrow alley way with a building in the background.")
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--t_align", type=int, default=600)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--torch_dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    if args.output is None:
        args.output = str(Path(__file__).parent.parent.parent)

    assert args.model_type in ["SDv1.5", "SDv2.1", "SDXLv1.0"]
    if args.model_id is None:
        if args.model_type == "SDv1.5":
            args.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif args.model_type == "SDv2.1":
            args.model_id = "stabilityai/stable-diffusion-2-1"
        elif args.model_type == "SDXLv1.0":
            args.model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    args_text = "Args:\n"
    for k, v in vars(args).items():
        args_text += f"{k}: {v}\n"
    print(args_text)

    # init pipeline
    scheduler = CustomDDIMInversionScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    if args.model_type in ["SDv1.5", "SDv2.1"]:
        pipe = SDInversionPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=args.torch_dtype, 
            variant=args.variant, 
            scheduler=scheduler
        )
    else:
        pipe = SDXLInversionPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=args.torch_dtype, 
            variant=args.variant, 
            scheduler=scheduler
        )
    pipe.to(args.device)
    lpips_loss = lpips.LPIPS(net='alex')

    # inference
    inv_result = pipe.inverse(
        image=args.image,
        prompt=args.prompt,
        guidance_scale=0.0,
        num_inference_steps=args.num_inference_steps
    )

    ori_image = inv_result.ori_image
    vae_latent = image2latents(pipe, ori_image)
    vae_recon = latents2image(pipe, vae_latent)

    recon_image = pipe(
        cache_attentions=inv_result.cache_attentions,
        t_align=args.t_align,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latents=inv_result.zT
    ).images[0]

    # metrics
    vae_psnr = psnr(np.array(ori_image), np.array(vae_recon))
    vae_ssim = ssim(np.array(ori_image), np.array(vae_recon), win_size=11, channel_axis=2)
    vae_lpips = lpips_loss(pil2tensor(ori_image), pil2tensor(vae_recon)).item()
    print(f"[VAE Reconstruction] PSNR: {vae_psnr:.2f}, SSIM: {vae_ssim:.4f}, LPIPS: {vae_lpips:.4f}")

    psnr_score = psnr(np.array(ori_image), np.array(recon_image))
    ssim_score = ssim(np.array(ori_image), np.array(recon_image), win_size=11, channel_axis=2)
    lpips_score = lpips_loss(pil2tensor(ori_image), pil2tensor(recon_image)).item()
    print(f"[RIVAL] PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}")

    # output
    fig = plt.figure(figsize=(15, 5))
    axs = fig.subplots(1, 3)
    axs[0].set_title("Origin")
    axs[0].imshow(np.array(ori_image))
    axs[1].set_title(f"VAE Recon. ({vae_psnr:.2f} dB)")
    axs[1].imshow(np.array(vae_recon))
    axs[2].set_title(f"RIVAL ({psnr_score:.2f} dB)")
    axs[2].imshow(np.array(recon_image))
    plt.savefig(osp.join(args.output, "result.png"), bbox_inches='tight')


if __name__ == "__main__":
    main()
