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

from inversions.dit_based.fireflow import (FireFlowFluxPipeline, FlowMatchEulerDiscreteInversionScheduler, 
                                           ImageLoader, image2latent, latent2image)
from inversions.utils import pil2tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)

    parser.add_argument("--image", type=str, default="../../demo/alley.jpg")
    parser.add_argument("--prompt", type=str, default="A narrow alley way with a building in the background.")
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    parser.add_argument("--torch_dtype", type=torch.dtype, default=torch.bfloat16)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    if args.output is None:
        args.output = str(Path(__file__).parent.parent.parent)

    args_text = "Args:\n"
    for k, v in vars(args).items():
        args_text += f"{k}: {v}\n"
    print(args_text)

    # init pipeline
    scheduler = FlowMatchEulerDiscreteInversionScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    pipe = FireFlowFluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=args.torch_dtype,
        scheduler=scheduler
    )
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    lpips_loss = lpips.LPIPS(net='alex')

    # inference
    im_loader = ImageLoader()
    im_loader.load_image_from_path(args.image)
    im_loader.direct_resize_image((1024, 1024))
    ori_image = im_loader.image
    inv_result = pipe.inverse(
        image=ori_image,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        height=ori_image.height,
        width=ori_image.width,
        guidance_scale=args.guidance_scale
    )
    recon_image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        height=ori_image.height,
        width=ori_image.width,
        guidance_scale=args.guidance_scale,
        latents=inv_result.init_noise
    ).images[0]

    vae_latent = image2latent(pipe, ori_image, generator=torch.Generator(args.device).manual_seed(args.seed))
    vae_recon = latent2image(pipe, vae_latent)

    # metrics
    vae_psnr = psnr(np.array(ori_image), np.array(vae_recon))
    vae_ssim = ssim(np.array(ori_image), np.array(vae_recon), win_size=11, channel_axis=2)
    vae_lpips = lpips_loss(pil2tensor(ori_image), pil2tensor(vae_recon)).item()
    print(f"[VAE Reconstruction] PSNR: {vae_psnr:.2f}, SSIM: {vae_ssim:.4f}, LPIPS: {vae_lpips:.4f}")

    psnr_score = psnr(np.array(ori_image), np.array(recon_image))
    ssim_score = ssim(np.array(ori_image), np.array(recon_image), win_size=11, channel_axis=2)
    lpips_score = lpips_loss(pil2tensor(ori_image), pil2tensor(recon_image)).item()
    print(f"[FireFlow] PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}")

    # output
    fig = plt.figure(figsize=(15, 5))
    axs = fig.subplots(1, 3)
    axs[0].set_title("Origin")
    axs[0].imshow(np.array(ori_image))
    axs[1].set_title(f"VAE Recon. ({vae_psnr:.2f} dB)")
    axs[1].imshow(np.array(vae_recon))
    axs[2].set_title(f"FireFlow ({psnr_score:.2f} dB)")
    axs[2].imshow(np.array(recon_image))
    plt.savefig(osp.join(args.output, "result.png"), bbox_inches='tight')


if __name__ == "__main__":
    main()
