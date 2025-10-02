
import os
from typing import Union, List, Optional

import torch
from PIL import Image

from diffusers.models import AutoencoderKL

from inversions.dit_based.meanflow_inversion.pipeline_meanflow import MeanFlowInversionPipeline
from inversions.dit_based.meanflow_inversion.models import DiT_models  # 确保您的 DiT_models 定义可以被正确导入

def main(args):
    # 1. 设置环境和种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    dit_model.load_state_dict(state_dict.get("ema", state_dict))
    dit_model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)

    # 3. 初始化 Pipeline
    pipeline = MeanFlowInversionPipeline(dit_model=dit_model, vae=vae)
    pipeline.to(device)

    # 4. 执行反演
    inversion_result = pipeline.inverse(
        image=args.input_image,
        class_label=args.class_label,
        num_inference_steps=args.num_steps,
        image_size=args.image_size
    )
    inverted_latent = inversion_result["inverted_latent"]
    original_image = inversion_result["original_image"]

    # 5. 从反演的 latent 解码出 z1 图像
    z1_image = pipeline(
        class_label=args.class_label,
        latents=inverted_latent,
        num_inference_steps=0,  # 直接解码，不去噪
        output_type="pil"
    )["image"]

    # 6. 执行生成（去噪）
    reconstructed_image = pipeline(
        class_label=args.class_label,
        latents=inverted_latent,
        num_inference_steps=args.num_steps,
        output_type="pil"
    )["image"]

    # 7. 保存结果并拼接
    os.makedirs(args.output_path, exist_ok=True)
    
    # 拼接三张图
    images = [original_image, z1_image, reconstructed_image]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.width

    filename = f"result_step{args.num_steps}_{args.remark}.png"
    save_path = os.path.join(args.output_path, filename)
    new_img.save(save_path)
    print(f"合成图已保存到 {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument("--model_name", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--model_path", type=str, default="/home/ailab/model_weights_nas/MeanFlow/MeanFlow_DiT-B-4_CFG-w3.0-k0.0_80e_20250927.pth")
    parser.add_argument("--vae", type=str, default="/home/zhuzh/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493")

    parser.add_argument("--class_label", type=int, default=1000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--input_image", type=str, default="/mnt/nas_ssd_cache/434_datasets_ssd/patch_datasets/patch_dataset_train_aligned_256-128/patch_images/aligned/hr/0361_patch012_hr.png", help="Path to input image (256x256)")

    # output
    parser.add_argument("--output_path", type=str, default="result")
    parser.add_argument("--remark", type=str, default="pipeline_refactor", help="Remark for the output filename")
    args = parser.parse_args()
    main(args)