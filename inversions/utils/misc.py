import random
import warnings
import numpy as np
import torch
import PIL.Image as pil
from PIL import Image
from typing import Optional, Union, Sequence, List
from mlcbase import Logger
from .utils import Text2ImageModels


def load_image_from_path(path: str, 
                         size: Sequence = (512, 512), 
                         offsets: Sequence = (0, 0, 0, 0), 
                         color_mode: str = "RGB"):
    """load image from local path and perform center crop if necessary

    Args:
        path (str): local path
        size (Sequence, optional): target image size (width, height). Defaults to (512, 512).
        offsets (Sequence, optional): offsets before center crop, (left, top, right, bottom). 
                                      Defaults to (0, 0, 0, 0).
        color_mode (str, optional): color mode. Defaults to "RGB".

    Returns:
        pil.Image: image
    """
    image = np.array(Image.open(path).convert(color_mode))

    # perform cropping using offsets
    h, w, _ = image.shape
    left = min(offsets[0], w - 1)
    top = min(offsets[1], h - 1)
    right = min(w - 1, w - offsets[2])
    bottom = min(h - 1, h - offsets[3])
    image = image[top:bottom, left:right, :]

    # perform center crop
    h, w, _ = image.shape
    if h > w:
        image = image[(h - w) // 2: (h - w) // 2 + w, :, :]
    elif h < w:
        image = image[:, (w - h) // 2: (w - h) // 2 + h, :]
        
    image = Image.fromarray(image)
    image = image.resize(size)
    return image


def image2latent(image: pil.Image, 
                 models: Text2ImageModels, 
                 generator: Optional[torch.Generator] = None, 
                 seed: Optional[int] = None,
                 need_grad: bool = False):
    """encode an image into the latent space using VAE encoder

    Args:
        image (pil.Image): image in pixel space
        models (Text2ImageModels): SD models
        generator (Optional[torch.Generator]): Defaults to None.
        seed (Optional[int]): Defaults to None.
        need_grad (bool): whether to run with grad. Defaults to False.

    Returns:
        torch.Tensor: latent code z_0
    """
    if generator is None:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=models.device).manual_seed(seed)
    
    image = models.image_processor.preprocess(image).to(device=models.device, dtype=models.dtype)
    if need_grad:
        latent = models.vae.encode(image).latent_dist.sample(generator).to(models.dtype)
    else:
        with torch.no_grad():
            latent = models.vae.encode(image).latent_dist.sample(generator).to(models.dtype)
    latent = models.vae.config.scaling_factor * latent

    return latent


def latent2image(latent: torch.Tensor, 
                 models: Text2ImageModels, 
                 output_type: str = "pil", 
                 need_grad: bool = False):
    """decode a latent code into the pixel space using VAE decoder

    Args:
        latent (torch.Tensor): latent code
        models (Text2ImageModels): SD models
        output_type (str): Options include: "pil", "np", "pt", "latent". Defaults to "pil".
        need_grad (bool): whether to run with grad. Defaults to False.

    Returns:
        pil.Image | np.ndarray | torch.Tensor: image in pixel space
    """
    latent = latent / models.vae.config.scaling_factor
    if need_grad:
        image = models.vae.decode(latent, return_dict=False)[0]
    else:
        with torch.no_grad():
            image = models.vae.decode(latent, return_dict=False)[0]
    image = models.image_processor.postprocess(image, output_type=output_type)
    if output_type == "pil":
        image = image[0]
    
    return image


def pil2tensor(image: pil.Image):
    """pillow image to tensor

    Args:
        image (pil.Image)

    Returns:
        torch.Tensor
    """
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    return image


@torch.no_grad()
def encode_prompt(prompt: Union[str, List[str]], 
                  models: Text2ImageModels, 
                  logger: Optional[Logger] = None):
    """encode textual prompt into text embedding

    Args:
        prompt (Union[str, List[str]]): conditional prompt
        models (Text2ImageModels)
        logger (Optional[Logger]): Defaults to None.

    Returns:
        torch.Tensor: prompt embedding
    """
    text_inputs = models.tokenizer(prompt, 
                                   padding="max_length", 
                                   max_length=models.tokenizer.model_max_length, 
                                   truncation=True, 
                                   return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    untruncated_ids = models.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = models.tokenizer.batch_decode(untruncated_ids[:, models.tokenizer.model_max_length - 1 : -1])
        warn_text = "The following part of your input was truncated because CLIP can only handle sequences up to" \
                    f" {models.tokenizer.model_max_length} tokens: {removed_text}"
        if logger is not None:
            logger.warning(warn_text)
        else:
            warnings.warn(warn_text)
    
    if hasattr(models.text_encoder.config, "use_attention_mask") and models.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(models.device)
    else:
        attention_mask = None

    prompt_embeds = models.text_encoder(text_input_ids.to(models.device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.to(dtype=models.dtype, device=models.device)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    return prompt_embeds
