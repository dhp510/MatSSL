import torchvision
from torchvision.transforms import (
    Compose, Resize, RandomResizedCrop, ColorJitter,
    RandomGrayscale, GaussianBlur, RandomHorizontalFlip,
    RandomVerticalFlip, RandomRotation, ToTensor, Normalize
)
from dotenv import load_dotenv
import os
import torch
from PIL import Image

load_dotenv(override=True)

input_size = int(os.getenv("SSL_INPUT_SIZE", "224"))

def create_transform():
    cj_prob = 0.8
    cj_strength = 1.0
    cj_bright = 0.4
    cj_contrast = 0.4
    cj_sat = 0.4
    cj_hue = 0.1
    min_scale = 0.2
    random_gray_scale = 0.2
    gaussian_blur = 0.5
    kernel_size = None
    sigmas = (0.1, 2.0)
    vf_prob = 0.0
    hf_prob = 0.5
    rr_prob = 0.0
    rr_degrees = None
    normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    
    transforms = [Resize((input_size, input_size))]

    transforms.append(RandomResizedCrop(input_size, scale=(min_scale, 1.0)))

    if cj_prob > 0:
        color_jitter = ColorJitter(
            brightness=cj_bright * cj_strength,
            contrast=cj_contrast * cj_strength,
            saturation=cj_sat * cj_strength,
            hue=cj_hue * cj_strength,
        )
        transforms.append(
            torchvision.transforms.RandomApply([color_jitter], p=cj_prob)
        )

    if random_gray_scale > 0:
        transforms.append(RandomGrayscale(p=random_gray_scale))

    if gaussian_blur > 0:
        blur_kernel = int(kernel_size) if kernel_size else int(0.1 * input_size)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        transforms.append(
            torchvision.transforms.RandomApply(
                [GaussianBlur(kernel_size=blur_kernel, sigma=sigmas)],
                p=gaussian_blur
            )
        )

    if hf_prob > 0:
        transforms.append(RandomHorizontalFlip(p=hf_prob))
    if vf_prob > 0:
        transforms.append(RandomVerticalFlip(p=vf_prob))
    if rr_prob > 0 and rr_degrees is not None:
        transforms.append(RandomRotation(degrees=rr_degrees))

    transforms.append(ToTensor())
    if normalize:
        transforms.append(Normalize(mean=normalize["mean"], std=normalize["std"]))

    return Compose(transforms)


# Custom collate function
def custom_collate_fn(batch):
    view_1, view_2 = [], []
    transform_1 = create_transform()
    transform_2 = create_transform()
    
    for item in batch:
        img, _, _ = item  # LightlyDataset returns (img, label, filename)
        if not isinstance(img, Image.Image):
            print(f"Error: Expected PIL Image, got {type(img)}")
            raise TypeError(f"Batch item is {type(img)}, expected PIL Image")
        view_1.append(transform_1(img))
        view_2.append(transform_2(img))
    
    return torch.stack(view_1), torch.stack(view_2)