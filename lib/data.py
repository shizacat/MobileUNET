"""
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
"""

from pathlib import Path, PurePath
from typing import Union, List, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


# Transform for mask

class PILToTensor:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return target


class PilConvertToRGB:
    def __call__(self, pil_img):
        return pil_img.convert("RGB")

class ReduceSize:
    """Reduce size of image by k times"""

    def __init__(self, k: Union[int, float]):
        self.k = k

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        img_width, img_height = pil_img.size
        return pil_img.resize(
            (int(img_width / self.k), int(img_height / self.k)), Image.NEAREST
        )


# Dataset

class MaskDataset(Dataset):
    """
    Format:
        The folder contents:
            - 'images' folder
            - 'masks' folder
        Which one contents files of image by named as: xxxxxx.[tif, jpg, png],
        xxxxx - is numeric number.
        Mask - every channel is number of class, 0 - is empty
    """
    def __init__(
        self,
        root_dir: Union[str, PurePath],
        transform=None,
        transform_mask=None
    ):
        self.root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir  # noqa: E501
        self.transform = transform
        self.transform_mask = transform if transform_mask is None else transform_mask  # noqa: E501
        self.fmt = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]  # supported format image
        self.images: List[PurePath] = []

        self._create_list()
    
    def _create_list(self):
        self.images = []

        images_dir = self.root_dir.joinpath("images")
        for x in images_dir.iterdir():
            if x.suffix.lower() not in self.fmt:
                continue
            # check mask
            if not self._get_path_mask(x).is_file():
                raise ValueError(f"The mask not found for: {x.name}")
            self.images.append(x)
    
    def _get_path_mask(self, img_path: PurePath) -> PurePath:
        masks_dir = self.root_dir.joinpath("masks")
        return masks_dir.joinpath(img_path.name)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        img_input = Image.open(self.images[idx])
        img_mask = Image.open(self._get_path_mask(self.images[idx]))

        if self.transform is not None:
            img_input = self.transform(img_input)
        if self.transform_mask is not None:
            img_mask = self.transform_mask(img_mask)
        return img_input, img_mask

    def __len__(self) -> int:
        return len(self.images)
