from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import pathlib

# we want image datasets to have keys for image, id, and text
# image is a width x height tensor
# implement as a dataloader

def generate_augmentations(resolution = 256, random_flip = False, center_crop = True):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class CelebADataset(Dataset):
    def __init__(self, path = "/mnt/sshd/saxon/CelebA/", extension = "jpg", augmentations = generate_augmentations()):
        imgpath = pathlib.path(path) / "img_align_celeba/"
        self.imglist = list(filter(lambda x : x.suffix == f".{extension}", imgpath.iterdir()))
        self.len = len(self.imglist)
        self.augmentations = augmentations
    
    def __getitem__(self, idx):
        image_PIL = Image.open(self.imglist[idx])
        image_tensor = self.augmentations(image_PIL.convert("RGB"))
        return {"index" : idx, "input" : image_tensor}

    def __len__(self):
        return self.len
    
