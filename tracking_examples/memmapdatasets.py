from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pathlib

# we want image datasets to have keys for image, id, and text
# image is a width x height tensor
# implement as a dataloader

class CelebADataset(Dataset):
    def __init__(self, path = "/mnt/sshd/saxon/CelebA/", extension = "jpg"):
        imgpath = pathlib.path(path) / "img_align_celeba/"
        self.imglist = list(filter(lambda x : x.suffix == "." + extension, imgpath.iterdir()))
        self.len = len(self.imglist)
    
    def __getitem__(self, idx):
        # load image
        im = Image.open(self.imglist[idx])
        return idx

    def __len__(self):
        return self.len