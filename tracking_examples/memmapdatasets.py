from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import pathlib

# we want image datasets to have keys for image, id, and text
# image is a width x height tensor
# implement as a dataloader

def generate_augmentations(arguments = {"resolution" : 256, "random_flip" : False, "center_crop" : True}):
    resolution, random_flip, center_crop = arguments["resolution"], arguments["random_flip"], arguments["center_crop"]
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
    def __init__(self, path = "/mnt/sshd/saxon/CelebA/", extension = "jpg", augmentations = {}, ignore_rows = []):
        basepath = pathlib.Path(path) 
        imgpath = basepath / "img_align_celeba/"
        self.imglist = list(filter(lambda x : x.suffix == f".{extension}", imgpath.iterdir()))
        default_vals = {"resolution" : 256, "random_flip" : False, "center_crop" : True}
        for key in default_vals.keys():
            if key not in augmentations.keys():
                augmentations[key] = default_vals[key]
        self.augmentations = generate_augmentations(augmentations)
        # labels contains one-hot coded attributes, can be converted into 
        self.labels = list(map(
            lambda x: list(map(
                lambda y: max(0, int(y)), x.strip().split(" ")[1:]
            )), 
            open(basepath / "list_attr_celeba.txt").readlines()[1:]
        ))
        self.indices = list(range(len(self.imglist)))
        if len(ignore_rows) > 0:
            for ignore_row in ignore_rows:
                self.indices.remove(ignore_row)
        self.len = len(self.indices)


    def __getitem__(self, idx):
        idx = self.indices[idx]
        image_PIL = Image.open(self.imglist[idx])
        image_tensor = self.augmentations(image_PIL.convert("RGB"))
        return {"index" : idx, "input" : image_tensor, "label" : torch.tensor(self.labels[idx])}

    def __len__(self):
        return self.len
    
