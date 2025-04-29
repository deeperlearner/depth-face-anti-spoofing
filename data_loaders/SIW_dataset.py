import os
import glob
import logging

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image


class SIWDataset(Dataset):
    def __init__(self, data_dir="./data", mode="test"):
        self.data_dir = data_dir
        self.mode = mode
        if mode == "test":
            self.df = pd.read_csv(os.path.join(data_dir, "Siw_filelist_sub.txt"), sep=' ', header=None)
            self.df = self.df.drop(columns=2)

        # Normalization
        self.transform = transforms.Compose(
            [
                # transforms.RandomRotation(10),
                transforms.Resize((128, 128)),
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __getitem__(self, idx):
        if self.mode == "test":
            file_path = os.path.join(self.data_dir, self.df.loc[idx, 0])
            live = self.df.loc[idx, 1]

            image = Image.open(file_path)
            if self.transform:
                image = self.transform(image)
            depth = image

            return image, depth, live

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    dataset = SIWDataset(data_dir="/media/back/internal/share/antispoof_data/Siw/crop/train")
    image, depth, live = dataset[0]
    print(len(dataset))
