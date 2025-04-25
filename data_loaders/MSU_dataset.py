import os
import glob
import logging

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image


class MSUDataset(Dataset):
    def __init__(self, data_dir="./data", mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        if mode == "train":
            self.attack_df = pd.read_csv(os.path.join(data_dir, "attack_train.csv"))
            self.face_df = pd.read_csv(os.path.join(data_dir, "face_train.csv"))
            self.depth_df = pd.read_csv(os.path.join(data_dir, "depth_train.csv"))
        elif mode == "test":
            self.attack_df = pd.read_csv(os.path.join(data_dir, "attack_test.csv"))
            self.face_df = pd.read_csv(os.path.join(data_dir, "face_test.csv"))
            self.depth_df = pd.read_csv(os.path.join(data_dir, "depth_test.csv"))

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
        self.depth_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        if self.mode == "train":
            # spoof
            real_idx = idx // 2
            if idx % 2 == 0:
                live = 0
                face_file = self.attack_df.loc[real_idx % len(self.attack_df)]["path"]
                face_file = os.path.join(self.data_dir, face_file)
                depth = torch.zeros((1, 32, 32))
            else: # live
                live = 1
                face_file = self.face_df.loc[real_idx % len(self.face_df)]["path"]
                face_file = os.path.join(self.data_dir, face_file)
                depth_file = self.depth_df.loc[idx % len(self.depth_df)]["path"]
                depth_file = os.path.join(self.data_dir, depth_file)

        image = Image.open(face_file)
        if self.transform:
            image = self.transform(image)
        if live:
            depth = Image.open(depth_file).convert('L')
            depth = self.depth_transform(depth)

        if self.mode == "train":
            return image, depth, live

    def __len__(self):
        if self.mode == "train":
            return max(len(self.attack_df), len(self.face_df)) * 2
        else:
            return len(self.attack_df) + len(self.face_df)

if __name__ == "__main__":
    dataset = MSUDataset()
