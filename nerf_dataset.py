import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image

class LegoDataset(Dataset):
    def __init__(self, split="train"):
        self.data_directory = "lego"
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])

        with open(os.path.join(self.data_directory, f'transforms_{split}.json'), "r", encoding="utf-8") as f:
            self.data = json.load(f)["frames"]

        self.first_point = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index = 0
        if self.first_point:
            return self.first_point
        img_name = os.path.join(self.data_directory, self.split, f"r_{index}.png")
        img = Image.open(img_name).convert('RGB')  # Convert to RGB if alpha channel is not needed

        if self.transform:
            img = self.transform(img)

        self.first_point = (img, torch.tensor(self.data[index]["transform_matrix"], dtype=torch.float32))

        return img, torch.tensor(self.data[index]["transform_matrix"], dtype=torch.float32)
