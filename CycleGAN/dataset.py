# Dataset
import torch.nn.functional as F
import math
from torchvision import transforms
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
class Custom_dataset(Dataset):
    RESIZE = 256
    # realA_path = Path("./data/trainA")
    # realB_path = Path("./data/trainB/")

    def __init__(self, realA_path, realB_path, use_transformA=False, use_transformB=False):
        super().__init__()
        self.listImgA = Custom_dataset.getImgList(realA_path)
        self.listImgB = Custom_dataset.getImgList(realB_path)
        self.imgA_path = Path(realA_path)
        self.imgB_path = Path(realB_path)
        if use_transformA:
            self.transformA = Custom_dataset.transformA

        if use_transformB:
            self.transformB = Custom_dataset.transformB

        self.img_len = self.listImgA if len(self.listImgA) > len(self.listImgB) else self.listImgB

    def __len__(self):
        return len(self.listImgA)

    def __getitem__(self, idx):

        idx_A = idx % len(self.listImgA)
        idx_B = idx % len(self.listImgB)
        imgA_path = self.listImgA[idx_A]
        imgB_path = self.listImgB[idx_B]

        imgA = Image.open(imgA_path)
        imgB = Image.open(imgB_path)
        while imgA.mode != "RGB":
            idx_A += 1
            imgA_path = self.listImgA[idx_A]
            imgA = Image.open(imgA_path)
        if self.transformA:
            imgA = self.transformA(imgA)
        else:
            imgA = transforms.ToTensor()(imgA)

        while imgB.mode != "RGB":
            idx_B += 1
            imgB_path = self.listImgB[idx_B]
            imgB = Image.open(imgB_path)
        if self.transformB:
            imgB = self.transformB(imgB)
        else:
            imgB = transforms.ToTensor()(imgB)

        return imgA, imgB

    @classmethod
    def transformA(cls, img):
        result = transforms.Compose([
                transforms.Resize((cls.RESIZE, cls.RESIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))  # Scales to [-1, 1]
            ])
        return result(img)

    @classmethod
    def transformB(cls, img):
        result = transforms.Compose([
                transforms.Resize((cls.RESIZE, cls.RESIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scales to [-1, 1]
            ])
        return result(img)
    @staticmethod
    def getImgList(path):
        nameList = glob.glob(f'{path}/*.jpg')
        return nameList
def get_dataloader(realA_path, realB_path, batch_size, use_transformA=True, use_transformB=True):
    data = Custom_dataset(realA_path, realB_path, use_transformA=use_transformA, use_transformB=use_transformB)
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader


# Mapping from A -> B (photos to sketches)