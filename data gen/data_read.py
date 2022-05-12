import albumentations as A
from torch.utils.data.dataset import Dataset # For custom datasets
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import os


transform = A.Compose([
A.Resize(width=224, height=224)
])

class CatDogDataset(Dataset):
    def __init__(self, imgs, transforms = transform):
        super().__init__()
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        #print(image_name)
        image = cv2.imread(image_name)
        #image = cv2.resize(image,(224, 224),interpolation = cv2.INTER_AREA)

        if 'two' in image_name:
            label=torch.tensor([1,0,0])
        if 'three' in image_name:
            label=torch.tensor([0,1,0])
        if 'four' in image_name:
            label=torch.tensor([0,0,1])

        if self.transforms is not None:
            augmentations = self.transforms(image=image)
            image = augmentations["image"]
            image=np.moveaxis(image,[0,1,2],[2,1,0])

        return image,label
    


def load_data(DIR_TRAIN):
    all_imgs=[]
    for root,dirs,files in os.walk(DIR_TRAIN):
        for name in files:
            if '.png' in os.path.join(root,name):
                all_imgs.append(os.path.join(root,name))   
    train_dataset = CatDogDataset(all_imgs)
    
    train_data_loader = DataLoader(
        dataset = train_dataset,
        num_workers = 0,
        batch_size = 2,pin_memory=True,
        shuffle = True
    )
    return train_data_loader





