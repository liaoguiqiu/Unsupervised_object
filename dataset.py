import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, split='train',resolution=(128,128)):
        super(PARTNET, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = "C:/1projects/codes/Object_centric/data/Choracic/images/"     + self.split
        # self.root_dir = "C:/1projects/codes/Object_centric/data/CLEVR_v1.0/images/"     + self.split

        # self.root_dir = "/home/linuxadmin/object_centric_learning/data/CLEVR_v1.0/images/"     + self.split
        # self.root_dir = "/media/data/Choracic/images/"     + self.split

        self.files = os.listdir(self.root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])
        self.resolution =resolution

    def __getitem__(self, index):
        path = self.files[index]
        # image = Image.open(os.path.join(self.root_dir, path, "CLEVR_train_000000.png")).convert("RGB")
        image = Image.open(os.path.join(self.root_dir, path )).convert("RGB")

        image = image.resize(self.resolution)
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)
