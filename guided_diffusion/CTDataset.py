import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from glob import glob


class CTDataset(Dataset):
    def __init__(self, metadata=None, img_dir=None, transform=None, target_transform=lambda x: x, patient_num = -1):
        
        self.training_paths = glob('/nfs/turbo/coe-liyues/bowenbw/3DCT/AAPM_fusion_training/*')
        self.transform = transform
        self.target_transform = target_transform
        self.patient_num = patient_num
        print("length of training data", len(self.training_paths))
        

    def __len__(self):
        return len(self.training_paths)

    def __getitem__(self, idx):
        
        image = np.load(self.training_paths[idx])
        image = np.transpose(image, (2,0,1))
        image = np.clip(image*2-1, -1, 1)

        return torch.from_numpy(image)

    
class CTCondDataset(Dataset):
    def __init__(self, metadata=None, img_dir=None, transform=None, target_transform=lambda x: x, patient_num = -1):
        
        self.training_paths = glob('/nfs/turbo/coe-liyues/bowenbw/3DCT/AAPM_fusion_training/*')
        self.transform = transform
        self.target_transform = target_transform
        self.patient_num = patient_num
        print("length of training data", len(self.training_paths))
        
    def __len__(self):
        return len(self.training_paths)
    
    
    def __getitem__(self, idx):
        image = None
        return None
    


if __name__ == "__main__":
    ds = CTDataset()
    params = {'batch_size': 2}
    training_generator = torch.utils.data.DataLoader(ds, **params)
    ct = 0
    for local_batch in training_generator:
        print(local_batch.shape)
        ct += 1
        if ct > 4:
            break