from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
import nibabel as nib
import torch

PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.Resize((128, 128, -1)),
    tio.CropOrPad(target_shape = (128, 128, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

def nibabel_reader(path):
    nib_img = nib.load(path)
    data = nib_img.get_fdata()
    affine = nib_img.affine
    data = torch.from_numpy(data).float()
    if data.ndim == 3:
        data = data.unsqueeze(0)
    return data, affine


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSFORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        subfolder_names = os.listdir(self.root_dir)
        folder_names = [os.path.join(
            self.root_dir, subfolder, 'arterial.nii.gz') for subfolder in subfolder_names]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx], reader = nibabel_reader)
        if img.shape[0] != 1:
            img = img.data.permute(-1, 0, 1, 2)
        img = self.preprocessing(img)
        img = self.transforms(img)
        return {'data': img.data.permute(0, -1, 1, 2)}
