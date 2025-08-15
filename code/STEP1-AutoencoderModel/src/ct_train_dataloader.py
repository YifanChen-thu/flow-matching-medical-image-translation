import os
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, Orientationd, ToTensord, ScaleIntensityD, CropForegroundd
)
from monai.data import PersistentDataset
import nibabel as nib

import numpy as np
from monai.transforms import MapTransform
import pandas as pd


def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)
    print("length of dataset_df: ", len(dataset_df))
    df = dataset_df[dataset_df['Splits'] == mode]
    if args.DEBUG:
        df = df[:10]

    return df


def get_ct_dataloader(args,
                         mode='train', batch_size=2, spatial_size = (128, 128, 128), deterministic=False,
                         key_to_load = ["mask", "density"], cache_dir="./cache"):

    # Get datalist
    datalist = get_dataframe(args, args.mode)
    
    image_keys = args.input_modality  # "ct", "ctc"#[k for k in key_to_load if k != "mask" and k != "patient_id"]
    image_keys = [i.lower() for i in image_keys]

    crop_transform = RandCropByPosNegLabeld(
        keys=key_to_load,
        label_key="mask",
        spatial_size=spatial_size,
        pos=1,
        neg=0,
        num_samples=1
    )

    # Define transforms
    transforms = Compose([
        LoadImaged(keys=key_to_load),
        EnsureChannelFirstd(keys=key_to_load),
        Orientationd(keys=key_to_load, axcodes="RAS"),
        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),  # Normalize all but mask
        crop_transform,
        ToTensord(keys=key_to_load),
    ])


    # Create dataset and dataloader
    dataset    = PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(mode=="train"),  num_workers=4)
    return dataloader