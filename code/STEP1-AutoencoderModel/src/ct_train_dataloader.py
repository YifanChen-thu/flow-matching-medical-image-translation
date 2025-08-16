import os
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, Orientationd, ToTensord, ScaleIntensityD, CropForegroundd, ResizeWithPadOrCropd
)
from monai.data import PersistentDataset
import nibabel as nib

import numpy as np
from monai.transforms import MapTransform
import pandas as pd

from monai.transforms import RandSpatialCropd
from monai import transforms as trans

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
    datalist = get_dataframe(args, mode)
    

    image_keys = args.input_modality  # "ct", "ctc"   # [k for k in key_to_load if k != "mask" and k != "patient_id"]
    # image_keys = [i.upper() for i in image_keys]
    print("cache_dir:", cache_dir)
    print("Read in List with Keys=", datalist.keys())
    print("Input Modality: ", image_keys, key_to_load)

    datalist = datalist.to_dict(orient='records')

    crop_transform = RandCropByPosNegLabeld(
        keys=key_to_load,
        label_key="mask",
        spatial_size=spatial_size,
        pos=1,
        neg=0,
        num_samples=1
    )

    

    crop_transform = RandSpatialCropd(
        keys=key_to_load,
        roi_size=spatial_size,   # crop size
        random_center=True,      # pick random center
        random_size=False        # fixed size
    )

    

    # Define transforms
    transforms = [
        LoadImaged(keys=key_to_load),
        EnsureChannelFirstd(keys=key_to_load),
        Orientationd(keys=key_to_load, axcodes="RAS"),
        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),  # Normalize all but mask
        crop_transform,
        ResizeWithPadOrCropd(
            keys=key_to_load,
            spatial_size=spatial_size,   # <- correct arg name
            allow_missing_keys=False,
            mode="constant",
            value=0  
        ),
        ToTensord(keys=key_to_load),
    ]


    padding_mode = "reflection"
    # {'maximum', 'median', 'edge', 'symmetric', 'wrap', 'constant', 'linear_ramp', 'empty', 'minimum', 'reflect', 'mean'}.

    if mode=='train':
        transforms.extend( [
            trans.RandFlipD(prob=0.5, spatial_axis=0, keys=image_keys),
            trans.RandFlipD(prob=0.5, spatial_axis=1, keys=image_keys),
            trans.RandFlipD(prob=0.5, spatial_axis=2, keys=image_keys),


            trans.RandAffineD(
                keys=image_keys,
                prob=0.5,
                rotate_range=(0.15, 0.15, 0.15),
                # translate_range=(8, 8, 8),          # voxels
                scale_range=(0.1, 0.1, 0.1),        # +/-10%
                shear_range=(0.05, 0.05, 0.05),
                # mode="bilinear",  # {"image": "bilinear", "label": "nearest"} if labelkeys else 
                padding_mode=padding_mode,
            ),
            
            # elastic (smooth nonrigid; great for anatomy)

            # transforms.Rand3DElasticD(
            #     keys=imagekeys,
            #     prob=0.3, sigma_range=(3, 5), magnitude_range=(2, 5),
            #     mode="bilinear",  # {"image": "bilinear", "label": "nearest"} if labelkeys else 
            #     padding_mode=padding_mode,
            #     # as_tensor_output=False
            # ),

            trans.EnsureTypeD(keys=image_keys, dtype="float32"),

        ])

    transforms = Compose(transforms)

    # Create dataset and dataloader
    dataset    = PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(mode=="train"),  num_workers=4)
    return dataloader