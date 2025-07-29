import os
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, Orientationd, ToTensord, ScaleIntensityD, CropForegroundd, Lambdad, SpatialPadd, Spacingd
)
from monai.data import PersistentDataset
import nibabel as nib

import numpy as np
import torch


def get_brats2021_datalist(img_dir, mask_dir, latent_dir):
    """
    Builds a list of dictionaries with image paths for each modality and segmentation.
    """


    mask_subjects = [os.path.join(mask_dir, d) for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
    print("== Found subjects:", len(mask_subjects))

    img_subjects = [os.path.join(img_dir, d) for d in os.listdir(img_dir)]  # all 


    img_datalist = []
    mask_datalist = []

    for patient_dir in img_subjects:
        patient_id = os.path.basename(patient_dir)

        if os.path.exists(patient_dir):
            img_datalist.append( { "t1n": patient_dir,
                                  "full_mask": patient_dir.replace("Original", "STAPLE").replace(".nii.gz", "_staple.nii.gz"),
                                  } )

        # print("patient_dir = ", patient_dir, "patient_id = ", patient_id)

    for patient_dir in mask_subjects:
        patient_id = os.path.basename(patient_dir) # -> (240, 240, 155)


        data_dict = {

            # "latent":  os.path.join(latent_dir, f"{patient_id}.npz"),  # Path to latent representation
            # "broken-latent":  os.path.join(latent_dir, f"{patient_id}-broken.npz"),  # Path to latent representation

            "mask":    os.path.join(patient_dir, f"{patient_id}-seg.nii.gz"),
            "density": os.path.join(patient_dir, f"{patient_id}-tumormap.nii.gz"),
        }


        required_keys = ["mask", "density"]  # "t2w", "t2f", "t1n"
        if all(os.path.exists(data_dict[k]) for k in required_keys):
            mask_datalist.append(data_dict)


    datalist = []
    length = min(len(mask_datalist), len(img_datalist))
    print(f"== Found {len(mask_datalist)} masks and {len(img_datalist)} images, using {length} pairs for inference.")

    # Loop over the minimum number of entries to avoid index errors
    for i in range(length):
        d = {"mask": mask_datalist[i]["mask"],
            "density": mask_datalist[i]["density"],

            # "broken-latent": mask_datalist[i]["broken-latent"],
            # "latent": img_datalist[i]["t1n"],  # Assuming t1n is the latent representation

            "full_mask": img_datalist[i]["full_mask"],  # Full mask path
            "t1n": img_datalist[i]["t1n"],
        }
        
        # print("d = ", d)

        datalist.append(d)



    first = datalist[0]
    print("----- Inspect shapes for first subject -----")
    for key, path in first.items():
        if key == "latent" or key == "broken-latent":
            # Load latent representation
            data = np.load(path, allow_pickle=True)["data"]
        else:
            # Load NIfTI image
            img  = nib.load(path)
            data = img.get_fdata()

        print(f"{key}: shape = {data.shape}, dtype (after load) = {data.dtype}, original dtype = {img.get_data_dtype()}")

    return datalist

from monai.transforms import MapTransform

class BraTSLabelToLayeredLabel(MapTransform):
    """
    Convert BraTS labels to layered tumor labels with decreasing order:
    3: Enhancing Tumor (ET)
    2: Non-enhancing/Necrotic Core (NET/NCR)
    1: Edema (ED)
    0: Background
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            new_label = np.zeros_like(label, dtype=np.float32)

            # BraTS original labels
            # 4 = Enhancing Tumor (ET)
            # 1 = Necrotic/Non-enhancing Tumor Core (NCR/NET)
            # 2 = Edema (ED)

            new_label[label >= 3] = 2  # Middle: Necrotic Core
            new_label[label == 1] = 3  # Inner: Enhancing Tumor
            new_label[label == 2] = 1  # Outer: Edema

            d[key] = new_label / 3.0
        return d




class CropAroundPositiveRegiond(MapTransform):
    """
    Crop a spatial region around positive voxels in a label volume.

    Args:
        keys (list[str]): Keys to apply the crop to.
        label_key (str): Key that contains the label (must be in the same shape).
        spatial_size (Sequence[int]): Desired output spatial size (H, W, D).
        pos (int or float): Threshold to consider a voxel positive.
    """
    def __init__(self, keys, label_key, spatial_size, pos=1):
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = torch.tensor(spatial_size)
        self.pos = pos

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]

        if isinstance(label, torch.Tensor):
            label = label.clone()
        else:
            label = torch.tensor(label)

        # Remove channel dim: assume label is (1, H, W, D)
        label = label[0]

        pos_indices = (label >= self.pos).nonzero(as_tuple=False)

        if pos_indices.numel() == 0:
            # No positives: fallback to random crop
            image_shape = torch.tensor(label.shape)
            max_start = torch.clamp(image_shape - self.spatial_size, min=0)
            start = torch.tensor([torch.randint(0, int(m) + 1, (1,)).item() if m > 0 else 0 for m in max_start])
        else:
            min_coords = pos_indices.min(dim=0)[0]
            max_coords = pos_indices.max(dim=0)[0]
            center = ((min_coords + max_coords) // 2).long()
            start = center - (self.spatial_size // 2)

        # Adjust if crop goes out of bounds
        image_shape = torch.tensor(label.shape)
        start = torch.clamp(start, min=0)
        end = start + self.spatial_size

        for i in range(3):
            if end[i] > image_shape[i]:
                end[i] = image_shape[i]
                start[i] = image_shape[i] - self.spatial_size[i]

        slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))

        for key in self.keys:
            original = d[key]
            if isinstance(original, torch.Tensor):
                if key == "t1n":
                    d["full_" + key] = original.clone()
                d[key] = original[(...,) + slices]
            else:
                if key == "t1n":
                    d["full_" + key] = np.copy(original)
                d[key] = original[(...,) + slices]

        
        d['start'] = start
        d['end']   = end

        return d




def get_infer_dataloader(img_dir="",
                         mask_dir="~/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
                         latent_dir="",
                         mode='train', batch_size=2, spatial_size = (128, 128, 128),
                         key_to_load = ["mask", "density"], cache_dir="./cache"):

    key_to_load = set(key_to_load)  # Ensure unique keys

    # Get datalist
    datalist = get_brats2021_datalist(img_dir, mask_dir, latent_dir)
    if mode == 'train':
        datalist = datalist[:int(0.8 * len(datalist))]
    else:
        datalist = datalist[int(0.8 * len(datalist)):]

    image_keys = [k for k in key_to_load if k != "mask"]



    # Define transforms
    transforms = Compose([
        LoadImaged(keys=key_to_load),
        EnsureChannelFirstd(keys=key_to_load),
        Orientationd(keys=key_to_load, axcodes="RAS"),

        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),  # Normalize all but mask
        BraTSLabelToLayeredLabel(keys=["mask"]),  # Convert BraTS labels to layered labels

        # RuntimeError: stack expects each tensor to be equal size, but got [1, 200, 256, 256] at entry 0 and [1, 224, 256, 256] at entry 1 [rank1]: Collate error on the key 'full_t1n' of dictionary data.

        Spacingd(
            keys=[key for key in key_to_load if key not in ("mask", "full_mask")],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),

        # Spacing for masks only
        Spacingd(
            keys=("mask", "full_mask"),
            pixdim=(1.0, 1.0, 1.0),
            mode="nearest"
        ),

        

        SpatialPadd(
            keys=[k for k in key_to_load if k not in ("latent", "broken-latent", "full_mask")],
            spatial_size=(96, 96, 96),
            mode="constant",  # or "reflect" if you prefer
            constant_values=0.0  # Padding value
        ),

        CropAroundPositiveRegiond(
            keys=[k for k in key_to_load if k != "latent" and k != "broken-latent" and k !=  "full_mask"],  # Exclude latent and mask from cropping
            label_key="mask",   # Automatically add in this
            spatial_size=spatial_size,
        ),


        ToTensord(keys=key_to_load),
    ])


    # Create dataset and dataloader
    dataset = PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=mode=="train",
                            num_workers=4)
    return dataloader