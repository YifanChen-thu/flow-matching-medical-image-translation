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

def get_brats2021_datalist(root_dir):
    """
    Builds a list of dictionaries with image paths for each modality and segmentation.
    """
    subjects = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print("== Found subjects:", len(subjects))

    datalist = []
    for patient_dir in subjects:
        patient_id = os.path.basename(patient_dir) # -> (240, 240, 155)

        data_dict = {
            "t1c":     os.path.join(patient_dir, f"{patient_id}-t1c.nii.gz"),
            "t1n":     os.path.join(patient_dir, f"{patient_id}-t1n.nii.gz"),
            "t2w":     os.path.join(patient_dir, f"{patient_id}-t2w.nii.gz"),
            "t2f":     os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz"),

            "mask":    os.path.join(patient_dir, f"{patient_id}-seg.nii.gz"),
            "density": os.path.join(patient_dir, f"{patient_id}-tumormap.nii.gz"),
            "patient_id": patient_id
        }
        required_keys = ["t1c", "t1n", "t2w", "t2f", "mask", "density"]
        if all(os.path.exists(data_dict[k]) for k in required_keys):
            datalist.append(data_dict)

    first = datalist[0]
    print("----- Inspect shapes for first subject -----")
    for key, path in first.items():
        if key == "patient_id":
            continue
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

            new_label[label >= 3] = 3  # Inner: Enhancing Tumor
            new_label[label == 1] = 2  # Middle: Necrotic Core
            new_label[label == 2] = 1  # Outer: Edema

            d[key] = new_label / 3.0
        return d





class CropAroundPositiveRegiond(MapTransform):
    def __init__(self, keys, label_key, spatial_size, pos=1):
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = np.array(spatial_size)
        self.pos = pos

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key].copy()[0]  # 1, 240, 240, 155 -> 240, 240, 155

        # Get the coordinates of the positive region
        pos_indices = np.array(np.where(label >= self.pos))
        if pos_indices.size == 0:
            # No positives: fallback to random crop
            image_shape = label.shape
            max_start = [max(0, dim - crop) for dim, crop in zip(image_shape, self.spatial_size)]
            start = np.array([np.random.randint(0, m + 1) if m > 0 else 0 for m in max_start])
            end = start + self.spatial_size
        else:

            # Compute bounding box of the positive region
            min_coords = pos_indices.min(axis=1)
            max_coords = pos_indices.max(axis=1)

            # Target crop center: center of the positive region
            center = ((min_coords + max_coords) / 2).astype(int)

            # Compute start and end indices of the crop
            image_shape = label.shape
            half_size = self.spatial_size // 2

            start = center - half_size
            end = start + self.spatial_size


        # Adjust if crop goes out of bounds
        for i in range(len(start)):
            if start[i] < 0:
                start[i] = 0
                end[i] = self.spatial_size[i]
            elif end[i] > image_shape[i]:
                end[i] = image_shape[i]
                start[i] = end[i] - self.spatial_size[i]

        # Perform the crop for all keys
        slices = tuple(slice(s, e) for s, e in zip(start, end))
        # slices =  (slice(np.int64(42), np.int64(138), None), slice(np.int64(19), np.int64(115), None), slice(np.int64(48), np.int64(144), None))


        for key in self.keys: # + (self.label_key,):
            # print("before key = ", key, "shape = ", d[key].shape)
            d[key] = d[key][(...,) + slices]
            # print("key = ", key, "shape = ", d[key].shape)

        # d[self.label_key] = d[self.label_key][(...,) + slices]

        return d




def get_ct_dataloader(root_dir="~/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
                         mode='train', batch_size=2, spatial_size = (128, 128, 128), deterministic=False,
                         key_to_load = ["mask", "density"], cache_dir="./cache"):

    # Get datalist
    datalist = get_brats2021_datalist(root_dir)
    if mode == 'train':
        datalist = datalist[:int(0.8 * len(datalist))]
    elif mode == 'val' or mode == 'test':
        datalist = datalist[int(0.8 * len(datalist)):]
    elif mode == 'test_all':
        pass
    
    image_keys = [k for k in key_to_load if k != "mask" and k != "patient_id"]


    if not deterministic:
        crop_transform = RandCropByPosNegLabeld(
            keys=key_to_load,
            label_key="mask",
            spatial_size=spatial_size,
            pos=1,
            neg=0,
            num_samples=1
        )
    else:
        crop_transform = CropAroundPositiveRegiond(
            keys=key_to_load,
            label_key="mask",
            spatial_size=spatial_size,
        )



    # Define transforms
    transforms = Compose([
        LoadImaged(keys=key_to_load),
        EnsureChannelFirstd(keys=key_to_load),
        Orientationd(keys=key_to_load, axcodes="RAS"),

        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),  # Normalize all but mask
        BraTSLabelToLayeredLabel(keys=["mask"]),  # Convert BraTS labels to layered labels

        crop_transform,

        ToTensord(keys=key_to_load),
    ])


    # Create dataset and dataloader
    dataset    = PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=mode=="train",
                            num_workers=4)
    return dataloader