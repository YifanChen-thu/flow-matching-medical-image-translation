import os
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, Orientationd, ToTensord, ScaleIntensityD, CropForegroundd, Lambdad  #, RandDropoutd
)
from monai.data import PersistentDataset
import nibabel as nib

import numpy as np
from monai.transforms import MapTransform
import h5py
from tqdm import tqdm
from monai.transforms import (
    Compose, EnsureChannelFirstd, ScaleIntensityD, RandSpatialCropd,
    RandFlipd, RandAffined, RandScaleIntensityd, RandShiftIntensityd, ToTensord
)

def get_brats2021_datalist_from_h5(h5f):
    """
    Load data list from HDF5 file containing all modalities and patient IDs.
    """
    
    modalities = ["t1c", "t1n", "t2w", "t2f"]
    
    patient_ids = h5f["patient_ids"][:].astype(str)
    datalist = []

    for i, pid in enumerate(patient_ids):
        data_dict = {
            "index": i,
            "patient_id": pid,
        }
        datalist.append(data_dict)

    # Inspect the first patient
    first = datalist[0]
    print("----- Inspect data for first subject -----")
    i = first["index"]
    for mod in modalities:
        if mod in h5f:
            arr = h5f[mod][i]
            print(f"{mod}: shape = {arr.shape}, dtype = {arr.dtype}")
        else:
            print(f"{mod}: [NOT FOUND in HDF5]")

    h5f.close()

    return datalist






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




class BraTSH5Dataset(Dataset):
    def __init__(self, datalist, h5f, transform=None, keys_to_load=None):
        self.datalist = datalist
        self.h5f = h5f
        self.transform = transform
        self.keys_to_load = keys_to_load or ["t1c", "t1n", "t2w", "t2f", "mask", "density"]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        entry = self.datalist[idx]
        i = entry["index"]

        data = {k: self.h5f[k][i] for k in self.keys_to_load if k in self.h5f}

        # Add patient_id (optional)
        data["patient_id"] = entry["patient_id"]

        if self.transform:
            data = self.transform(data)

        return data


def get_brats_dataloader_from_h5(h5_path, mode='train', batch_size=2, spatial_size=(128, 128, 128), keys_to_load=["t1c", "t1n", "t2w", "t2f", "mask", "density"]):
    from torch.utils.data import DataLoader
    from monai.transforms import CropForegroundd  # Or your BraTSCrop

    # Load datalist + HDF5
    h5f = h5py.File(h5_path, 'r')
    datalist, h5f = get_brats2021_datalist_from_h5(h5f)

    # Train/val split
    split = int(0.8 * len(datalist))
    if mode == "train":
        datalist = datalist[:split]
    else:
        datalist = datalist[split:]

    image_keys = [k for k in keys_to_load if k != "mask"]

    # Define transforms
    transforms = Compose([
        EnsureChannelFirstd(keys=keys_to_load),
        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),
        RandSpatialCropd(keys=keys_to_load, roi_size=spatial_size, random_center=True, random_size=False),

        # Spatial augmentations
        RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=2),

        RandAffined(
            keys=keys_to_load,
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="border"
        ),

        # Intensity augmentation
        # RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
        # RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),

        # CropForegroundd(keys=image_keys + ["mask"], source_key="mask", spatial_size=spatial_size),
        ToTensord(keys=keys_to_load),
    ])





    dataset   = BraTSH5Dataset(datalist, h5f, transform=transforms, keys_to_load=keys_to_load)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=4)
    return dataloader





if __name__ == "__main__":
    # It takes about 40 minutes to prepare the h5
    save_h5 = False
    DEBUG   = False

    def load_nifti(path):   
        return nib.load(path).get_fdata(dtype=np.float32)


    root_dir = "/home/rent_user/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    save_dir = "/home/rent_user/HDD_8T_2/brats_h5"
    save_path = os.path.join(save_dir, "brats_data_all.h5")

    os.makedirs(save_dir, exist_ok=True)
    
    if save_h5:
        subjects = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print("== Found subjects:", len(subjects))
        if DEBUG: subjects = subjects[:3]  # Limit to 10 subjects for debugging


        modalities = ["t1c", "t1n", "t2w", "t2f"]
        all_data = {}
        for patient_dir in tqdm(subjects):
            patient_id = os.path.basename(patient_dir)
            paths = {
                mod: os.path.join(patient_dir, f"{patient_id}-{mod}.nii.gz")
                for mod in modalities
            }
            all_data[patient_id] = paths

        patient_ids = sorted(all_data.keys())
        data_by_modality = {mod: [] for mod in modalities}
        valid_patient_ids = []

        # ✅ Load actual NIfTI data
        for pid in tqdm(patient_ids, desc="Loading NIfTI"):
            try:
                for mod in modalities:
                    if not os.path.exists(all_data[pid][mod]):
                        raise FileNotFoundError(f"Missing modality {mod}")
                volumes = {mod: load_nifti(all_data[pid][mod]) for mod in modalities}

                for mod in modalities:
                    data_by_modality[mod].append(volumes[mod])
                valid_patient_ids.append(pid)

            except Exception as e:
                print(f"[ERROR] {pid} skipped due to: {e}")


        # ✅ Save to HDF5
        
        with h5py.File(save_path, "w") as h5f:
            for mod in modalities:
                stacked = np.stack(data_by_modality[mod], axis=0)
                h5f.create_dataset(mod, data=stacked, compression="gzip")
            h5f.create_dataset("patient_ids", data=np.array(valid_patient_ids, dtype="S"))


        print(f"✅ HDF5 saved to: {save_path}")
        print(f"🧠 Total patients saved: {len(valid_patient_ids)}")
    else:
        import time
        # ✅ Load existing HDF5
        save_path = os.path.join(save_dir, "brats_data_all.h5")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"File not found: {save_path}")
        start_time = time.time()

        h5f = h5py.File(save_path, "r")

        # Get patient IDs
        valid_patient_ids = h5f["patient_ids"][:].astype(str)

        # Prepare data for dataloader
        data_by_modality = {mod: h5f[mod][:] for mod in ["t1c", "t1n", "t2w", "t2f"]}
        end_time = time.time()

        print(f"✅ Loaded HDF5 data from: {save_path}")
        print(f"🧠 Total patients: {len(valid_patient_ids)}")
        print(f"⏱️ Load time: {end_time - start_time:.2f} seconds")



    # 🔍 Print stats for one patient (e.g., first)
    example_idx = 0
    example_id = valid_patient_ids[example_idx]

    print(f"\n🔎 Example Patient: {example_id}")
    for mod in modalities:
        volume = data_by_modality[mod][example_idx]
        print(f"  [{mod.upper()}] shape: {volume.shape}, mean: {volume.mean():.2f}, "
              f"min: {volume.min():.2f}, max: {volume.max():.2f}, std: {volume.std():.2f}")
