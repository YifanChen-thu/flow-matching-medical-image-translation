#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("/home/yifan/data")

CT_SUBFOLDERS = [
    "Adrenal_CT_train_val_test",
    "Bladder_Kidney_CT_train_val_test",
    "Lung_CT_train_val_test",
    "Stomach_Colon_Liver_Pancreas_CT_train_val_test",
    "Uterus_Ovary_CT_train_val_test",
]

Brain_MR_SUBFOLDERS = [
    "Breast_MR_train_val_test",
    # "Brain_MR_train_val_test",
]

CT_KEYS = {
    "ct": re.compile(r"(?:^|[_\\-\\.])ct(?:[_\\-\\.]|$)", re.IGNORECASE),
    "ctc": re.compile(r"(?:^|[_\\-\\.])ctc(?:[_\\-\\.]|$)", re.IGNORECASE),
}

NIFTI_EXTS = {".nii", ".nii.gz"}


# ----------------------------
# Helpers
# ----------------------------
def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def normalize_to_uint8(slice_data: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_data, (0.5, 99.5))
    slice_data = np.clip((slice_data - lo) / (hi - lo + 1e-8), 0, 1)
    return (slice_data * 255).astype(np.uint8)


def get_modality(name: str) -> str | None:
    for key, pat in CT_KEYS.items():
        if pat.search(name):
            return key.upper()
    return None


def save_volume(vol_path: Path, out_root: Path, vis_root: Path):
    vol = load_nifti(vol_path)
    case_name = vol_path.stem.replace(".nii", "")
    modality = get_modality(vol_path.name)
    if modality is None:
        print(f"[WARN] Skipping (no modality match): {vol_path.name}")
        return

    # Example path parts:
    # .../Uterus_Ovary_CT_train_val_test/train/C3N-00866/.../C3N-00866_2000-03-05_CT.nii
    dataset_name = vol_path.parents[3].name   # Uterus_Ovary_CT_train_val_test
    split_name   = vol_path.parents[2].name   # train / val / test
    case_id      = case_name                  # e.g. C3N-00866_2000-03-05

    # Output dirs preserve dataset + split
    out_case_dir = out_root / dataset_name / split_name / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)

    num_slices = vol.shape[-1]
    for idx in range(num_slices):
        slice_data = vol[..., idx]
        slice_img = Image.fromarray(normalize_to_uint8(slice_data))
        slice_dir = out_case_dir / f"slice_{idx:03d}"

        slice_dir.mkdir(parents=True, exist_ok=True)
        slice_img.save(slice_dir / f"{modality}.jpg", quality=95)

    # Save mid slice for inspection
    mid_idx = num_slices // 2
    vis_dir = vis_root / dataset_name / split_name / case_id
    vis_dir.mkdir(parents=True, exist_ok=True)
    mid_img = Image.fromarray(normalize_to_uint8(vol[..., mid_idx]))
    mid_img.save(vis_dir / f"{modality}_mid.jpg", quality=95)



from math import log10

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two same-sized arrays (float32)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = np.max([img1.max(), img2.max()])
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))


# ----------------------------
# Core: save CT/CTC pair
# ----------------------------
def normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def save_ct_ctc_pair(ct_path: Path, ctc_path: Path, out_root: Path, vis_root: Path):
    ct_vol = load_nifti(ct_path)
    ctc_vol = load_nifti(ctc_path)

    assert ct_vol.shape == ctc_vol.shape, f"Shape mismatch: {ct_path} vs {ctc_path}"

    case_name = ct_path.stem.replace(".nii", "")
    dataset_name = ct_path.parents[3].name   # e.g. Uterus_Ovary_CT_train_val_test
    split_name   = ct_path.parents[2].name   # train / val / test
    case_id      = case_name                 # e.g. C3N-00866_2000-03-05

    out_case_dir = out_root / dataset_name / split_name / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)

    num_slices = ct_vol.shape[-1]
    psnr_vals = []

    for idx in range(num_slices):
        ct_slice  = ct_vol[..., idx]
        ctc_slice = ctc_vol[..., idx]

        # Compute PSNR
        ct_slice_norm  = normalize(ct_slice)
        ctc_slice_norm = normalize(ctc_slice)

        # compute PSNR
        psnr_val = psnr(ct_slice_norm, ctc_slice_norm)
        psnr_val = np.clip(psnr_val, 0, 100)  # keep in a safe range
        psnr_vals.append(psnr_val)

        slice_dir = out_case_dir / f"slice_{idx:03d}"
        slice_dir.mkdir(parents=True, exist_ok=True)

        if np.std(ct_slice_norm) < 1e-1 or np.std(ctc_slice_norm) < 1e-1:   # threshold can be tuned
            print(f"[WARNING] Slice may be vacant: {slice_dir}, ", np.std(ct_slice_norm), np.std(ctc_slice_norm))
            continue

        # Save normalized images
        ct_img  = Image.fromarray(normalize_to_uint8(ct_slice))
        ctc_img = Image.fromarray(normalize_to_uint8(ctc_slice))

        ct_img.save(slice_dir / "CT.jpg", quality=95)
        ctc_img.save(slice_dir / "CTC.jpg", quality=95)

    # Save mid slice to vis
    mid_idx = num_slices // 2
    vis_dir = vis_root / dataset_name / split_name / case_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    mid_ct = Image.fromarray(normalize_to_uint8(ct_vol[..., mid_idx]))
    mid_ctc = Image.fromarray(normalize_to_uint8(ctc_vol[..., mid_idx]))
    mid_ct.save(vis_dir / "CT_mid.jpg", quality=95)
    mid_ctc.save(vis_dir / "CTC_mid.jpg", quality=95)

    print(f"[{dataset_name}/{split_name}/{case_id}] Avg PSNR (CT vs CTC): {np.mean(psnr_vals):.3f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR, help="Root dataset folder")
    parser.add_argument("--output", type=Path, required=True, help="Output 2D dataset root")
    parser.add_argument("--vis", type=Path, required=True, help="Output 2Dvis folder")
    args = parser.parse_args()

    for sub in CT_SUBFOLDERS:
        subdir = args.input / sub
        if not subdir.exists():
            continue
        print(f"[INFO] Processing {subdir}")

        # Group files by case (CT + CTC pair)
        case_files = {}
        for vol_path in subdir.rglob("*"):
            if not any(str(vol_path).endswith(ext) for ext in NIFTI_EXTS):
                continue
            stem = vol_path.stem.replace(".nii", "")
            base_id = stem.split("_CT")[0].split("_CTC")[0]
            case_files.setdefault(base_id, {})[get_modality(vol_path.name)] = vol_path

        # Process pairs
        for case_id, files in case_files.items():
            if "CT" in files and "CTC" in files:
                try:
                    save_ct_ctc_pair(files["CT"], files["CTC"], args.output, args.vis)
                except Exception as e:
                    print(f"[ERROR] Failed on {case_id}: {e}")
            else:
                print(f"[WARN] Missing CT/CTC pair for case {case_id}")


if __name__ == "__main__":
    main()