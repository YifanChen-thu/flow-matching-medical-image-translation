#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from PIL import Image, ImageDraw, ImageFont

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

Brain_MR_KEYS = {
    "t1": re.compile(r"(?:^|[_\-\.])t1(?:[_\-\.]|$)", re.IGNORECASE),
    "t1gd": re.compile(r"(?:^|[_\-\.])t1gd(?:[_\-\.]|$)", re.IGNORECASE),
    "t2": re.compile(r"(?:^|[_\-\.])t2(?:[_\-\.]|$)", re.IGNORECASE),
    "flair": re.compile(r"(?:^|[_\-\.])flair(?:[_\-\.]|$)", re.IGNORECASE),
    "mask": re.compile(r"glistrboost(?!.*manuallycorrected)", re.IGNORECASE),
    "mask_correct": re.compile(r"glistrboost.*manuallycorrected", re.IGNORECASE),
}

CT_KEYS = {
    "ct": re.compile(r"(?:^|[_\-\.])ct(?:[_\-\.]|$)", re.IGNORECASE),
    "ctc": re.compile(r"(?:^|[_\-\.])ctc(?:[_\-\.]|$)", re.IGNORECASE),
}

NIFTI_EXTS = {".nii", ".nii.gz"}

# ----------------------------
# Utils
# ----------------------------
def is_nifti(p: Path) -> bool:
    if p.suffix == ".gz" and p.name.endswith(".nii.gz"):
        return True
    return p.suffix in NIFTI_EXTS

def load_ras_img(path: Path) -> nib.Nifti1Image:
    """Load and reorient to RAS (closest canonical)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img

def get_data_u8(img: nib.Nifti1Image, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Return uint8 volume after percentile windowing."""
    vol = img.get_fdata(dtype=np.float32)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros(vol.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)
    return (vol * 255.0).astype(np.uint8)

def middle_indices_xyz(shape):
    x, y, z = shape
    return x // 2, y // 2, z // 2

def extract_middle_slices_xyz(vol_u8: np.ndarray):
    """
    Expect vol shape (X, Y, Z). Return axial, coronal, sagittal 2D arrays.
    Axial:   slice along Z
    Coronal: slice along Y
    Sagittal:slice along X
    """
    if vol_u8.ndim != 3:
        raise ValueError(f"Volume must be 3D, got {vol_u8.shape}")
    xc, yc, zc = middle_indices_xyz(vol_u8.shape)
    axial    = vol_u8[:, :, zc]    # (X, Y)
    coronal  = vol_u8[:, yc, :]    # (X, Z)
    sagittal = vol_u8[xc, :, :]    # (Y, Z)



    # For consistent display: transpose to (H, W) as (Y, X)-like
    # and flip vertically so that "up" is superior-ish.
    axial    = np.flipud(axial.T)
    coronal  = np.flipud(coronal.T)
    sagittal = np.flipud(sagittal.T)
    return axial, coronal, sagittal

def to_pil(img2d_u8: np.ndarray) -> Image.Image:
    return Image.fromarray(img2d_u8)

def label_tile(pil_img: Image.Image, text: str) -> Image.Image:
    img = pil_img.convert("L")
    draw = ImageDraw.Draw(img)
    # Use default bitmap font; small border for legibility
    w, h = img.size
    pad = 6
    box_w = int(min(120, w*0.35))
    box_h = 20
    draw.rectangle([pad, pad, pad+box_w, pad+box_h], fill=0)
    draw.text((pad+4, pad+3), text, fill=255)
    return img

def make_grid(rows, tile_h=256):
    # Resize each tile to a standard height, keep aspect
    scaled_rows = []
    for r in rows:
        scaled = []
        for img in r:
            w, h = img.size
            new_w = int(round(w * (tile_h / float(h))))
            scaled.append(img.resize((new_w, tile_h), Image.BILINEAR))
        scaled_rows.append(scaled)

    W = max(sum(im.size[0] for im in r) for r in scaled_rows)
    H = sum(max(im.size[1] for im in r) for r in scaled_rows)
    canvas = Image.new("L", (W, H), color=0)

    y = 0
    for r in scaled_rows:
        x = 0
        row_h = max(im.size[1] for im in r)
        for im in r:
            canvas.paste(im, (x, y))
            x += im.size[0]
        y += row_h
    return canvas

def infer_case_name(path: Path) -> str:
    return path.parent.name

def match_by_keys(files, keymap):
    out = defaultdict(list)
    for f in files:
        name = f.name
        for k, rgx in keymap.items():
            if rgx.search(name):
                out[k].append(f)
    return out

def find_nifti_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and is_nifti(p)]

# ----------------------------
# CT: paired CT + CTC
# ----------------------------

from PIL import Image
import numpy as np
def pad_to_tile(img: Image.Image, target_size: tuple) -> Image.Image:
    """Pad an image to target size with zeros, preserving aspect ratio."""
    w, h = img.size
    tw, th = target_size
    
    # Scale image while keeping aspect ratio
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Create black canvas
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    
    # Paste centered
    offset_x = (tw - new_w) // 2
    offset_y = (th - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    
    return canvas



def process_ct_case(ct_path: Path, ctc_path: Path, out_dir: Path, case_name: str):
    try:
        ct_img  = load_ras_img(ct_path)
        ctc_img = load_ras_img(ctc_path)
    except Exception as e:
        print(f"[CT] Load fail {ct_path} or {ctc_path}: {e}", file=sys.stderr)
        return

    # Resample CTC to CT’s grid if needed (shape/affine mismatch)
    if ctc_img.shape != ct_img.shape or not np.allclose(ctc_img.affine, ct_img.affine):
        try:
            ctc_img = resample_from_to(ctc_img, (ct_img.shape, ct_img.affine), order=1)  # linear
        except Exception as e:
            print(f"[CT] Resample fail {case_name}: {e}", file=sys.stderr)
            return

    ct_u8  = get_data_u8(ct_img,  p_low=0, p_high=100)
    ctc_u8 = get_data_u8(ctc_img, p_low=0, p_high=100)

    print("ct_u8:", ct_u8.shape, ctc_u8.shape)

    ax_ct, co_ct, sa_ct = extract_middle_slices_xyz(ct_u8)
    ax_cc, co_cc, sa_cc = extract_middle_slices_xyz(ctc_u8)



    # Label tiles so you know what you’re seeing
    tile_size = (256, 256)

    row1 = [
        label_tile(pad_to_tile(to_pil(ax_ct), tile_size), "Axial CT"),
        label_tile(pad_to_tile(to_pil(ax_cc), tile_size), "Axial CTC")
    ]
    row2 = [
        label_tile(pad_to_tile(to_pil(co_ct), tile_size), "Coronal CT"),
        label_tile(pad_to_tile(to_pil(co_cc), tile_size), "Coronal CTC")
    ]
    row3 = [
        label_tile(pad_to_tile(to_pil(sa_ct), tile_size), "Sagittal CT"),
        label_tile(pad_to_tile(to_pil(sa_cc), tile_size), "Sagittal CTC")
    ]

    grid = make_grid([row1, row2, row3], tile_h=tile_size[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_name}.jpg"
    grid.save(out_path, quality=92)
    print(f"[CT] Saved {out_path}")





def run_ct(data_dir: Path, out_dir: Path):
    for sub in CT_SUBFOLDERS:
        root = data_dir / sub
        if not root.exists():
            print(f"[CT] Missing folder: {root}", file=sys.stderr)
            continue

        case_to_files = defaultdict(list)
        for f in find_nifti_files(root):
            case_to_files[f.parent].append(f)

        for case_dir, files in case_to_files.items():
            matched = match_by_keys(files, CT_KEYS)
            if not matched.get("ct") or not matched.get("ctc"):
                continue
            ct_path  = sorted(matched["ct"])[0]
            ctc_path = sorted(matched["ctc"])[0]
            case_name = infer_case_name(ct_path)
            process_ct_case(ct_path, ctc_path, out_dir, case_name)

# ----------------------------
# Optional MRI viewer (unchanged logic, but RAS + correct axes)
# ----------------------------
def process_mri_case(seq_paths: dict, out_dir: Path, case_name: str):
    order = ["t1", "t1gd", "t2", "flair"]
    tiles = []
    for k in order:
        if k in seq_paths:
            try:
                img = load_ras_img(sorted(seq_paths[k])[0])
                vol_u8 = get_data_u8(img)
                ax, _, _ = extract_middle_slices_xyz(vol_u8)
                tiles.append(label_tile(to_pil(ax), k.upper()))
            except Exception as e:
                print(f"[MRI] Failed {case_name} {k}: {e}", file=sys.stderr)

    if not tiles:
        return
    rows = [tiles[i:i+2] for i in range(0, len(tiles), 2)]
    grid = make_grid(rows, tile_h=256)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_name}_mri.jpg"
    grid.save(out_path, quality=92)
    print(f"[MRI] Saved {out_path}")

def run_mri(data_dir: Path, out_dir: Path):
    for sub in Brain_MR_SUBFOLDERS:
        root = data_dir / sub
        if not root.exists():
            print(f"[MRI] Missing folder: {root}", file=sys.stderr)
            continue

        case_to_files = defaultdict(list)
        for f in find_nifti_files(root):
            case_to_files[f.parent].append(f)

        for case_dir, files in case_to_files.items():
            matched = match_by_keys(files, Brain_MR_KEYS)
            seq_paths = {k: v for k, v in matched.items() if k in {"t1", "t1gd", "t2", "flair"}}
            if not seq_paths:
                continue
            process_mri_case(seq_paths, out_dir, case_dir.name)

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Middle-slice montages with proper orientation; CT: paired CT|CTC per plane; MRI: sequences."
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Root data directory")
    parser.add_argument("--out_dir", type=Path, default=Path("vis"), help="Output directory")
    parser.add_argument("--modality", choices=["ct", "mri", "both"], default="ct",
                        help="Process CT (paired), MRI, or both")
    args = parser.parse_args()

    if args.modality in ("ct", "both"):
        run_ct(args.data_dir, args.out_dir)

    if args.modality in ("mri", "both"):
        run_mri(args.data_dir, args.out_dir)

if __name__ == "__main__":
    main()
