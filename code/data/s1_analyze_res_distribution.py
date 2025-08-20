#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

# Exclude obvious mask/label files in CT
CT_EXCLUDE = re.compile(r"(mask|label?)", re.IGNORECASE)  # |seg(mentation)

# ---- Organ inference rules ----
ORGAN_PATTERNS = {
    "adrenal": re.compile(r"adrenal", re.IGNORECASE),
    "bladder": re.compile(r"bladder", re.IGNORECASE),
    "kidney": re.compile(r"kidney|renal", re.IGNORECASE),
    "lung": re.compile(r"lung", re.IGNORECASE),
    "stomach": re.compile(r"stomach|gastric", re.IGNORECASE),
    "colon": re.compile(r"colon|colorectal|bowel", re.IGNORECASE),
    "liver": re.compile(r"liver|hepatic", re.IGNORECASE),
    "pancreas": re.compile(r"pancreas|pancreatic", re.IGNORECASE),
    "uterus": re.compile(r"uterus|uterine|endometrium", re.IGNORECASE),
    "ovary": re.compile(r"ovary|ovarian", re.IGNORECASE),
    "breast": re.compile(r"breast|mamm", re.IGNORECASE),
    "brain": re.compile(r"brain|intracranial|gli", re.IGNORECASE),
}

# --- Exam-key extraction ---
CT_EXAM_RX = re.compile(r"^(?P<prefix>.*?)(?:[_\-\.])(?P<mod>ctc|ct)$", re.IGNORECASE)

# ----------------------------
# Helper functions
# ----------------------------
def subject_id_from_path(p: Path) -> str:
    parts_low = [part.lower() for part in p.parts]
    for i, part in enumerate(parts_low):
        if part in {"train", "val", "test"}:
            if i + 1 < len(p.parts):
                return p.parts[i + 1]
    return p.parent.name

def split_from_path(p: Path) -> str:
    for part in p.parts:
        low = part.lower()
        if low in {"train", "val", "test"}:
            return low
    return ""

def ensure_path(path_like) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)

def infer_organ_group_from_dataset(ds_name: str) -> str:
    low = ds_name.lower()
    for organ, rx in ORGAN_PATTERNS.items():
        if rx.search(low):
            return organ
    return low

def infer_organ_from_filename(ds_name: str, file_path: Path) -> str:
    name = file_path.name.lower()
    for organ, rx in ORGAN_PATTERNS.items():
        if rx.search(name):
            return organ
    return infer_organ_group_from_dataset(ds_name)

def strip_mr_modality(stem: str) -> str:
    for k, rx in Brain_MR_KEYS.items():
        m = rx.search(stem)
        if m:
            start, end = m.span()
            new = (stem[:start] + stem[end:]).strip("_-.")
            return new if new else stem
    return stem

def ct_exam_key_from_file(file_path: Path):
    """
    Return (mod, exam_id) for CT. If no 'ct'/'ctc' token, fallback to 'ct' to include scans like Uterus/Ovary.
    """
    name = file_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = file_path.stem
    m = CT_EXAM_RX.match(stem)
    if m:
        return m.group("mod").lower(), m.group("prefix")
    for k, rx in CT_KEYS.items():
        if rx.search(stem):
            prefix = rx.sub("", stem).strip("_-.")
            return k, prefix
    return "ct", stem  # fallback

def mr_exam_key_from_file(file_path: Path):
    name = file_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = file_path.stem
    exam = strip_mr_modality(stem) or stem
    mod = None
    for k, rx in Brain_MR_KEYS.items():
        if rx.search(stem):
            mod = k
            break
    return mod, exam

# ----------------------------
# Robust H/W/D decision
# ----------------------------
def canonical_hwd_from_header(shape3, zooms3):
    """
    Robustly decide H, W, D from (shape, spacing):
      - D = axis with largest spacing (through-plane).
      - H/W = remaining axes; H = axis with larger voxel count, W = the other.
    Returns: (H,W,D counts), (sH,sW,sD spacings), (idxH,idxW,idxD)
    """
    n = list(map(int, shape3))
    z = list(zooms3)
    s = []
    for v in z:
        try:
            s.append(