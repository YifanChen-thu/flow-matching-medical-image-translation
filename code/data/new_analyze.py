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
            s.append(float(v))
        except Exception:
            s.append(1.0)
    s = [abs(v) if (v and np.isfinite(v)) else 1.0 for v in s]

    idxD = max(range(3), key=lambda i: (s[i], -n[i]))
    inplane = [i for i in range(3) if i != idxD]
    if n[inplane[0]] >= n[inplane[1]]:
        idxH, idxW = inplane[0], inplane[1]
    else:
        idxH, idxW = inplane[1], inplane[0]

    return (n[idxH], n[idxW], n[idxD]), (s[idxH], s[idxW], s[idxD]), (idxH, idxW, idxD)

from nibabel.processing import resample_to_output

# ----------------------------
# Read header (now returns ORIGINAL pixel sizes too)
# ----------------------------
def read_hwd(path: Path, target_spacing=(2.0, 2.0, 2.0)):
    """Read NIfTI header and return shape/zooms after *virtual* resampling to target spacing,
    **and** the original (canonicalized) H/W/D and spacings.

    Returns
    -------
    H, W, D, sH, sW, sD, extra_dims_str,
    OrigH, OrigW, OrigD, OrigSpacingH, OrigSpacingW, OrigSpacingD
    """
    img = nib.load(str(path), mmap=True)
    shape = np.array(img.header.get_data_shape()[:3], dtype=float)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)

    # Canonical mapping from the *original* header
    (H0, W0, D0), (sH0, sW0, sD0), _ = canonical_hwd_from_header(shape[:3], zooms[:3])

    target_spacing = np.array(target_spacing, dtype=float)

    print(f"Before: {path} : shape={tuple(shape.astype(int))}, voxel size={tuple(zooms)}")

    if shape.size < 3:
        raise ValueError(f"Not a 3D/4D NIfTI: {path} shape={shape}, voxel size={zooms}")

    # Compute new shape (virtual resample, preserves axis order)
    new_shape = np.rint(shape * (zooms / target_spacing)).astype(int)
    new_zooms = target_spacing

    print(f"After (virtual resample): shape={tuple(new_shape)}, voxel size={tuple(new_zooms)}")

    (H, W, D) = new_shape
    (sH, sW, sD) = new_zooms

    extras = shape[3:] if len(shape) > 3 else ()

    return (
        int(H), int(W), int(D), float(sH), float(sW), float(sD),
        ("x".join(map(str, extras)) if extras else ""),
        int(H0), int(W0), int(D0), float(sH0), float(sW0), float(sD0),
    )

# ----------------------------
# Indexers
# ----------------------------
def index_mr_dataset(dataset_dir: Path, dataset_name: str):
    exams = defaultdict(lambda: {
        "Dataset": dataset_name,
        "Subject": "",
        "ExamID": "",
        "Splits": set(),
        "T1": "", "T1Gd": "", "T2": "", "FLAIR": "",
        "Mask": "", "Mask_Correct": ""
    })
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        if not (str(p).endswith(".nii") or str(p).endswith(".nii.gz")):
            continue
        subj = subject_id_from_path(p)
        sp = split_from_path(p)
        mod, exam_id = mr_exam_key_from_file(p)
        if mod is None:
            continue
        key = (subj, exam_id)
        ex = exams[key]
        ex["Dataset"] = dataset_name
        ex["Subject"] = subj
        ex["ExamID"] = exam_id
        if sp:
            ex["Splits"].add(sp)
        current = ex.get(mod.upper() if mod in {"t1", "t1gd", "t2", "flair"} else ("Mask_Correct" if mod == "mask_correct" else "Mask"), "")
        new = str(p)
        if not current or (current.endswith(".nii") and new.endswith(".nii.gz")):
            if mod in {"t1", "t1gd", "t2", "flair"}:
                ex[mod.upper()] = new
            elif mod == "mask_correct":
                ex["Mask_Correct"] = new
            elif mod == "mask":
                ex["Mask"] = new
    rows = []
    for (_, _), ex in sorted(exams.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ex["Splits"] = ",".join(sorted(ex["Splits"]))
        rows.append(ex)
    return rows

def index_ct_dataset(dataset_dir: Path, dataset_name: str):
    """
    Build rows per (subject, exam_id) with paired CT/CTC paths on the same row.
    """
    exams = defaultdict(lambda: {
        "Dataset": dataset_name,
        "Subject": "",
        "ExamID": "",
        "Splits": set(),
        "CT": "", "CTC": ""
    })
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        if not (str(p).endswith(".nii") or str(p).endswith(".nii.gz")):
            continue
        if CT_EXCLUDE.search(p.name):
            continue
        subj = subject_id_from_path(p)
        sp = split_from_path(p)
        mod, exam_id = ct_exam_key_from_file(p)
        if mod not in {"ct", "ctc"}:
            mod = "ct"
        key = (subj, exam_id)
        ex = exams[key]
        ex["Dataset"] = dataset_name
        ex["Subject"] = subj
        ex["ExamID"] = exam_id
        if sp:
            ex["Splits"].add(sp)
        field = "CT" if mod == "ct" else "CTC"
        current = ex[field]
        new = str(p)
        if not current or (current.endswith(".nii") and new.endswith(".nii.gz")):
            ex[field] = new
    rows = []
    for (_, _), ex in sorted(exams.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ex["Splits"] = ",".join(sorted(ex["Splits"]))
        rows.append(ex)
    return rows

# ----------------------------
# Collect CT pairs / MR (unchanged)
# ----------------------------
def collect_ct_pairs(data_dir: Path):
    rows = []
    for ds in CT_SUBFOLDERS:
        ds_path = data_dir / ds
        if not ds_path.exists():
            print(f"[WARN] Not found: {ds_path}")
            continue
        idx_rows = index_ct_dataset(ds_path, ds)
        for r in idx_rows:
            rep_path = Path(r["CT"] or r["CTC"])
            rows.append({
                "Dataset": r["Dataset"],
                "OrganGroup": infer_organ_group_from_dataset(ds),
                "Organ": infer_organ_from_filename(ds, rep_path) if rep_path else infer_organ_group_from_dataset(ds),
                "Subject": r["Subject"],
                "ExamID": r["ExamID"],
                "Split": r["Splits"],
                "CT": r["CT"],
                "CTC": r["CTC"],
            })
    return rows

# ----------------------------
# Size comparison helpers
# ----------------------------
def compare_sizes(primary_hwd, secondary_hwd):
    """
    Compare two (H,W,D) triplets.
    Returns:
      match_exact: bool
      match_inplane_swap: bool (D equal and {H,W} equal ignoring order)
      diffs: (dH, dW, dD) chosen to minimize total abs diff (we can swap H/W on secondary)
    """
    H1, W1, D1 = map(int, primary_hwd)
    H2, W2, D2 = map(int, secondary_hwd)

    match_exact = (H1 == H2 and W1 == W2 and D1 == D2)
    match_inplane_swap = (D1 == D2) and ({H1, W1} == {H2, W2})

    diffs_no_swap = (H1 - H2, W1 - W2, D1 - D2)
    diffs_swap = (H1 - W2, W1 - H2, D1 - D2)
    diffs = diffs_swap if sum(map(abs, diffs_swap)) < sum(map(abs, diffs_no_swap)) else diffs_no_swap

    return match_exact, match_inplane_swap, diffs

# ----------------------------
# Analysis (CT pairs)
# ----------------------------
def analyze_ct_pairs(data_dir: Path, out_csv: Path, plots_dir: Path, target_res: float = None):
    pairs = collect_ct_pairs(data_dir)
    results = []
    errors = []

    for row in tqdm(pairs, desc="Reading CT/CTC headers"):
        ct_path = Path(row["CT"]) if row["CT"] else None
        ctc_path = Path(row["CTC"]) if row["CTC"] else None

        try:
            # Primary (record size once): prefer CT, else CTC
            if ct_path and ct_path.exists():
                (
                    H, W, D, sH, sW, sD, extra,
                    H0, W0, D0, sH0, sW0, sD0,
                ) = read_hwd(ct_path)
                primary_mod = "CT"
            elif ctc_path and ctc_path.exists():
                (
                    H, W, D, sH, sW, sD, extra,
                    H0, W0, D0, sH0, sW0, sD0,
                ) = read_hwd(ctc_path)
                primary_mod = "CTC"
            else:
                raise FileNotFoundError("Both CT and CTC paths missing")

            # Optional isotropic counts (no resampling)
            H_iso = W_iso = D_iso = TotalVoxels_iso = None
            if target_res is not None and target_res > 0:
                L_H, L_W, L_D = H * sH, W * sW, D * sD
                H_iso = int(round(L_H / target_res))
                W_iso = int(round(L_W / target_res))
                D_iso = int(round(L_D / target_res))
                TotalVoxels_iso = int(H_iso * W_iso * D_iso)

            # Compare with secondary
            Pair = bool(ct_path and ct_path.exists() and ctc_path and ctc_path.exists())
            SizeMatch = MatchExact = MatchInplaneSwap = None
            DiffH = DiffW = DiffD = None
            if Pair:
                sec_path = ctc_path if primary_mod == "CT" else ct_path
                H2, W2, D2, *_ = read_hwd(sec_path)
                MatchExact, MatchInplaneSwap, (DiffH, DiffW, DiffD) = compare_sizes((H, W, D), (H2, W2, D2))
                SizeMatch = bool(MatchExact or MatchInplaneSwap)

            rec = {
                **row,
                "HasCT": bool(ct_path and ct_path.exists()),
                "HasCTC": bool(ctc_path and ctc_path.exists()),
                "Pair": Pair,
                "PrimaryMod": primary_mod,
                # Record resampled size ONCE (from primary)
                "H": H, "W": W, "D": D,
                "SpacingH": sH, "SpacingW": sW, "SpacingD": sD,
                "ExtraDims": extra,
                "TotalVoxels": int(H * W * D),
                # Original (from header, canonicalized)
                "OrigH": H0, "OrigW": W0, "OrigD": D0,
                "OrigSpacingH": sH0, "OrigSpacingW": sW0, "OrigSpacingD": sD0,
                "OrigTotalVoxels": int(H0 * W0 * D0),
                # Optional iso
                "TargetRes_mm": target_res if target_res else "",
                "H_iso": H_iso if H_iso is not None else "",
                "W_iso": W_iso if W_iso is not None else "",
                "D_iso": D_iso if D_iso is not None else "",
                "TotalVoxels_iso": TotalVoxels_iso if TotalVoxels_iso is not None else "",
                # Pair comparison
                "SizeMatch": SizeMatch if SizeMatch is not None else "",
                "MatchExact": MatchExact if MatchExact is not None else "",
                "MatchInplaneSwap": MatchInplaneSwap if MatchInplaneSwap is not None else "",
                "DiffH": DiffH if DiffH is not None else "",
                "DiffW": DiffW if DiffW is not None else "",
                "DiffD": DiffD if DiffD is not None else "",
            }
            results.append(rec)

        except Exception as e:
            errors.append((row.get("CT",""), row.get("CTC",""), str(e)))

    if errors:
        print(f"[WARN] {len(errors)} pairs failed. First few:")
        for a, b, msg in errors[:10]:
            print(f"  - CT: {a}  CTC: {b}  -> {msg}")

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote CT pair CSV: {out_csv} ({len(df)} rows)")

    # --------------- Plots (per exam, not per modality) ---------------
    if df.empty:
        print("[INFO] No data to plot.")
        return df

    plots_dir.mkdir(parents=True, exist_ok=True)

    organs = sorted(df["Organ"].dropna().unique().tolist())

    # Histograms for H/W/D (colored by organ)
    def save_hist_by_organ(col, bins=50, suffix=""):
        plt.figure()
        for org in organs:
            vals = df.loc[df["Organ"] == org, col].dropna().astype(float).values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=bins, alpha=0.5, label=org)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"CT pairs: {col} distribution by organ")
        plt.legend(title="Organ", fontsize="small")
        fig_path = plots_dir / f"hist_{col}_by_organ{suffix}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"[OK] Saved {fig_path}")

    for col in ["H", "W", "D"]:
        save_hist_by_organ(col, bins=50)

    # NEW: Box distributions for H, W, D respectively (per organ)
    def save_box_metric_by_organ(col, suffix=""):
        data, labels = [], []
        for org in organs:
            sub = df[df["Organ"] == org]
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            data.append(vals.values)
            labels.append(org)
        if data:
            plt.figure()
            plt.boxplot(data, labels=labels, vert=True, showfliers=False)
            plt.ylabel(col)
            plt.title(f"CT pairs: {col} by organ")
            plt.xticks(rotation=30, ha="right")
            outp = plots_dir / f"box_{col}_by_organ{suffix}.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=160)
            plt.close()
            print(f"[OK] Saved {outp}")

    for col in ["H", "W", "D"]:
        # keep your existing boxplot per organ
        save_box_metric_by_organ(col)

        # nicely formatted per-dataset summary for this metric
        print(f"\n{col} ranges by Dataset:")

        # collect stats per dataset
        stats = []
        for ds, g in df.groupby("Dataset", dropna=False):
            ds_name = "Unknown" if pd.isna(ds) else str(ds)
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if vals.empty:
                continue
            stats.append({
                "Dataset": ds_name,
                "n": int(vals.count()),
                "min": float(vals.min()),
                "p25": float(vals.quantile(0.25)),
                "median": float(vals.quantile(0.50)),
                "p75": float(vals.quantile(0.75)),
                "max": float(vals.max()),
            })

        if not stats:
            print("  (no data)")
            continue

        # compute column widths for alignment
        ds_w = max(len(s["Dataset"]) for s in stats)
        n_w  = max(len(str(s["n"])) for s in stats)
        def _fmt(x):
            if isinstance(x, float) and x.is_integer():
                return str(int(x))
            return f"{x:.2f}" if isinstance(x, float) else str(x)
        num_w = max(3, max(len(_fmt(s[k])) for s in stats for k in ("min", "p25", "median", "p75", "max")))

        # header
        header = (
            f"{'Dataset':<{ds_w}}  "
            f"{'n':>{n_w}}  "
            f"{'min':>{num_w}}  "
            f"{'p25':>{num_w}}  "
            f"{'med':>{num_w}}  "
            f"{'p75':>{num_w}}  "
            f"{'max':>{num_w}}"
        )
        print(header)
        print("-" * len(header))

        # rows sorted by dataset name
        for s in sorted(stats, key=lambda x: x["Dataset"].lower()):
            print(
                f"{s['Dataset']:<{ds_w}}  "
                f"{s['n']:>{n_w}d}  "
                f"{_fmt(s['min']):>{num_w}}  "
                f"{_fmt(s['p25']):>{num_w}}  "
                f"{_fmt(s['median']):>{num_w}}  "
                f"{_fmt(s['p75']):>{num_w}}  "
                f"{_fmt(s['max']):>{num_w}}"
            )
        print("\n")

    # Also TotalVoxels and (optionally) isotropic versions
    def save_box(col="TotalVoxels", suffix=""):
        data, labels = [], []
        for org in organs:
            sub = df[df["Organ"] == org]
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            data.append(vals.values)
            labels.append(org)
        if data:
            plt.figure()
            plt.boxplot(data, labels=labels, vert=True, showfliers=False)
            plt.ylabel(col)
            plt.title(f"CT pairs: {col} by organ")
            plt.xticks(rotation=30, ha="right")
            outp = plots_dir / f"box_{col}_by_organ{suffix}.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=160)
            plt.close()
            print(f"[OK] Saved {outp}")

    save_box("TotalVoxels")
    if "H_iso" in df.columns and (df["H_iso"] != "").any():
        for col in ["H_iso", "W_iso", "D_iso"]:
            save_hist_by_organ(col, bins=50, suffix="_iso")
            save_box_metric_by_organ(col, suffix="_iso")
        if "TotalVoxels_iso" in df.columns:
            save_box("TotalVoxels_iso", suffix="_iso")

    # ----------------------------
    # NEW: distributions of ORIGINAL spatial pixel sizes (mm)
    # ----------------------------
    for col in ["OrigSpacingH", "OrigSpacingW", "OrigSpacingD"]:
        save_hist_by_organ(col, bins=50, suffix="_orig")
        save_box_metric_by_organ(col, suffix="_orig")

    # Mismatch summary
    mism = df[df["Pair"] & (df["SizeMatch"] == False)]
    print(f"[INFO] Pairs total: {int(df['Pair'].sum())}, mismatched shapes: {len(mism)}")
    if not mism.empty:
        mism.to_csv(plots_dir / "mismatched_pairs.csv", index=False)
        print(f"[OK] Saved mismatched pair list -> {plots_dir / 'mismatched_pairs.csv'}")

    # 3D scatter per exam colored by organ (based on resampled HWD)
    sample = df.sample(min(5000, len(df)), random_state=0) if len(df) > 5000 else df
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for org in organs:
        sub = sample[sample["Organ"] == org]
        if len(sub) == 0:
            continue
        ax.scatter(sub["H"], sub["W"], sub["D"], s=6, alpha=0.7, label=org)
    ax.set_xlabel("H (in-plane)")
    ax.set_ylabel("W (in-plane)")
    ax.set_zlabel("D (through-plane)")
    ax.set_title(f"CT pairs: H×W×D by organ (n={len(sample)})")
    ax.legend(title="Organ", loc="upper left", fontsize="small")
    fig_path = plots_dir / "scatter3d_HWD_by_organ.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"[OK] Saved {fig_path}")

    return df

# ----------------------------
# (Optional) MR analysis (unchanged behavior, now with H/W/D boxplots too + original spacing)
# ----------------------------
def analyze_mr(data_dir: Path, out_csv: Path, plots_dir: Path):
    exams = []
    for ds in Brain_MR_SUBFOLDERS:
        ds_path = data_dir / ds
        if not ds_path.exists():
            print(f"[WARN] Not found: {ds_path}")
            continue
        idx_rows = index_mr_dataset(ds_path, ds)
        for r in idx_rows:
            for mod in ["T1", "T1Gd", "T2", "FLAIR"]:
                if r.get(mod):
                    p = Path(r[mod])
                    try:
                        (
                            H, W, D, sH, sW, sD, extra,
                            H0, W0, D0, sH0, sW0, sD0,
                        ) = read_hwd(p)
                        exams.append({
                            "Dataset": r["Dataset"],
                            "OrganGroup": infer_organ_group_from_dataset(ds),
                            "Organ": infer_organ_from_filename(ds, p),
                            "Subject": r["Subject"],
                            "ExamID": r["ExamID"],
                            "Split": r["Splits"],
                            "Modality": mod,
                            "Path": str(p),
                            # resampled
                            "H": H, "W": W, "D": D,
                            "SpacingH": sH, "SpacingW": sW, "SpacingD": sD,
                            # original
                            "OrigH": H0, "OrigW": W0, "OrigD": D0,
                            "OrigSpacingH": sH0, "OrigSpacingW": sW0, "OrigSpacingD": sD0,
                            "ExtraDims": extra,
                            "TotalVoxels": int(H * W * D),
                            "OrigTotalVoxels": int(H0 * W0 * D0),
                        })
                    except Exception as e:
                        print(f"[WARN] MR read failed: {p} -> {e}")
    df = pd.DataFrame(exams)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote MR CSV: {out_csv} ({len(df)} rows)")

    if df.empty:
        return df

    plots_dir.mkdir(parents=True, exist_ok=True)
    organs = sorted(df["Organ"].dropna().unique().tolist())

    def save_hist_by_organ(col, bins=50):
        plt.figure()
        for org in organs:
            vals = pd.to_numeric(df.loc[df["Organ"] == org, col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=bins, alpha=0.5, label=org)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"MR {col} by organ")
        plt.legend(title="Organ", fontsize="small")
        fig_path = plots_dir / f"hist_{col}_by_organ.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"[OK] Saved {fig_path}")

    def save_box_metric_by_organ(col):
        data, labels = [], []
        for org in organs:
            sub = df[df["Organ"] == org]
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            data.append(vals.values)
            labels.append(org)
        if data:
            plt.figure()
            plt.boxplot(data, labels=labels, vert=True, showfliers=False)
            plt.ylabel(col)
            plt.title(f"MR {col} by organ")
            plt.xticks(rotation=30, ha="right")
            outp = plots_dir / f"box_{col}_by_organ.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=160)
            plt.close()
            print(f"[OK] Saved {outp}")
            

    for col in ["H", "W", "D"]:
        save_hist_by_organ(col, bins=50)
        save_box_metric_by_organ(col)
        print("")

    # NEW: distributions of ORIGINAL spatial pixel sizes (mm) for MR
    for col in ["OrigSpacingH", "OrigSpacingW", "OrigSpacingD"]:
        save_hist_by_organ(col, bins=50)
        save_box_metric_by_organ(col)

    return df

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "CT: pair CT/CTC on same row, compare sizes, record size once; MR unchanged. "
            "Robust H/W/D mapping. Adds box distributions for H, W, D. "
            "Now also records and plots ORIGINAL pixel sizes (spacing) per axis."
        )
    )
    parser.add_argument("--task", "-t", required=True, choices=["CT", "MR"], help="Choose modality.")
    parser.add_argument("--data_dir", "-d", default=str(DATA_DIR), help="Root data directory.")
    parser.add_argument("--out_csv", "-o", default=None, help="Output CSV path (default: code/data/<TASK>_sizes.csv)")
    parser.add_argument("--plots_dir", "-p", default=None, help="Directory for plots (default: code/plots/<TASK>_sizes/)")
    parser.add_argument("--target_res", type=float, default=None, help="Optional isotropic resolution (mm) for predicted counts (CT only).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_csv = Path(args.out_csv) if args.out_csv else Path(f"code/data/{args.task}_sizes.csv")
    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(f"code/plots/{args.task}_sizes")

    if args.task == "CT":
        analyze_ct_pairs(data_dir, out_csv, plots_dir, target_res=args.target_res)
    else:
        analyze_mr(data_dir, out_csv, plots_dir)

if __name__ == "__main__":
    main()
