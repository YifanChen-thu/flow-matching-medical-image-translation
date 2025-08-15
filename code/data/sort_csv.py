#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

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

# ----------------------------
# Helpers
# ----------------------------
def subject_id_from_path(p: Path) -> str:
    """
    Handles:
      /.../train/<SUBJECT>/file.nii.gz
      /.../train/<SUBJECT>/<Some Info>/file.nii.gz
    Returns the <SUBJECT> folder name in both cases.
    """
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

def write_csv(rows, header, out_path):
    out_path = ensure_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# --- Exam-key extraction ---
# e.g. "TCGA-G3-AAUZ_2005-11-08_CT.nii.gz" -> ("ct", "TCGA-G3-AAUZ_2005-11-08")
CT_EXAM_RX = re.compile(r"^(?P<prefix>.*?)(?:[_\-\.])(?P<mod>ctc|ct)$", re.IGNORECASE)

# For MR, we strip the matched modality token and use what's left as an exam id (often SUBJECT_DATE).
def strip_mr_modality(stem: str) -> str:
    # try to remove exactly one modality token at end or before separators
    for k, rx in Brain_MR_KEYS.items():
        m = rx.search(stem)
        if m:
            start, end = m.span()
            # remove token and trailing separators around it
            new = (stem[:start] + stem[end:]).strip("_-.")
            return new if new else stem
    return stem

def ct_exam_key_from_file(file_path: Path):
    stem = file_path.stem  # preserves ".nii" if .nii.gz; better use name sans suffixes:
    name = file_path.name
    # drop extensions .nii or .nii.gz
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = file_path.stem
    m = CT_EXAM_RX.match(stem)
    if m:
        return m.group("mod").lower(), m.group("prefix")
    # Fallback: try to detect any CT/CTC token anywhere
    for k, rx in CT_KEYS.items():
        if rx.search(stem):
            prefix = rx.sub("", stem).strip("_-.")
            return k, prefix
    return None, stem

def mr_exam_key_from_file(file_path: Path):
    name = file_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = file_path.stem
    exam = strip_mr_modality(stem)
    # If exam collapsed to empty, fallback to stem
    exam = exam if exam else stem
    # Also return which modality we detected
    mod = None
    for k, rx in Brain_MR_KEYS.items():
        if rx.search(stem):
            mod = k
            break
    return mod, exam

# ----------------------------
# Indexers (now group by exam)
# ----------------------------
def index_mr_dataset(dataset_dir: Path, dataset_name: str):
    """
    Build rows per (subject, exam_id).
    Returns rows and stats.
    """
    # dict[(subject, exam_id)] -> dict of fields
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
            continue  # skip unknown files

        key = (subj, exam_id)
        ex = exams[key]
        ex["Dataset"] = dataset_name
        ex["Subject"] = subj
        ex["ExamID"] = exam_id
        if sp:
            ex["Splits"].add(sp)

        # prefer .nii.gz over .nii if duplicate
        current = ex.get(mod.upper() if mod in {"t1", "t1gd", "t2", "flair"} else ("Mask_Correct" if mod == "mask_correct" else "Mask"), "")
        new = str(p)
        if not current or (current.endswith(".nii") and new.endswith(".nii.gz")):
            if mod in {"t1", "t1gd", "t2", "flair"}:
                ex[mod.upper()] = new
            elif mod == "mask_correct":
                ex["Mask_Correct"] = new
            elif mod == "mask":
                ex["Mask"] = new

    # finalize rows
    rows = []
    complete = 0
    with_mask = 0
    for (_, _), ex in sorted(exams.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ex["Splits"] = ",".join(sorted(ex["Splits"]))
        rows.append(ex)
        if ex["T1"] and ex["T1Gd"] and ex["T2"] and ex["FLAIR"]:
            complete += 1
        if ex["Mask"] or ex["Mask_Correct"]:
            with_mask += 1

    stats = {
        "subjects": len(set(k[0] for k in exams.keys())),
        "exams": len(exams),
        "complete_exams_4mods": complete,
        "exams_with_any_mask": with_mask,
    }
    return rows, stats

def index_ct_dataset(dataset_dir: Path, dataset_name: str):
    """
    Build rows per (subject, exam_id) where exam_id comes from filename prefix.
    For CT, a 'pair' is an exam that has both CT and CTC.
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

        subj = subject_id_from_path(p)
        sp = split_from_path(p)
        mod, exam_id = ct_exam_key_from_file(p)
        if mod not in {"ct", "ctc"}:
            continue

        key = (subj, exam_id)
        ex = exams[key]
        ex["Dataset"] = dataset_name
        ex["Subject"] = subj
        ex["ExamID"] = exam_id
        if sp:
            ex["Splits"].add(sp)

        # prefer .nii.gz over .nii
        field = "CT" if mod == "ct" else "CTC"
        current = ex[field]
        new = str(p)
        if not current or (current.endswith(".nii") and new.endswith(".nii.gz")):
            ex[field] = new

    rows = []
    pairs = 0
    for (_, _), ex in sorted(exams.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ex["Splits"] = ",".join(sorted(ex["Splits"]))
        rows.append(ex)
        if ex["CT"] and ex["CTC"]:
            pairs += 1

    stats = {
        "subjects": len(set(k[0] for k in exams.keys())),
        "exams": len(exams),
        "pairs_CT_and_CTC": pairs,
    }
    return rows, stats

# ----------------------------
# Driver
# ----------------------------
def run(task: str, out_csv):
    global DATA_DIR
    out_path = ensure_path(out_csv)

    task = task.upper().strip()
    if task not in {"MR", "CT"}:
        raise SystemExit("TASK must be MR or CT")

    subfolders = MR_SUBFOLDERS if task == "MR" else CT_SUBFOLDERS

    all_rows = []
    per_ds_stats = []

    for ds in subfolders:
        ds_path = DATA_DIR / ds
        if not ds_path.exists():
            print(f"[WARN] Not found: {ds_path}")
            continue

        if task == "MR":
            rows, stats = index_mr_dataset(ds_path, ds)
            header = ["Dataset", "Subject", "ExamID", "Splits", "T1", "T1Gd", "T2", "FLAIR", "Mask", "Mask_Correct"]
            print(f"{ds}: subjects={stats['subjects']}, exams={stats['exams']}, complete_exams_4mods={stats['complete_exams_4mods']}, exams_with_any_mask={stats['exams_with_any_mask']}")
        else:
            rows, stats = index_ct_dataset(ds_path, ds)
            header = ["Dataset", "Subject", "ExamID", "Splits", "CT", "CTC"]
            print(f"{ds}: subjects={stats['subjects']}, exams={stats['exams']}, CT+CTC pairs={stats['pairs_CT_and_CTC']}")

        all_rows.extend(rows)
        per_ds_stats.append((ds, stats))

    # write combined only once
    write_csv(all_rows, header, out_path)
    print(f"[OK] Wrote combined CSV: {out_path} with {len(all_rows)} rows")

    # summary
    print("\n=== SUMMARY ===")
    for ds, stats in per_ds_stats:
        if task == "MR":
            print(f"{ds}: subjects={stats['subjects']}, exams={stats['exams']}, complete_4mods={stats['complete_exams_4mods']}, with_any_mask={stats['exams_with_any_mask']}")
        else:
            print(f"{ds}: subjects={stats['subjects']}, exams={stats['exams']}, pairs_CT_and_CTC={stats['pairs_CT_and_CTC']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index MR or CT datasets into CSVs (one row per subject-exam).")
    parser.add_argument("--task", "-t", required=True, choices=["MR", "CT"], help="Choose modality.")
    parser.add_argument("--data_dir", "-d", default=str(DATA_DIR), help="Root data directory (default: /home/yifan/data)")
    parser.add_argument("--out_csv", "-o", default=None, help="Output CSV path (default: code/data/<TASK>_pair.csv)")
    args = parser.parse_args()

    # update DATA_DIR from args
    DATA_DIR = Path(args.data_dir).expanduser().resolve()

    out_csv = args.out_csv or f"code/data/{args.task}_pair.csv"
    run(args.task, out_csv=out_csv)
