#!/usr/bin/env python3
import argparse
import re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from nibabel.processing import resample_to_output

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

# Map the FIVE dataset groups to a single organ key (used for reporting buckets only)
ORG_BY_SUBFOLDER = {
    "Adrenal_CT_train_val_test": "adrenal",
    "Bladder_Kidney_CT_train_val_test": "bladder",
    "Lung_CT_train_val_test": "lung",
    "Stomach_Colon_Liver_Pancreas_CT_train_val_test": "stomach",
    "Uterus_Ovary_CT_train_val_test": "uterus",
}

# Target in-plane size (H, W)
TARGET_INPLANE = (256, 256)

# Z spacing factor relative to H spacing (mm/pixel)
Z_PER_H_FACTOR = 2.5

# Where to save when not in debug mode
OUT_DIR = Path("/date/hao/PairedContrast/CT/low_256x256_2.5xD")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Patterns
# ----------------------------
CT_KEYS = {
    "ct": re.compile(r"(?:^|[_\-\.])ct(?:[_\-\.]|$)", re.IGNORECASE),
    "ctc": re.compile(r"(?:^|[_\-\.])ctc(?:[_\-\.]|$)", re.IGNORECASE),
}
CT_EXCLUDE = re.compile(r"(mask|label?)", re.IGNORECASE)
CT_EXAM_RX = re.compile(r"^(?P<prefix>.*?)(?:[_\-\.])(?P<mod>ctc|ct)$", re.IGNORECASE)

# ----------------------------
# Helpers
# ----------------------------
def is_ct_image(file_path: Path) -> bool:
    name = file_path.name
    return (
        (name.endswith(".nii") or name.endswith(".nii.gz"))
        and any(rx.search(name) for rx in CT_KEYS.values())
        and not CT_EXCLUDE.search(name)
    )

def organ_from_subfolder(path: Path) -> str:
    # Determine which of the five top-level subfolders it belongs to
    for sub in CT_SUBFOLDERS:
        if sub in map(str, path.parents):
            return ORG_BY_SUBFOLDER[sub]
    # Fallback by scanning entire path string
    s = str(path)
    for sub, org in ORG_BY_SUBFOLDER.items():
        if sub in s:
            return org
    return "unknown"

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

def ct_exam_key_from_file(file_path: Path):
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
    return "ct", stem

def plan_target_size_and_spacing(shape_xyz, spacing_xyz):
    """
    Compute new size (x,y,z) and spacing (x,y,z) in nib ordering (x,y,z).
    - Force in-plane size to 256x256.
    - H spacing (mm/pixel) = physical H extent / 256; W spacing analogous.
    - Z spacing = Z_PER_H_FACTOR * (H spacing).
    - Depth slices Z = round(physical Z extent / Z spacing), clamped to >= 1.
    """
    x, y, z = map(int, shape_xyz)
    sx, sy, sz = map(float, spacing_xyz)
    Lx, Ly, Lz = x * sx, y * sy, z * sz  # physical extents in mm

    # In-plane target sizes (W, H) in nib ordering are (x,y)
    new_x = int(TARGET_INPLANE[1])  # W (cols)
    new_y = int(TARGET_INPLANE[0])  # H (rows)

    # Derived in-plane spacings (mm/pixel)
    new_sx = Lx / new_x if new_x > 0 else sx  # along W (cols)
    new_sy = Ly / new_y if new_y > 0 else sy  # along H (rows)

    # Z spacing derived from H spacing
    new_sz = float(Z_PER_H_FACTOR) * float(new_sy)

    # Compute #slices to preserve physical coverage in Z
    new_z = int(round(Lz / new_sz)) if new_sz > 0 else int(z)
    new_z = max(1, new_z)

    return (new_x, new_y, new_z), (new_sx, new_sy, new_sz)

def resample_sitk_from_nib(data_xyz, spacing_xyz, new_size_xyz, new_spacing_xyz):
    """
    Resample using SimpleITK.
    Convert numpy (x,y,z) to SITK (z,y,x) order and back.
    """
    arr_zyx = np.transpose(data_xyz, (2, 1, 0))
    img = sitk.GetImageFromArray(arr_zyx)
    sx, sy, sz = map(float, spacing_xyz)
    img.SetSpacing((sx, sy, sz))

    nx, ny, nz = map(int, new_size_xyz)
    nsx, nsy, nsz = map(float, new_spacing_xyz)
    print("set:", nsx, nsy, nsz )

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetSize((nx, ny, nz))
    resampler.SetOutputSpacing((nsx, nsy, nsz))
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())

    out = resampler.Execute(img)
    out_arr_zyx = sitk.GetArrayFromImage(out)
    return np.transpose(out_arr_zyx, (2, 1, 0))  # back to (x,y,z)

# ----------------------------
# Index and pair cases (CT + CTC)
# ----------------------------
def index_ct_pairs(root: Path):
    """
    Return dict keyed by (split, subject, exam_id) with CT/CTC paths and organ key.
    Each key is ONE case; CT/CTC treated consistently and counted once.
    """
    cases = {}
    for subfolder in CT_SUBFOLDERS:
        folder_path = root / subfolder
        if not folder_path.exists():
            print(f"[WARN] Not found: {folder_path}")
            continue
        organ_key = ORG_BY_SUBFOLDER[subfolder]
        for p in folder_path.rglob("*"):
            if not p.is_file() or not is_ct_image(p):
                continue
            mod, exam_id = ct_exam_key_from_file(p)
            split = split_from_path(p)
            subj = subject_id_from_path(p)
            key = (split, subj, exam_id)
            ex = cases.get(key) or {
                "split": split, "subject": subj, "exam_id": exam_id,
                "organ": organ_key, "ct": None, "ctc": None,
            }
            if mod == "ctc":
                ex["ctc"] = p
            else:
                ex["ct"] = p
            cases[key] = ex
    return cases

# ----------------------------
# Per-case processing
# ----------------------------

def _nib_header_to_dict(img: nib.Nifti1Image) -> dict:
    """Make a JSON-serializable snapshot of a NIfTI header (safe fields only)."""
    hdr = img.header
    out = {
        "shape": tuple(img.shape),
        "dtype": str(img.get_data_dtype()),
        "zoom_mm": tuple(map(float, hdr.get_zooms()[:len(img.shape)])),
        "qform_code": int(hdr["qform_code"]),
        "sform_code": int(hdr["sform_code"]),
        "pixdim": tuple(map(float, hdr["pixdim"][:8])),
    }
    # Store matrices as flat lists (JSON-safe)
    try:
        q = img.get_qform()
        s = img.get_sform()
        if q is not None:
            out["qform"] = np.asarray(q, dtype=float).reshape(-1).tolist()
        if s is not None:
            out["sform"] = np.asarray(s, dtype=float).reshape(-1).tolist()
    except Exception:
        pass
    return out

def _sitk_resample_image(itk_img: sitk.Image, target_size_xyz: tuple, target_spacing_xyz: tuple,
                         interp=sitk.sitkLinear, default_value=0.0) -> sitk.Image:
    """Resample with preserved origin/direction; target_size/spacing are (W,H,D)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(float(default_value))
    # Use the same physical space frame
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    # SimpleITK uses size index order (x,y,z) == (W,H,D)
    resampler.SetSize(tuple(int(x) for x in target_size_xyz))
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing_xyz))
    # Compute a centered output start so that volume stays in place
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    return resampler.Execute(itk_img)

def _nib_to_sitk(img: nib.Nifti1Image) -> sitk.Image:
    """Convert nibabel image to SimpleITK while preserving geometry."""
    arr = np.asanyarray(img.dataobj)  # lazy read
    itk = sitk.GetImageFromArray(arr.astype(np.float32), isVector=False)
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    itk.SetSpacing(tuple(float(x) for x in zooms))
    # Derive origin/direction from sform/qform if available
    affine = img.affine
    R = np.array(affine[:3, :3], dtype=float)
    dirs = []
    for i in range(3):
        v = R[:, i]
        n = np.linalg.norm(v)
        dirs += list((v / n) if n > 0 else [1.0 if j == i else 0.0 for j in range(3)])
    itk.SetDirection(tuple(dirs))
    itk.SetOrigin(tuple(float(x) for x in affine[:3, 3]))
    return itk

def _sitk_to_nib(itk_img: sitk.Image, template_nib: nib.Nifti1Image) -> nib.Nifti1Image:
    """Build a nibabel image using SITK geometry and template header for codes."""
    arr = sitk.GetArrayFromImage(itk_img)  # returns (D,H,W)
    arr = np.asarray(arr)
    # Build affine from spacing, direction, origin
    sp = np.array(itk_img.GetSpacing(), float)      # (W,H,D)
    dir_flat = np.array(itk_img.GetDirection(), float)  # length 9
    R = dir_flat.reshape(3, 3)
    T = np.array(itk_img.GetOrigin(), float)        # (W,H,D)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = R * sp  # scale axes by spacing
    affine[:3, 3] = T
    new_hdr = template_nib.header.copy()
    new_hdr.set_zooms(tuple(sp.tolist()) + ((1.0,),) if arr.ndim == 4 else tuple(sp.tolist()))
    new_img = nib.Nifti1Image(arr, affine=affine, header=new_hdr)
    # Carry over qform/sform codes
    try:
        q = template_nib.get_qform(coded=True)
        s = template_nib.get_sform(coded=True)
        if q is not None and q[1] != 0:
            new_img.set_qform(q[0], int(q[1]))
        if s is not None and s[1] != 0:
            new_img.set_sform(s[0], int(s[1]))
    except Exception:
        pass
    return new_img

def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def process_case(case: dict, out_dir: Path, debug: bool = True):
    """
    Resample CT/CTC to a planned target grid, preserve geometry/metadata,
    and emit provenance JSON to avoid any info loss.

    Returns:
        plan (dict) – includes per-modality input/output and header snapshots.
    """
    # --- 1) Choose reference & inspect ---
    ref_path: Path = case.get("ct") or case.get("ctc")
    if not ref_path:
        raise ValueError("Case has neither CT nor CTC path.")
    ref_img_nib = nib.load(str(ref_path))

    ref_data_shape = ref_img_nib.shape[:3]
    ref_spacing    = ref_img_nib.header.get_zooms()[:3]

    

    # --- 2) Plan target grid ---
    target_size, target_spacing = plan_target_size_and_spacing(ref_data_shape, ref_spacing)

    print("input: ", ref_data_shape, ref_spacing)
    print("target:", target_size, target_spacing)


    
    plan = {
        "case_id": f"{case['split']}::{case['subject']}::{case['exam_id']}",
        "organ": case.get("organ"),
        "target_size": tuple(int(x) for x in target_size),          # (W,H,D)
        "target_spacing": tuple(float(x) for x in target_spacing),  # (W,H,D) mm
        "h_spacing": float(target_spacing[1]),
        "w_spacing": float(target_spacing[0]),
        "z_spacing": float(target_spacing[2]),
        "reference": {
            "path": str(ref_path),
            "header": _nib_header_to_dict(ref_img_nib),
        },
        "outputs": [],  # filled below per modality
    }

    # --- 3) Process each modality present ---
    for mod in ("ct", "ctc"):
        src = case.get(mod)
        if not src:
            continue

        # relative output path mirrors DATA_DIR structure
        rel = Path(src).relative_to(DATA_DIR)
        out_path = (out_dir / rel).with_suffix(".nii.gz")
        out_path_json = out_path.with_suffix(".provenance.json")  # sidecar with rich metadata

        # Gather original info
        src_img_nib = nib.load(str(src))
        src_hdr_dict = _nib_header_to_dict(src_img_nib)
        src_dtype = str(src_img_nib.get_data_dtype())

        # Build output record (even if debug)
        out_rec = {
            "modality": mod.upper(),
            "src_path": str(src),
            "out_path": str(out_path),
            "original": {
                "shape": tuple(src_img_nib.shape[:3]),
                "spacing": tuple(map(float, src_img_nib.header.get_zooms()[:3])),
                "dtype": src_dtype,
                "header": src_hdr_dict,
            },
            "target": {
                "size": plan["target_size"],
                "spacing": plan["target_spacing"],
                "dtype": "float32",
            },
        }

        if not debug:
            # --- Resample with SimpleITK (keeps geometry robustly) ---
            
            img = src_img_nib #nib.load(str(src_img_nib))
            hdr = img.header.copy()

            old_spacing = tuple(float(x) for x in hdr.get_zooms()[:3])

            # Resample to target spacing
            # This adjusts both the data array and the affine for you.
            
            order = 1
            cval  = 0  

            resampled_img = resample_to_output(
                img,
                voxel_sizes=plan["target_spacing"],
                order=order,
                cval=cval,
            )


            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(resampled_img, str(out_path))



            # --- Provenance sidecar (rich, human-readable, no loss) ---
            provenance = {
                "modality": mod.upper(),
                "source_header": src_hdr_dict,
                "source_dtype": src_dtype,
                "planning": {
                    "from_shape": tuple(src_img_nib.shape[:3]),
                    "from_spacing": tuple(map(float, src_img_nib.header.get_zooms()[:3])),
                    "target_size": plan["target_size"],
                    "target_spacing": plan["target_spacing"],
                    "interpolator": "sitkLinear",
                    "default_value": 0.0,
                },
                "output_header": _nib_header_to_dict(nib.load(str(out_path))),
            }
            _write_json(out_path_json, provenance)

        print(f"[SAVED]{' (dry-run)' if debug else ''} {mod.upper()} -> {out_path} depth={plan['target_size'][2]}")
        plan["outputs"].append(out_rec)

    return plan

# ----------------------------
# Summaries
# ----------------------------
def _percentiles(vals):
    """Return (min, p25, median, p75, max) for a 1D list-like of numbers; None if empty."""
    if not vals:
        return (None, None, None, None, None)
    a = np.asarray(vals, dtype=float)
    # numpy >=1.22 uses 'method', older uses 'interpolation'; default works across versions
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    return (int(np.min(a)) if np.allclose(a, np.round(a)) else float(np.min(a)),
            int(p25) if np.allclose(p25, round(p25)) else float(p25),
            int(p50) if np.allclose(p50, round(p50)) else float(p50),
            int(p75) if np.allclose(p75, round(p75)) else float(p75),
            int(np.max(a)) if np.allclose(a, np.round(a)) else float(np.max(a)))

def summarize_depth(plans):
    """
    Summarize resampled depth (number of slices, i.e., target_size[2]) per organ and overall.
    Returns: dict[organ] = {"n", "min", "p25", "median", "p75", "max"}, plus "__overall__".
    """
    by_org = defaultdict(list)
    overall = []

    for plan in plans:
        organ = plan.get("organ") or "unknown"
        # Depth is the 3rd element (W,H,D) -> D
        d = int(plan["target_size"][2])
        by_org[organ].append(d)
        overall.append(d)

    summary = {}
    for org, vals in by_org.items():
        mn, p25, med, p75, mx = _percentiles(vals)
        summary[org] = {
            "n": len(vals),
            "min": mn,
            "p25": p25,
            "median": med,
            "p75": p75,
            "max": mx,
        }

    mn, p25, med, p75, mx = _percentiles(overall)
    summary["__overall__"] = {
        "n": len(overall),
        "min": mn,
        "p25": p25,
        "median": med,
        "p75": p75,
        "max": mx,
    }
    return summary

def summarize_float(plans, key):
    """
    Summarize a float field from each plan (e.g., 'h_spacing', 'w_spacing', 'z_spacing')
    per organ and overall. Returns same structure as summarize_depth.
    """
    by_org = defaultdict(list)
    overall = []

    for plan in plans:
        organ = plan.get("organ") or "unknown"
        v = float(plan[key])
        by_org[organ].append(v)
        overall.append(v)

    summary = {}
    for org, vals in by_org.items():
        if not vals:
            summary[org] = {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
            continue
        a = np.asarray(vals, dtype=float)
        p25, p50, p75 = np.percentile(a, [25, 50, 75])
        summary[org] = {
            "n": len(vals),
            "min": float(np.min(a)),
            "p25": float(p25),
            "median": float(p50),
            "p75": float(p75),
            "max": float(np.max(a)),
        }

    if overall:
        a = np.asarray(overall, dtype=float)
        p25, p50, p75 = np.percentile(a, [25, 50, 75])
        overall_st = {
            "n": len(overall),
            "min": float(np.min(a)),
            "p25": float(p25),
            "median": float(p50),
            "p75": float(p75),
            "max": float(np.max(a)),
        }
    else:
        overall_st = {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}

    summary["__overall__"] = overall_st
    return summary

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resample CT to 256x256 in-plane; set Z spacing = 2.5 × H pixel size. "
            "CT and CTC are paired per (split, subject, exam_id) and treated consistently; stats count ONE per case."
        )
    )
    parser.add_argument("--data_dir", "-d", default=str(DATA_DIR), help="Root data directory")
    parser.add_argument("--out_dir",  "-o", default=str(OUT_DIR),  help="Output directory (used when --no-debug)")
    parser.add_argument("--debug", action="store_true", help="Dry run: only compute plans & stats")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug: actually resample & save")
    args = parser.parse_args()

    root = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    cases = index_ct_pairs(root)
    print(f"[INFO] Indexed cases: {len(cases)} (each counted once in stats)")

    plans = []
    for key, case in tqdm(sorted(cases.items())):
        plan = process_case(case, out_dir, debug=args.debug)
        plans.append(plan)
        if args.debug:
            print(
                f"[PLAN] {plan['case_id']} | organ={plan['organ']} | depth={plan['target_size'][2]} | "
                f"H/W/Z spacing = {plan['h_spacing']:.3f}/{plan['w_spacing']:.3f}/{plan['z_spacing']:.3f} mm"
            )

    # Depth stats
    depth_summary = summarize_depth(plans)

    def fmt(v):   return "-" if v is None else str(v)
    def fmtf(v):  return "-" if v is None else f"{v:.3f}"

    print("\nDepth (#slices) after resampling — by organ (5 groups):")
    header = f"{'organ':<12}  {'n':>6}  {'min':>6}  {'p25':>6}  {'median':>6}  {'p75':>6}  {'max':>6}"
    print(header)
    print("-" * len(header))
    for org in [ORG_BY_SUBFOLDER[s] for s in CT_SUBFOLDERS]:
        st = depth_summary.get(org, {"n":0,"min":None,"p25":None,"median":None,"p75":None,"max":None})
        print(f"{org:<12}  {fmt(st['n']):>6}  {fmt(st['min']):>6}  {fmt(st['p25']):>6}  {fmt(st['median']):>6}  {fmt(st['p75']):>6}  {fmt(st['max']):>6}")
    print("OVERALL:")
    st = depth_summary["__overall__"]
    print(f"n={fmt(st['n'])}, min={fmt(st['min'])}, p25={fmt(st['p25'])}, median={fmt(st['median'])}, p75={fmt(st['p75'])}, max={fmt(st['max'])}")

    # H/W/Z spacing stats (mm/pixel)
    h_summary = summarize_float(plans, "h_spacing")
    w_summary = summarize_float(plans, "w_spacing")
    z_summary = summarize_float(plans, "z_spacing")

    print("\nH pixel spacing (mm) after resampling — by organ:")
    hdr = f"{'organ':<12}  {'n':>6}  {'min':>8}  {'p25':>8}  {'median':>8}  {'p75':>8}  {'max':>8}"
    print(hdr)
    print("-" * len(hdr))
    for org in [ORG_BY_SUBFOLDER[s] for s in CT_SUBFOLDERS]:
        st = h_summary.get(org, {"n":0,"min":None,"p25":None,"median":None,"p75":None,"max":None})
        print(f"{org:<12}  {fmt(st['n']):>6}  {fmtf(st['min']):>8}  {fmtf(st['p25']):>8}  {fmtf(st['median']):>8}  {fmtf(st['p75']):>8}  {fmtf(st['max']):>8}")
    print("OVERALL (H): n={}, min={}, p25={}, median={}, p75={}, max={}".format(
        fmt(h_summary["__overall__"]["n"]),
        fmtf(h_summary["__overall__"]["min"]),
        fmtf(h_summary["__overall__"]["p25"]),
        fmtf(h_summary["__overall__"]["median"]),
        fmtf(h_summary["__overall__"]["p75"]),
        fmtf(h_summary["__overall__"]["max"]),
    ))

    print("\nW pixel spacing (mm) after resampling — by organ:")
    print(hdr)
    print("-" * len(hdr))
    for org in [ORG_BY_SUBFOLDER[s] for s in CT_SUBFOLDERS]:
        st = w_summary.get(org, {"n":0,"min":None,"p25":None,"median":None,"p75":None,"max":None})
        print(f"{org:<12}  {fmt(st['n']):>6}  {fmtf(st['min']):>8}  {fmtf(st['p25']):>8}  {fmtf(st['median']):>8}  {fmtf(st['p75']):>8}  {fmtf(st['max']):>8}")
    print("OVERALL (W): n={}, min={}, p25={}, median={}, p75={}, max={}".format(
        fmt(w_summary["__overall__"]["n"]),
        fmtf(w_summary["__overall__"]["min"]),
        fmtf(w_summary["__overall__"]["p25"]),
        fmtf(w_summary["__overall__"]["median"]),
        fmtf(w_summary["__overall__"]["p75"]),
        fmtf(w_summary["__overall__"]["max"]),
    ))

    print("\nZ pixel spacing (mm) after resampling — by organ (Z = {:.2f} × H spacing):".format(Z_PER_H_FACTOR))
    print(hdr)
    print("-" * len(hdr))
    for org in [ORG_BY_SUBFOLDER[s] for s in CT_SUBFOLDERS]:
        st = z_summary.get(org, {"n":0,"min":None,"p25":None,"median":None,"p75":None,"max":None})
        print(f"{org:<12}  {fmt(st['n']):>6}  {fmtf(st['min']):>8}  {fmtf(st['p25']):>8}  {fmtf(st['median']):>8}  {fmtf(st['p75']):>8}  {fmtf(st['max']):>8}")
    print("OVERALL (Z): n={}, min={}, p25={}, median={}, p75={}, max={}".format(
        fmt(z_summary["__overall__"]["n"]),
        fmtf(z_summary["__overall__"]["min"]),
        fmtf(z_summary["__overall__"]["p25"]),
        fmtf(z_summary["__overall__"]["median"]),
        fmtf(z_summary["__overall__"]["p75"]),
        fmtf(z_summary["__overall__"]["max"]),
    ))

if __name__ == "__main__":
    main()
