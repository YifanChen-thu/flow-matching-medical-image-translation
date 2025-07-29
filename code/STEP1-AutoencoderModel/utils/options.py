import argparse
import os
import torch
from .utils_yaml import load_yaml



# Step 1: Create the full parser using YAML values as defaults
parser = argparse.ArgumentParser()

# ---------------- Dataset ----------------
parser.add_argument('--config', default="", type=str, help="Directory to store cached data")
parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for DataLoader')
parser.add_argument('--num_samples', default=2, type=int, help='Number of samples per CT scan')
parser.add_argument('--input_modality', nargs='+', required=True, help='List of modalities (e.g., t1c t1n t2w t2f)')
   

# ---------------- Paths ----------------
parser.add_argument('--data_dir', default="", type=str, help="Directory containing the dataset")
parser.add_argument('--cache_dir', default="", type=str, help="Directory to store cached data")
parser.add_argument('--output_dir', default="", type=str, help="Directory to store outputs")

# ---------------- Model Checkpoints ----------------
parser.add_argument('--aekl_ckpt', default=None, type=str)
parser.add_argument('--diff_ckpt', default=None, type=str)
parser.add_argument('--cnet_ckpt', default=None, type=str)
parser.add_argument('--disc_ckpt', default=None, type=str)

# ---------------- GPU ----------------
parser.add_argument('--gpu', type=str, default='0', help='GPU index to use')
parser.add_argument('--dist', action='store_true', help='Use distributed training')
parser.add_argument('--DEBUG', action='store_true', help='Enable debug mode')

# ---------------- Training ----------------
parser.add_argument('--max_batch_size', default=1, type=int)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=2.5e-5, type=float)
parser.add_argument('--resume', default="", type=str, help="Path to resume training from a checkpoint")

# ---------------- Diffusion ----------------
parser.add_argument('--num_train_timesteps', default=1000, type=int)
parser.add_argument("--use_vig", action="store_true")
parser.add_argument("--use_seg", action="store_true")
parser.add_argument("--use_t", action="store_true")
parser.add_argument("--use_contrastive", action="store_true")
parser.add_argument("--num_classes", default=0, type=int)

# ---------------- Mamba / ZigMa ----------------
parser.add_argument('--embed_dim', default=768, type=int, help='Embedding dimension for ZigMa model')
parser.add_argument('--depth', default=24, type=int, help='Depth of ZigMa model')
parser.add_argument('--patch_size', default=2, type=int, help='Patch size for ZigMa model')
parser.add_argument('--scan_type', default='zigzagN8', type=str, help='Scan type: zigzagN8, hilbertN8, etc.')
parser.add_argument('--use_pe', default=2, type=int, help='Positional embedding type: 0=none, 1=fixed, 2=learnable')

parser.add_argument('--d_context', default=1, type=int, help='ZigMa context length')

# ---------------- Task and Channels ----------------
parser.add_argument("--in_channels", default=3, type=int, help="Model input channel count")
parser.add_argument("--cond_channels", default=64, type=int, help="Conditioning channel count")
parser.add_argument("--seg_channels", default=64, type=int, help="Segmentation channel count")
parser.add_argument("--contrastive_channel", default=0, type=int, help="Contrastive feature channel count")

# ---------------- Misc ----------------
parser.add_argument("--comment", default="", type=str, help="Optional comment or tag for this run")
parser.add_argument("--project", action="store_true", help="Enable project mode")

# Parse CLI args
args = parser.parse_args()


