import os, gc
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.networks.schedulers import DDPMScheduler
from monai.inferers import DiffusionInferer
# from src.mask_diffusion_inferer import MaskDiffusionInferer
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

# from src.diffusion import (
#     sample_using_diffusion
# )
from accelerate import Accelerator
from PIL import Image

from src import utils_usage as utils  
from utils import args, import_from_dotted_path, utils_metric

from collections import OrderedDict
from copy import deepcopy

from src import diffusion
from src.infer_dataloader import get_infer_dataloader
from accelerate import DistributedDataParallelKwargs

from src import networks
from src import init_autoencoder

# from monai.generative.losses import PerceptualLoss



# from step1.model2D import utils_usage as utils

accelerator = Accelerator()
DEVICE = accelerator.device

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



os.makedirs(args.cache_dir,  exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

from torch import nn
from monai.networks.schedulers import DDIMScheduler
import imageio.v2 as imageio
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers.ddim import DDIMPredictionType



spatial_size = (96, 96, 96)


def prepare_latent(latents, mask, ratio=4):
    size = [s // ratio for s in spatial_size]

    # BG mask
    shrink_mask = F.interpolate(
        (mask <= 0).float(),
        size=size,
        mode='trilinear',
        align_corners=False
    )

    return latents * shrink_mask, shrink_mask
      

@torch.no_grad()
def sample_using_diffusion(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        latents: torch.Tensor,
        context: torch.Tensor,
        fore_mask: torch.Tensor,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              prediction_type = DDIMPredictionType.SAMPLE,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    context = context.to(device).to(device)

    z = torch.randn((context.shape[0], 4, *[s // 4 for s in spatial_size])).to(device)  # [1, 4, 32, 32]


    # fore_mask = (context > 0).float()  # Foreground mask, 1 for tumor, 0 for background
    z = z * fore_mask + (1 - fore_mask) * latents # Apply the mask to the latent space


    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad(), accelerator.autocast():
            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z.float(),
                timesteps=timestep,
                context=context.float(),
            )

            z, _ = scheduler.step(noise_pred, t, z)
            z = z * fore_mask + (1 - fore_mask) * latents  # Apply the mask to the latent space

    # decode the latent
    z = z / scale_factor

    z = utils.to_vae_latent_trick(z.cpu(), unpadded_z_shape=(context.shape[0], 4, *[s // 4 for s in spatial_size]))

    
    x = autoencoder.decode(z.to(device)).cpu()  #.sample.cpu().squeeze(1)

    return x



mask_key       = "mask"
latent_key     = "latent"
file_key       = "filename"
broken_latent_key = "broken-latent"


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def save_image(path, x):
    """
    Save a single image to the specified path.
    """

    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    # Save the image
    imageio.imwrite(path, x)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def save_image(path, image_array):
    """Convert np array to uint8 image and save"""
    image_array = np.clip(image_array, 0, 1)  # normalize for safety
    image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)

def get_middle_slices(volume):  # [C, D, H, W]
    d, h, w = volume.shape[1:]
    axial = volume[:, d // 2, :, :]     # shape: [C, H, W]
    coronal = volume[:, :, h // 2, :]   # shape: [C, D, W]
    sagittal = volume[:, :, :, w // 2]  # shape: [C, D, H]
    return [axial, coronal, sagittal]

def save_grid_image_by_plane(image_np, recon_broken, recon_np, context_np, save_root, b, epoch, modality_names=None):
    image_channels = image_np[b].shape[0]
    row_labels = modality_names[:image_channels] if modality_names else [f"Mod{i}" for i in range(image_channels)]
    row_labels += ["Mask"]
    col_labels = ["Input", "Broken", "Recon"]
    total_rows = image_channels + 1
    total_cols = 3  # input, broken, recon
    plane_names = ["Axial", "Coronal", "Sagittal"]

    image_views  = get_middle_slices(image_np[b])
    broken_views = get_middle_slices(recon_broken[b])
    recon_views  = get_middle_slices(recon_np[b])
    mask_views   = get_middle_slices(context_np[b])

    for p, plane in enumerate(plane_names):
        fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

        for r in range(image_channels):
            # Input, Broken, Recon for modality r
            axes[r, 0].imshow(image_views[p][r], cmap="gray")
            axes[r, 1].imshow(broken_views[p][r], cmap="gray")
            axes[r, 2].imshow(recon_views[p][r], cmap="gray")
            for c in range(total_cols):
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(row_labels[r], fontsize=12)

        # Last row: only show mask
        axes[image_channels, 0].imshow(mask_views[p][0], cmap="gray")
        axes[image_channels, 1].axis("off")
        axes[image_channels, 2].axis("off")
        axes[image_channels, 0].axis("off")
        axes[image_channels, 0].set_ylabel("Mask", fontsize=12)

        # Column headers
        for c in range(total_cols):
            axes[0, c].set_title(col_labels[c], fontsize=12)

        fig.suptitle(f"Sample {b} - {plane} View", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_root, f"{epoch}_sample{b}_{plane}.jpg")
        plt.savefig(save_path, dpi=150)
        plt.close()




def save_full_grid_image_by_plane(original_full_np, inpainted_np, full_mask_np, save_root, b, epoch):
    """
    Save side-by-side comparison of original vs inpainted full volume.
    
    Args:
        original_full_np: [B, 1, D, H, W] numpy array
        inpainted_np:     [B, 1, D, H, W] numpy array
        save_root: output dir
        b: batch index
        epoch: epoch index
    """
    original = original_full_np[b]  # [1, D, H, W]
    inpainted = inpainted_np[b]     # [1, D, H, W]

    original_views = get_middle_slices(original)
    inpainted_views = get_middle_slices(inpainted)
    full_mask_views = get_middle_slices(full_mask_np[b])  # [1, D, H, W]

    plane_names = ["Axial", "Coronal", "Sagittal"]
    col_labels = ["Original", "Inpainted", "Full Mask"]

    for p, plane in enumerate(plane_names):
        fig, axes = plt.subplots(1, 3, figsize=(6, 3))

        axes[0].imshow(original_views[p][0], cmap="gray")
        axes[1].imshow(inpainted_views[p][0], cmap="gray")
        axes[2].imshow(full_mask_views[p][0], cmap="gray")

        for c in range(3):
            axes[c].set_title(col_labels[c], fontsize=12)
            axes[c].axis("off")

        fig.suptitle(f"Full Volume Sample {b} - {plane} View", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_root, f"{epoch}_sample{b}_FULL_{plane}.jpg")
        plt.savefig(save_path, dpi=150)
        plt.close()



from monai.utils import ensure_tuple
import random


def get_random_edge_crop_coords(mask, crop_size=(96, 96, 96)):
    """
    Generate a 96^3 crop inside the non-zero region of `mask`,
    biased to be near the edges of the full_mask region.
    Returns (start_coords), (end_coords)
    """
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        raise ValueError("No non-zero voxels found in full_mask")

    zmin, ymin, xmin = nonzero.min(dim=0).values
    zmax, ymax, xmax = nonzero.max(dim=0).values + 1  # make end exclusive

    dz, dy, dx = crop_size

    # Calculate valid range where crop can start and still fit
    max_z0 = max(zmax - dz, zmin)
    max_y0 = max(ymax - dy, ymin)
    max_x0 = max(xmax - dx, xmin)

    # Bias toward edges by randomly picking from near min or max
    z0 = random.choice([random.randint(zmin, zmin + 10), random.randint(max_z0 - 10, max_z0)])
    y0 = random.choice([random.randint(ymin, ymin + 10), random.randint(max_y0 - 10, max_y0)])
    x0 = random.choice([random.randint(xmin, xmin + 10), random.randint(max_x0 - 10, max_x0)])

    # Clamp to ensure we stay inside bounds
    z0 = max(zmin, min(z0, max_z0))
    y0 = max(ymin, min(y0, max_y0))
    x0 = max(xmin, min(x0, max_x0))

    z1 = z0 + dz
    y1 = y0 + dy
    x1 = x0 + dx

    return (int(z0), int(y0), int(x0)), (int(z1), int(y1), int(x1))





def images_to_tensorboard(
        batch,
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        scale_factor,
        modality_names=["T1c", "T1n", "T2w", "T2f"]
):
    """
    Visualize the generation on tensorboard. Replaces a slice in full_t1n with inpainted output.
    """
    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder

    # --- Load inputs ---
                            # [B, 1, H, W]
    input_image = torch.cat([batch[key] for key in image_key], dim=1).to(DEVICE)  # [B, C, H, W]
    t1n_full = batch["full_t1n"].to(DEVICE)                    # [B, 1, D, H, W] or [B, D, H, W]


    full_mask = batch["full_mask"].to(DEVICE)  # [B, 1, D, H, W] or [B, D, H, W]
    full_mask = (full_mask>0.1).float()

    print(f"t1n_full shape: {t1n_full.shape}, full_mask shape: {full_mask.shape}", "unique v alue of full_mask:", torch.unique(full_mask))



    context = batch["mask"].to(DEVICE) 

    start, end = get_random_edge_crop_coords(full_mask[0,0], crop_size=(96, 96, 96))
    z0, y0, x0 = start
    z1, y1, x1 = end

    input_image = t1n_full[..., z0:z1, y0:y1, x0:x1]

    full_shape = t1n_full.shape  # [B, 1, D, H, W]

    with torch.no_grad(), accelerator.autocast():
        latents, _ = ae.encode(input_image)   # [B, latent_dim, H, W]
        latents = latents * scale_factor

    # --- Prepare for diffusion ---
    inputs_latents, bg_mask = prepare_latent(latents, context, ratio=4)
    masked_latents = latents * bg_mask

    with torch.no_grad(), accelerator.autocast():
        image = sample_using_diffusion(
            autoencoder=ae,
            diffusion=diffusion,
            latents=latents,  # could also try masked_latents
            context=context,
            fore_mask=1 - bg_mask,
            num_inference_steps=100,
            device=DEVICE,
            scale_factor=scale_factor
        )

        recon_origin = ae.decode(latents / scale_factor).cpu().numpy()
        recon_broken = ae.decode(masked_latents / scale_factor).cpu().numpy()
        recon_np = image.cpu().numpy()

    context_np = context.cpu().numpy()
    image_np = recon_origin

    # --- Inpaint the image slice back into full_t1n ---
    inpainted_t1n = t1n_full.clone()  # [B, 1, D, H, W]
    for b in range(image.shape[0]):
        if inpainted_t1n.dim() == 5:
            inpainted_t1n[b, 0][..., z0:z1, y0:y1, x0:x1] = image[b, 0]
            full_mask[b, 0][..., z0:z1, y0:y1, x0:x1] = 0  # Update full_mask to reflect inpainted region
        elif inpainted_t1n.dim() == 4:
            inpainted_t1n[b][..., z0:z1, y0:y1, x0:x1] = image[b, 0]
            full_mask[b][..., z0:z1, y0:y1, x0:x1] = 0  # Update full_mask to reflect inpainted region
        else:
            raise ValueError(f"Unexpected t1n_full shape: {inpainted_t1n.shape}")

    # --- Save visuals ---
    save_root = "./diffusion_samples"
    os.makedirs(save_root, exist_ok=True)
    modality_names = [m.upper() for m in modality_names]

    for b in range(min(image_np.shape[0], 3)):
        save_grid_image_by_plane(
            image_np, recon_broken, recon_np, context_np,
            save_root, b, epoch,
            modality_names=modality_names[:image_np.shape[1]]  # channel count
        )


        save_full_grid_image_by_plane(
            t1n_full.cpu().numpy(), inpainted_t1n.cpu().numpy(), full_mask.cpu().numpy(),
            save_root, b, epoch
            # modality_names= # channel count
        )

    return inpainted_t1n  #



class CustomPerceptualLoss(nn.Module):
    def __init__(self, in_channels=3,):
        super().__init__()

        from monai.networks.nets import resnet10, resnet18, resnet34
        import torch.nn as nn

        # Load pretrained DenseNet121 on MedNIST (6-class)
        feature_model = resnet18(spatial_dims=3, n_input_channels=1, 
                                    pretrained=True, feed_forward=False,       
                                    shortcut_type='A',
                                    bias_downsample=True)

        # Adjust the model in_channels if necessary
        if in_channels != 1:
            # Step 2: Modify the first conv layer to accept more channels
            old_conv = feature_model.conv1  #stem[0]
            new_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            # Step 3: Copy pretrained weights
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight)
                elif in_channels < 4:
                    # Repeat weights from the 1-channel model
                    new_conv.weight.copy_(old_conv.weight.repeat(1, in_channels, 1, 1, 1) / in_channels)
                else:
                    # Use mean across input dim if in_channels is large
                    new_conv.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1)
                    )

            # Replace the conv layer
            feature_model.conv1 = new_conv



        feature_model.eval()
        for p in feature_model.parameters():
            p.requires_grad = False


        # print("feature_model = ", feature_model)

        self.feature_extractor = nn.Sequential(
            feature_model.conv1,
            feature_model.bn1,
            feature_model.act,
            feature_model.maxpool,
            feature_model.layer1,
            feature_model.layer2
        )

        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        # Get features from both inputs
        input_feats  = self.feature_extractor(input).detach()
        target_feats = self.feature_extractor(target)
        return self.criterion(input_feats, target_feats)



if __name__ == '__main__':
    image_key = args.input_modality

    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)


    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Define Dataloader ----------------
    num_train_timesteps = 1000
    
    dimension = 3
    spatial_size = (96, 96, 96)  #(128, 128, 128)
    # spatial_size = (64, 64, 64)  # (128, 128, 128)
    key_to_load  = ["mask", "density", "full_mask"] #["mask", "density"]
    
    in_channels  = len(image_key)   
    key_to_load.extend(image_key)

    train_loader      = get_infer_dataloader(args.img_dir, args.data_dir, args.latent_dir, mode="train", batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)
    valid_loader      = get_infer_dataloader(args.img_dir, args.data_dir, args.latent_dir, mode="test",  batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)


    print("Setting up Autoencoder model...")
    autoencoder = init_autoencoder(in_channels=in_channels)
    print("Finish setting up...")
    try:
        weight = torch.load(args.aekl_ckpt, map_location=DEVICE)
        weight = remove_module_prefix(weight)

        autoencoder.load_state_dict(weight)
    except FileNotFoundError:
        print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")
        
    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference

    diffusion = networks.init_latent_diffusion(args).to(DEVICE)
    # load from diff_ckpt
    if args.diff_ckpt is not None:
        try:
            weight = torch.load(args.diff_ckpt, map_location=DEVICE)
            weight = remove_module_prefix(weight)
            diffusion.load_state_dict(weight)

        except FileNotFoundError:
            print(f"File {args.diff_ckpt} not found, using random initialization for diffusion model.")


    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        prediction_type = DDPMPredictionType.SAMPLE,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inferer   = DiffusionInferer(scheduler=scheduler)


    scale_factor = 1.89 # 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    autoencoder, train_loader, diffusion = accelerator.prepare( autoencoder, train_loader, diffusion )

    writer   = SummaryWriter()
    global_counter = {'train': 0}  # , 'valid': 0 }
    loaders  = {'train': train_loader}  # , # 'valid': valid_loader }
    datasets = {'train': train_loader.dataset}  # , 'valid': validset }

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder

    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"{mode.upper()} Epoch {epoch}")

            for step, batch in progress_bar:
                if args.DEBUG and step >= 10:
                    print(f"[DEBUG] Step {step}: {batch[latent_key].shape}")
                    break

                # visualize results
                images_to_tensorboard(
                    batch=batch,
                    writer=writer,
                    epoch=epoch,
                    mode=mode,
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    scale_factor=scale_factor,
                    modality_names=image_key
                )




        # save the model                
        savepath = os.path.join(args.output_dir, f'dm-unet-ep-{epoch}-{image_key_str}.pth')
        torch.save(diffusion.state_dict(), savepath)
        try:
            savepath = os.path.join(args.output_dir, f'dm-unet-ep-{epoch - 1}-{image_key_str}.pth')
            os.remove(savepath)
        except FileNotFoundError:
            print(f"File {savepath} not found, skipping deletion.")

        print("Saving models to: ", savepath)

        gc.collect()
        torch.cuda.empty_cache()
