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
from src.brats_dataloader import get_brats_dataloader
from accelerate import DistributedDataParallelKwargs

from src import networks
from src import init_autoencoder

# from monai.generative.losses import PerceptualLoss
from src.flow import compute_ut, compute_xt

# from step1.model2D import utils_usage as utils

from accelerate.utils import DistributedDataParallelKwargs

# Set the desired behavior
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

# Initialize accelerator with custom DDP config
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

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
        x0, x1,
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


    # x0, x1,
    z = x0
    t_steps = torch.linspace(0.0, 1.0, num_inference_steps, device=device)


    progress_bar = tqdm(range(len(t_steps)), desc="Sampling", disable=not verbose)

    dt = t_steps[1] - t_steps[0] 


    for i, t in enumerate(progress_bar):
        with torch.no_grad(), accelerator.autocast():
            timestep = torch.tensor([t_steps[i]]).to(device)
            
            v = diffusion(x=z.float(), timesteps=timestep,)  # [B, C, ...] = v(x_t, t)

            z = z + v * dt

    # decode the latent
    z = z / scale_factor
    z = utils.to_vae_latent_trick(z.cpu(), unpadded_z_shape=(z.shape[0], 4, *[s // 4 for s in spatial_size]))
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

def save_grid_image_by_plane(image_np, recon_broken, recon_np, save_root, b, epoch, modality_names=None):
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


def images_to_tensorboard(
        batch,
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        scale_factor,
        modality_names = ["T1c", "T1n", "T2w", "T2f"] 
):
    """
    Visualize the generation on tensorboard
    """


    x0 = batch[latent_key].to(DEVICE).clone() * scale_factor
    x1 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor

    # print("inputs_latents = ", inputs_latents.shape, "context = ", context.shape)

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder

    with torch.no_grad(), accelerator.autocast():
        image = sample_using_diffusion(
            autoencoder=ae,
            diffusion=diffusion,
            x0=x0, #inputs_latents,
            x1=x1,
            num_inference_steps=100,
            device=DEVICE,
            scale_factor=scale_factor
        )

        recon_origin = ae.decode(x0 / scale_factor).cpu().numpy()  # [B, 1, H, W]
        recon_broken = ae.decode(x1 / scale_factor).cpu().numpy()  # [B, 1, H, W]


    image_np     = recon_origin  #.cpu().numpy()  # [B, 1, H, W]
    recon_np     = image.cpu().numpy()  #.max(axis=1, keepdims=True)  # [B, 3, H, W] -> [B, 1, H, W]

    save_root = "./fm_samples"
    os.makedirs(save_root, exist_ok=True)

    # for b in range(image_np.shape[0]):
   
    modality_names = [m.upper() for m in modality_names]


    for b in range(min(image_np.shape[0], 3)):
        # print("image_np[b] =", image_np[b].shape, recon_broken[b].shape, recon_np[b].shape, context_np[b].shape)
        save_grid_image_by_plane(image_np, recon_broken, recon_np, 
                                 save_root, b, epoch,
                                 modality_names=modality_names[:image_np[b].shape[0]])



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
    key_to_load  = ["mask", "density"] #["mask", "density"]
    
    in_channels  = len(image_key)   
    key_to_load.extend(image_key)

    train_loader      = get_brats_dataloader(args.data_dir, args.latent_dir, mode="train", batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)
    valid_loader      = get_brats_dataloader(args.data_dir, args.latent_dir, mode="test",  batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)


    print("Setting up Autoencoder model...")
    autoencoder = init_autoencoder(in_channels=in_channels)
    print("Finish setting up...")
    try:
        weight = torch.load(args.aekl_ckpt)
        weight = remove_module_prefix(weight)

        autoencoder.load_state_dict(weight)

    except FileNotFoundError:
        print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")
        
    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference

    perceptual_loss_fn = CustomPerceptualLoss(in_channels=4)  # in_channels=4


    diffusion = networks.init_latent_diffusion(args, use_image=False).to(DEVICE)


    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        prediction_type = DDPMPredictionType.SAMPLE,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inferer   = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-6)  # AdamW

    # with torch.no_grad(), accelerator.autocast():
    #     z = data_loader.dataset[0:5][latent_key]
    a = train_loader.dataset[0][latent_key]

    with torch.no_grad(), accelerator.autocast():
        z_list = [train_loader.dataset[i][latent_key] for i in range(5)]
        z = torch.stack(z_list, dim=0)  # Stack into a single tensor


    scale_factor = 1 / torch.std(z)  #  1.3949819803237915
    print(f"Scaling factor set to {scale_factor}")

    autoencoder, optimizer, train_loader, diffusion, perceptual_loss_fn = accelerator.prepare(
        autoencoder, optimizer, train_loader, diffusion, perceptual_loss_fn, 
        # ddp_kwargs={"find_unused_parameters": True}
    )

    writer   = SummaryWriter()
    global_counter = {'train': 0}  # , 'valid': 0 }
    loaders  = {'train': train_loader}  # , # 'valid': valid_loader }
    datasets = {'train': train_loader.dataset}  # , 'valid': validset }

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
    gradient_accumulation_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 4  # for example


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

                # with autocast(device_type='cuda',enabled=True):
                with accelerator.autocast():
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    # Use to be  context: context tensor (N, 1, ContextDim).
                    B = batch[latent_key].shape[0]
                    t = torch.rand(B, device=DEVICE).long()
                    x0        = batch[latent_key].to(DEVICE).clone() * scale_factor
                    x1        = batch[broken_latent_key].to(DEVICE).clone() * scale_factor

                    # x0 -> x1
                    xt = compute_xt(x0=x0, x1=x1, t=t)
                    ut = compute_ut(x0=x0, x1=x1, t=t)
                    
                    # vt = self.forward(x=xt, t=t, **cond_kwargs)
         
                    pred = diffusion(x=xt, timesteps=t, context=None)

                    loss = F.mse_loss(pred, ut)  # MSE Loss

          

                if mode == 'train':
                    # Accumulated Loss
                    loss = loss / gradient_accumulation_steps  # normalize loss
                    accelerator.backward(loss)

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(loader)):
                        optimizer.step()
                        optimizer.zero_grad()


                epoch_loss += loss.item()

                progress_bar.set_postfix({
                    "Step": step,
                    "Loss": epoch_loss / (step + 1),
                    # "Percept": perceptual_loss.item(),
                })

                global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

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
        # torch.save(diffusion.state_dict(), savepath)
        

        if accelerator.is_main_process:
            accelerator.save(diffusion.state_dict(), savepath)
            try:
                savepath = os.path.join(args.output_dir, f'dm-unet-ep-{epoch - 1}-{image_key_str}.pth')
                os.remove(savepath)
            except FileNotFoundError:
                print(f"File {savepath} not found, skipping deletion.")

        print("Saving models to: ", savepath)

        gc.collect()
        torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
