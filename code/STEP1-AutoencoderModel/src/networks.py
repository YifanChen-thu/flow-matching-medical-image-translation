# ----- This is the Unet3D model from MAISI -----

import torch.nn as nn
from monai.networks.nets import (
    AutoencoderKL,
)

from .module.scripts.utils import define_instance
import os
from typing import Optional
import torch
from monai.apps import download_url

import os
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator,
    DiffusionModelUNet, 
    ControlNet
)
from collections import OrderedDict


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict



def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize

    Returns:
        nn.Module: the initialized neural network
    """



    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        # device = next(network.parameters()).device # Use the same device as the model
        checkpoint = torch.load(checkpoints_path, map_location='cpu', weights_only=True)

        new_state_dict = {}
        if "autoencoder" in checkpoints_path:
            for key in checkpoint:
                new_key = key
                if (
                        "decoder.blocks.3.conv.conv" in key or
                        "decoder.blocks.6.conv.conv" in key or
                        "decoder.blocks.9.conv.conv" in key
                ):
                    new_key = key.replace("conv.conv", "postconv.conv")
                new_state_dict[new_key] = checkpoint[key]

        elif "controlnet" in checkpoints_path or "cnet" in checkpoints_path:
            for key, val in checkpoint.items():
                if "module." in key:
                    key = key.replace("module.", "")

                if "to_out.0" in key:
                    new_key = key.replace("to_out.0", "out_proj")
                    new_state_dict[new_key] = val
                else:
                    new_state_dict[key] = val

        elif "diffusion" in checkpoints_path:
            for k, v in checkpoint.items():
                new_k = k

                # Common renames
                new_k = new_k.replace("to_out.0", "out_proj")
                new_k = new_k.replace("proj_out.0", "proj_out.conv")
                new_k = new_k.replace("proj_in.0", "proj_in.conv")
                new_k = new_k.replace("conv_in.0", "conv_in.conv")
                new_k = new_k.replace("conv_out.0", "out.2.conv")  # if out is ConvBlock
                new_k = new_k.replace("time_embedding.linear_1", "time_embed.0")
                new_k = new_k.replace("time_embedding.linear_2", "time_embed.2")

                new_state_dict[new_k] = v


        print("Loaded pretrained weights from", checkpoints_path)
        network.load_state_dict(new_state_dict)


    return network



def patch_conv_layer(layer, new_in_channels=None, new_out_channels=None):
    """Utility to patch Conv3d layer input or output channels while reusing pretrained weights."""
    assert isinstance(layer, nn.Conv3d), "Only supports nn.Conv3d layers"

    in_channels = new_in_channels if new_in_channels else layer.in_channels
    out_channels = new_out_channels if new_out_channels else layer.out_channels

    new_layer = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias is not None
    )

    with torch.no_grad():
        # Handle input channels (first layer)
        if new_in_channels and new_in_channels > layer.in_channels:
            for i in range(new_in_channels):
                new_layer.weight[:, i] = layer.weight[:, 0]
        elif new_in_channels:
            new_layer.weight[:, :new_in_channels] = layer.weight[:, :new_in_channels]

        # Handle output channels (last layer)
        if new_out_channels and new_out_channels > layer.out_channels:
            for i in range(new_out_channels):
                new_layer.weight[i] = layer.weight[0]
        elif new_out_channels:
            new_layer.weight[:new_out_channels] = layer.weight[:new_out_channels]

        if layer.bias is not None:
            if new_out_channels and new_out_channels > layer.out_channels:
                new_layer.bias[:] = layer.bias[0]
            else:
                new_layer.bias[:new_out_channels] = layer.bias[:new_out_channels]

    return new_layer



def init_autoencoder(in_channels=1) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
                   "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
        },
    ]
    root_dir = "./"
    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        download_url(url=file["url"], filepath=file["path"])

    import pickle

    with open("src/module/inference_args.pkl", "rb") as f:
        args = pickle.load(f)

    autoencoder  = define_instance(args, "autoencoder_def")# .to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True, map_location="cpu")
    autoencoder.load_state_dict(checkpoint_autoencoder)

    # print("Step up in_channels = ", in_channels)

    # Patch first Conv3d (input layer)
    first = autoencoder.encoder.blocks[0].conv.conv
    # print("isinstance(first, nn.Conv3d) and first.in_channels != in_channels: = ", isinstance(first, nn.Conv3d), first.in_channels != in_channels)
    if isinstance(first, nn.Conv3d) and first.in_channels != in_channels:
        # print("Replaceing first Conv3d layer in autoencoder with in_channels =", in_channels)
        autoencoder.encoder.blocks[0].conv.conv = patch_conv_layer(first, new_in_channels=in_channels)

    # Patch last Conv3d (output layer)
    last = autoencoder.decoder.blocks[-1].conv.conv
    if isinstance(last, nn.Conv3d) and last.out_channels != in_channels:
        # print("Replaceing last Conv3d layer in autoencoder with out_channels =", in_channels)
        autoencoder.decoder.blocks[-1].conv.conv = patch_conv_layer(last, new_out_channels=in_channels)

    autoencoder = autoencoder.float()  # Ensure float32 precision for stability
    return autoencoder



def init_patch_discriminator(checkpoints_path: Optional[str] = None, spatial_dims=3, in_channels=1, num_layers_d=3) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=spatial_dims,
                                             num_layers_d=num_layers_d,
                                             channels=32,
                                             in_channels=in_channels,
                                             out_channels=1)

    return load_if(checkpoints_path, patch_discriminator)



def init_controlnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    controlnet = ControlNet(spatial_dims=3, 
                            in_channels=3,
                            num_res_blocks=2, 
                            channels=(256, 512, 768),
                            attention_levels=(False, True, True), 
                            norm_num_groups=32, 
                            norm_eps=1e-6, 
                            resblock_updown=True, 
                            num_head_channels=(0, 512, 768), 
                            transformer_num_layers=1, 
                            with_conditioning=True,
                            cross_attention_dim=8, 
                            num_class_embeds=None, 
                            upcast_attention=True, 
                            use_flash_attention=False, 
                            conditioning_embedding_in_channels=4,  
                            conditioning_embedding_num_channels=(256,))
    return load_if(checkpoints_path, controlnet)



if __name__ == "__main__":
    import time
    autoencoder = Autoencoder3D()
    autoencoder.eval()
    img = torch.randn(1, 1, 128, 144, 128)  # (128, 144, 128)


    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        img = img.cuda()

    start = time.time()
    with torch.no_grad():
        latent, z_sigma = autoencoder.encode(img)
        print("latent shape = ", latent.shape)  # torch.Size([1, 3, 16, 18, 16])
        reconstruction = autoencoder.decode(latent)
        print("recon shape = ", reconstruction.shape)


    print("time elapsed = ", time.time() - start)
