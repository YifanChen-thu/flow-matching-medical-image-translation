import torch


def fuse_weight_3_to_1_channel(autoencoder):

    # Input
    # Fuse the weights of the RGB channels into a single grayscale channel
    conv_in = autoencoder.encoder.conv_in

    # The original conv_in layer has 3 input channels (RGB), so get the weights for 3 channels
    weights_rgb = conv_in.weight.data  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

    # To convert to grayscale, average the weights across the 3 input channels (RGB channels)
    weights_gray = weights_rgb.mean(dim=1, keepdim=True)  # Shape: (out_channels, 1, kernel_size, kernel_size)

    # Now assign the averaged weights to a new conv_in layer with 1 input channel
    new_conv_in = torch.nn.Conv2d(
        in_channels=1,  # Now 1 channel (grayscale)
        out_channels=conv_in.out_channels,  # Same output channels as before
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding
    )

    # Copy the averaged weights into the new conv_in layer
    new_conv_in.weight.data = weights_gray

    # Now you can replace the original conv_in with the new one
    autoencoder.encoder.conv_in = new_conv_in

    # Output
    conv_out = autoencoder.decoder.conv_out
    # The original conv_out layer has 3 output channels, so get the weights
    weights_rgb_out = conv_out.weight.data  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

    # To convert to grayscale (1 output channel), average the weights across the 3 output channels
    weights_gray_out = weights_rgb_out.mean(dim=0, keepdim=True)  # Shape: (1, in_channels, kernel_size, kernel_size)

    # Now assign the averaged weights to a new conv_out layer with 1 output channel
    new_conv_out = torch.nn.Conv2d(
        in_channels=conv_out.in_channels,  # Same number of input channels as before
        out_channels=1,  # Change to 1 output channel (grayscale)
        kernel_size=conv_out.kernel_size,
        stride=conv_out.stride,
        padding=conv_out.padding
    )

    # Copy the averaged weights into the new conv_out layer
    new_conv_out.weight.data = weights_gray_out

    autoencoder.decoder.conv_out = new_conv_out


    return autoencoder


def fuse_weight_3_to_4_channel(autoencoder):
    # 第一步: 3通道→1通道 (类似原函数)
    conv_in = autoencoder.encoder.conv_in
    weights_rgb = conv_in.weight.data
    weights_gray = weights_rgb.mean(dim=1, keepdim=True)  # 融合为1通道
    
    # 第二步: 1通道→4通道 (复制权重)
    weights_4ch = weights_gray.repeat(1, 4, 1, 1)  # 复制为4通道
    
    # 创建新的4通道输入层
    new_conv_in = torch.nn.Conv2d(
        in_channels=4,  # 4通道输入
        out_channels=conv_in.out_channels,  # 保持与原来相同的输出通道
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding
    )
    new_conv_in.weight.data = weights_4ch
    
    # 输出层 - 修改为输出1通道，而不是4通道
    conv_out = autoencoder.decoder.conv_out
    weights_rgb_out = conv_out.weight.data
    weights_gray_out = weights_rgb_out.mean(dim=0, keepdim=True)  # 输出为1通道
    
    new_conv_out = torch.nn.Conv2d(
        in_channels=conv_out.in_channels,
        out_channels=1,  # 输出1通道
        kernel_size=conv_out.kernel_size,
        stride=conv_out.stride,
        padding=conv_out.padding
    )
    new_conv_out.weight.data = weights_gray_out
    
    # 替换原有层
    autoencoder.encoder.conv_in = new_conv_in
    autoencoder.decoder.conv_out = new_conv_out
    
    return autoencoder




def load_if(checkpoints_path, network):
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
