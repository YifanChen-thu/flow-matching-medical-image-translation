import torch
import torch.nn.functional as F
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Initialize LPIPS model (use 'alex' or 'vgg')
lpips_model = lpips.LPIPS(net='alex')


def psnr_3d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # PSNR (scikit-image, over 3D volume)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    return psnr_value

def psnr_2d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, H, W)
    """
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # PSNR (scikit-image, over 2D image)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    return psnr_value

def ssim_3d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # SSIM (scikit-image, for 3D: need to loop over slices)
    ssim_total = 0
    count = 0
    for i in range(pred.shape[0]):
        ssim_slice = structural_similarity(
            target[i], pred[i], data_range=1.0
        )
        ssim_total += ssim_slice
        count += 1
    ssim_value = ssim_total / count

    return ssim_value


def ssim_3d_multichannel(pred, target, in_channels=4):
    """
    Compute mean SSIM over all batches, channels, and depth slices.

    Args:
        pred (np.ndarray): shape (B, C, D, H, W)
        target (np.ndarray): same shape
        in_channels (int): number of input channels

    Returns:
        float: mean SSIM over all slices, channels, and batches
    """
    assert isinstance(pred, np.ndarray), "Expected pred to be a NumPy array"
    assert isinstance(target, np.ndarray), "Expected target to be a NumPy array"
    assert pred.shape == target.shape, "Input arrays must have the same shape"
    assert pred.ndim == 5, "Expected shape (B, C, D, H, W)"

    B, C, D, H, W = pred.shape

    ssim_total = 0.0
    count = 0

    for b in range(B):
        for c in range(in_channels):
            for d in range(D):
                pred_slice = np.clip(pred[b, c, d], 0, 1).astype(np.float32)
                target_slice = np.clip(target[b, c, d], 0, 1).astype(np.float32)

                ssim_val = structural_similarity(
                    target_slice, pred_slice, data_range=1.0
                )
                ssim_total += ssim_val
                count += 1

    return ssim_total / count if count > 0 else 0.0


def compute_3dmetrics(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # PSNR (scikit-image, over 3D volume)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, for 3D: need to loop over slices)
    ssim_total = 0
    count = 0
    for i in range(pred.shape[0]):
        ssim_slice = structural_similarity(
            target[i], pred[i], data_range=1.0
        )
        ssim_total += ssim_slice
        count += 1
    ssim_value = ssim_total / count

    # LPIPS (expects 2D RGB images, so we can take central slices or mean across slices)
    # Take middle slice along z-axis and replicate to 3 channels
    mid_slice_pred   = torch.tensor(pred[pred.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mid_slice_target = torch.tensor(target[target.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    lpips_value = lpips_model(mid_slice_pred, mid_slice_target).item()

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value
    }


def compute_dice(pred_binary, target_binary, eps=1e-6):
    intersection = (pred_binary & target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


import numpy as np
def compute_3dmetrics(pred, target, input):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    # if not isinstance(input, torch.Tensor):
    #     input = torch.tensor(input)

    input = input.squeeze()
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # PSNR (scikit-image, over 3D volume)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, for 3D: need to loop over slices)
    ssim_total = 0
    count = 0
    for i in range(pred.shape[0]):
        ssim_slice = structural_similarity(
            target[i], pred[i], data_range=1.0
        )
        ssim_total += ssim_slice
        count += 1
    ssim_value = ssim_total / count

    # LPIPS (expects 2D RGB images, so we can take central slices or mean across slices)
    # Take middle slice along z-axis and replicate to 3 channels
    mid_slice_pred   = torch.tensor(pred[pred.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mid_slice_target = torch.tensor(target[target.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    lpips_value = lpips_model(mid_slice_pred, mid_slice_target).item()

    change_threshold = 0.02
    gt_diff   = np.abs(target - input)
    pred_diff = np.abs(pred   - input)


    gt_change_map   = (gt_diff > change_threshold)
    pred_change_map = (pred_diff > change_threshold)

    map_mae = np.abs(gt_diff - pred_diff).sum() / gt_change_map.sum()

    dice_change = compute_dice(pred_change_map, gt_change_map)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value,
        'CHANGE_MAE': map_mae,
        'CHANGE_DICE': dice_change
    }



def compute_2dmetrics(pred, target, lpips_model):
    """
    pred, target: torch.Tensor of shape (1, 1, H, W)
    lpips_model: a preloaded LPIPS model (expects input in shape [N, 3, H, W])
    """
    pred = pred.squeeze()  # shape (H, W)
    target = target.squeeze()  # shape (H, W)
    pred = pred.clip(0, 1)

    # PSNR (scikit-image, over 2D image)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, over 2D image)
    ssim_value = structural_similarity(target, pred, data_range=1.0)

    # LPIPS (expects 2D RGB images, so we replicate to 3 channels)
    pred_rgb = pred.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)    # shape (1, 3, H, W)
    target_rgb = target.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # shape (1, 3, H, W)
    lpips_value = lpips_model(pred_rgb, target_rgb).item()

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value
    }