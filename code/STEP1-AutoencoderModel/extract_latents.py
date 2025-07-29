import os, gc, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import warnings
import numpy as np
import torch
from tqdm import tqdm
from monai.utils import set_determinism

import matplotlib.pyplot as plt

from utils import args, import_from_dotted_path, utils_metric
from src import utils_usage


from src import init_autoencoder, KLDivergenceLoss, init_patch_discriminator, GradientAccumulation
from utils.utils_image import save_image, pad_to_shape

from src.brats_dataloader import get_brats_dataloader
from accelerate import Accelerator
import shutil

accelerator = Accelerator()
DEVICE = accelerator.device


from collections import OrderedDict


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict

def extract_middle_slices_per_channel(volume):
    # volume: (C, H, W, D)
    C, H, W, D = volume.shape
    axial_slices = []
    coronal_slices = []
    sagittal_slices = []

    for c in range(C):
        vol_c = volume[c]
        axial    = vol_c[:, :, D // 2]
        coronal  = vol_c[:, W // 2, :]
        sagittal = vol_c[H // 2, :, :]
        axial_slices.append(axial)
        coronal_slices.append(coronal)
        sagittal_slices.append(sagittal)

    return axial_slices, coronal_slices, sagittal_slices


if __name__ == '__main__':

    image_key = args.input_modality

    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)



    in_channels = len(image_key)  # 4 channels: t1c, t1n, t2w, t2f
    dimension = 3
    spatial_size = (96, 96, 96)  # (128, 128, 128)
    # spatial_size = (64, 64, 64)  # (128, 128, 128)
    key_to_load = ["mask"]  # ["mask", "density"]
    # image_key = ["t1c", "t1n", "t2w", "t2f"]
    key_to_load.extend(image_key)

    data_loader = get_brats_dataloader(args.data_dir, mode="test_all", batch_size=args.batch_size, deterministic=True,
                                        spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)


    print("Setting up Autoencoder model...")
    autoencoder = init_autoencoder(in_channels)
    print("Finish setting up...")
    weight = torch.load(args.aekl_ckpt)
    weight = remove_module_prefix(weight)
    # for k in weight:
    #     weight[k] = weight[k].float()

    autoencoder.load_state_dict(weight)

    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference
 
    print(autoencoder.decoder)

    autoencoder, data_loader = accelerator.prepare(autoencoder, data_loader)
    
    if isinstance(autoencoder, torch.nn.parallel.DistributedDataParallel):
        autoencoder = autoencoder.module



    # shutil.rmtree(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, image_key_str)

    with torch.no_grad(), accelerator.autocast():
    # with torch.no_grad():
        os.makedirs(args.output_dir, exist_ok=True)  # 确保输出目录存在
        ids = 0

        print("Test Source...")
        for batch in tqdm(data_loader, total=len(data_loader)):
            masks        = batch["mask"].to(DEVICE)  # 获取掩码
            masks        = ( masks <=0 ).float()  # 将掩码转换为 float 类型
            images       = torch.cat([batch[key].float() for key in image_key], dim=1)

            patient_id   = batch["patient_id"]  # 获取源文件路径

            images = images.to(DEVICE)
            z_mu, z_sigma = autoencoder.encode(images)
            mri_latent = z_mu.cpu().numpy() 

            # 保存潜在表示, in Batch
            for i, latent in enumerate(mri_latent):
                destpath = os.path.join(args.output_dir, patient_id[i] + '.npz')
                np.savez_compressed(destpath, data=latent)
                # print("destpath = ", destpath)


            z_mu, z_sigma = autoencoder.encode(images.clone() * masks)
            mri_latent = z_mu.cpu().numpy() 
            for i, latent in enumerate(mri_latent):
                destpath = os.path.join(args.output_dir, patient_id[i] + '-broken.npz')
                np.savez_compressed(destpath, data=latent)


            del mri_latent
            gc.collect()
            torch.cuda.empty_cache()

            if ids == 1:

                reconstruction, z_mu, z_sigma = autoencoder(images.float())
                # posterior = autoencoder.encode(images).latent_dist  # [0]
                # z = posterior.mean
                # reconstruction = autoencoder.decode(z).sample  # [0]

                os.makedirs("./reconstruction", exist_ok=True)

                reconstruction = reconstruction.cpu().numpy()

                modalities = ["T1c", "T1n", "T2w", "T2f"]
                planes = ["Axial", "Coronal", "Sagittal"]

                # 保存重建图像
                for i, recon in enumerate(reconstruction[:5]):
                    recon_destpath = os.path.join("./reconstruction", f"recon_{patient_id[i]}.jpg")

                    recon = np.squeeze(recon)
                    img   = images[i].cpu().numpy().squeeze()

                    recon = np.clip(recon, 0, 1)


                    if len(recon.shape) == 3:
                        recon = np.expand_dims(recon, axis=0)
                        img   = np.expand_dims(img, axis=0)

                    recon_ax, recon_cor, recon_sag = extract_middle_slices_per_channel(recon)
                    img_ax, img_cor, img_sag       = extract_middle_slices_per_channel(img)

                    fig, axs = plt.subplots(in_channels, 6, figsize=(18, 12))
                    axs = np.atleast_2d(axs)


                    for m in range(in_channels):  # modalities
                        # Axial
                        axs[m, 0].imshow(recon_ax[m], cmap='gray')
                        axs[m, 0].set_title(f'{modalities[m]} - Recon')
                        axs[m, 1].imshow(img_ax[m], cmap='gray')
                        axs[m, 1].set_title(f'{modalities[m]} - Input')

                        # Coronal
                        axs[m, 2].imshow(recon_cor[m], cmap='gray')
                        axs[m, 2].set_title(f'{modalities[m]} - Recon')
                        axs[m, 3].imshow(img_cor[m], cmap='gray')
                        axs[m, 3].set_title(f'{modalities[m]} - Input')

                        # Sagittal
                        axs[m, 4].imshow(recon_sag[m], cmap='gray')
                        axs[m, 4].set_title(f'{modalities[m]} - Recon')
                        axs[m, 5].imshow(img_sag[m], cmap='gray')
                        axs[m, 5].set_title(f'{modalities[m]} - Input')

                        for j in range(6):
                            axs[m, j].axis('off')

                    plt.tight_layout()
                    save_path = os.path.join("./reconstruction", f"{patient_id[i]}.jpg")
                    plt.savefig(save_path)
                    plt.close()

            ids += 1


    print("Latent representations saved successfully.")

