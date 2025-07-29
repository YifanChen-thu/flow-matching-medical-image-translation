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

from torch.nn import L1Loss
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.losses import PerceptualLoss, PatchAdversarialLoss
import torch.nn.functional as F


from utils import args, import_from_dotted_path, utils_metric
from src import utils_usage


from src import init_autoencoder, KLDivergenceLoss, init_patch_discriminator, GradientAccumulation
from utils.utils_image import save_image, pad_to_shape

from src.brats_dataloader import get_brats_dataloader
from accelerate import Accelerator

torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore")

set_determinism(0)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_epoch = 1
import torch.nn as nn

accelerator = Accelerator()
DEVICE = accelerator.device


# import os
# os.environ["TMPDIR"] = "~/HDD_8T_2/temp"



ACTIVATION_CLASSES = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.PReLU, nn.RReLU)

def disable_inplace_activations(model: nn.Module):
    for name, module in model.named_modules():
        # print("Search : ", name, module.__class__.__name__, type(module), getattr(module, 'inplace', False))

        if  getattr(module, 'inplace', False):
            print("--> Disabling inplace activation in module: ", name, module.__class__.__name__)
            module.inplace = False

    return model



def print_model_shapes(model, input_size):
    import torch
    x = torch.randn(input_size, dtype=torch.float16).to(DEVICE)
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, (list, tuple)):
            out_shape = [o.shape for o in output]
        else:
            out_shape = output.shape
        print(f"{module.__class__.__name__:<30} | Input: {input[0].shape} -> Output: {out_shape}")

    for name, layer in model.named_modules():
        if layer != model:
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        with accelerator.autocast():
            model(x)

    for h in hooks:
        h.remove()


def to_numpy_image(tensor):
    # Assumes input: [1, H, W] or [H, W]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().squeeze()
        array = tensor.numpy()
    else:
        array = tensor
    return array


def perform_mask_image(images, c_num_keep=-1):
    B, C, H, W, D = images.shape
    mask = torch.zeros((B, C), device=images.device)

    for i in range(B):
        if c_num_keep == -1:
            num_keep = torch.randint(1, C, (1,)).item()
        else:
            num_keep = c_num_keep

        keep_indices = torch.randperm(C)[:num_keep]
        mask[i, keep_indices] = 1

    mask = mask[:, :, None, None, None]  # [B, C, 1, 1, 1]

    mask_images = images * mask 
    return mask_images


def validate_model(model, dataloader, device, save_root=None, max_batches=6,
                   step_name="", image_key="source", image_key_str=None):
    model.eval()
    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)

    avg_psnr, avg_ssim = [], []

    # Print the model architecture, and the input and output channel sizes of each layer
    DEBUG = False
    if DEBUG:
        print("Model architecture:\n")
        print(model)  # Print full model

        print("\nLayer-wise input and output shapes:")
        print_model_shapes(model, input_size=(2, len(image_key), 96, 96, 96))

    with torch.no_grad(), accelerator.autocast():
        for idx, batch in enumerate(dataloader):
            if args.DEBUG and idx >= 5:
                break
            if idx >= max_batches:
                break
            # images = torch.cat(batch[image_key], dim=1).to(DEVICE)
            images = torch.cat([batch[key] for key in image_key], dim=1).to(DEVICE)

            reconstruction, z_mu, z_sigma = model(images)

            # Move to CPU for metrics and visualization
            image_np = images.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()

            # print(f"Batch {idx + 1}/{len(dataloader)}: image shape {image_np.shape}, recon shape {recon_np.shape}")

            # Compute PSNR and SSIM
            psnr_val = utils_metric.psnr_3d(image_np, recon_np)
            # Batch 1/251: image shape (1, 1, 96, 96, 96), recon shape (1, 1, 96, 96, 96)
            ssim_val = utils_metric.ssim_3d_multichannel(image_np, recon_np, in_channels=1) 

            avg_psnr.append(psnr_val)
            avg_ssim.append(ssim_val)

            # Save middle slice comparison image
            middle_slice = image_np.shape[2] // 2
            image_mid = image_np[:, :, middle_slice, :, :]  # B, C, H, W
            recon_mid = recon_np[:, :, middle_slice, :, :]
            if save_root is not None:
                for b in range(image_mid.shape[0]):
                    orig  = image_mid[b]  # [4, H, W]
                    recon = recon_mid[b]  # [4, H, W]

                    # Concatenate orig and recon for each of the 4 slices → [4, 2H, W]
                    combined_slices = [np.concatenate([orig[i], recon[i]], axis=1) for i in range(image_mid.shape[1])]  # each is [H, 2W]

                    # Stack them vertically → [2H, 4W]
                    combined = np.concatenate(combined_slices, axis=0)  # [4H, 2W]

                    save_path = os.path.join(save_root, f"{step_name}_img_{idx}_{b}.jpg")
                    save_image(save_path, combined)

    print(f"{step_name} - AVG_PSNR: {np.mean(avg_psnr):.2f}, AVG_SSIM: {np.mean(avg_ssim):.4f}")



# image_key    = ["t1n"]   # ["t1c", "t1n", "t2w", "t2f"]

# image_key    = ["t1c", "t1n", "t2w", "t2f"]


message = "kl_01"

if __name__ == '__main__':
    image_key = args.input_modality

    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)

    if message != "":
        image_key_str += f"-{message}"


    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Define Dataloader ----------------
    in_channels = len(image_key)  # 4 channels: t1c, t1n, t2w, t2f
    dimension = 3
    spatial_size = (96, 96, 96)  #(128, 128, 128)
    # spatial_size = (64, 64, 64)  # (128, 128, 128)
    key_to_load  = ["mask"] #["mask", "density"]

    
    key_to_load.extend(image_key)

    train_loader     = get_brats_dataloader(args.data_dir, mode="train", batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)
    test_loader      = get_brats_dataloader(args.data_dir, mode="test",  batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)


    # ---------------- Define AutoEncoder Model ----------------
    print("Setting up Autoencoder model...")
    autoencoder = init_autoencoder(in_channels).to(DEVICE)
    print("Finish setting up...")



    discriminator = init_patch_discriminator(args.disc_ckpt, spatial_dims=dimension,
                                             in_channels=in_channels, num_layers_d=3).to(DEVICE)
    discriminator = disable_inplace_activations(discriminator)


    def remove_module_prefix(state_dict):
        from collections import OrderedDict
        """Remove 'module.' prefix from keys (if exists)."""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        return new_state_dict


    if args.resume:
        dis_path = os.path.join(args.output_dir, f'dis-{args.resume}.pth')
        gen_path = os.path.join(args.output_dir, f'ae-{args.resume}.pth')
        if os.path.exists(dis_path):
            discriminator.load_state_dict(remove_module_prefix(torch.load(dis_path)))
        else:
            print(f"Discriminator checkpoint not found: {dis_path}. Starting from scratch.")

        if os.path.exists(gen_path):
            autoencoder.load_state_dict(remove_module_prefix(torch.load(gen_path)))
        else:
            print(f"Autoencoder checkpoint not found: {gen_path}. Starting from scratch.")
        print("Resuming from checkpoint:", gen_path)


    adv_weight = 0.025
    perceptual_weight = 0.001
    kl_weight = 1e-7

    l1_loss_fn  = L1Loss()
    kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    adv_loss_fn = disable_inplace_activations(adv_loss_fn)

    adv_loss_fn.activation = torch.nn.LeakyReLU(negative_slope=0.05, inplace=False)  # <- safe


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=dimension,  # 如果您的网络是3D的，请确认输入维度。如果是2D图像，请用spatial_dims=2
                                      network_type="squeeze",
                                      is_fake_3d=True,
                                      fake_3d_ratio=0.2).to(DEVICE)

    trainable = [p for n, p in autoencoder.named_parameters() if p.requires_grad]
    all_sum = sum(p.numel() for p in autoencoder.parameters())
    print(f"\nAll parameters: {all_sum / 1e6:.2f} M")
    trainable_sum = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_sum / 1e6:.2f} M")

    optimizer_g = torch.optim.Adam(trainable, lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    avgloss = utils_usage.AverageLoss()
    writer  = SummaryWriter()
    total_counter = 0

    save_root = "./image_result/step1_ae_train/"
    os.makedirs(save_root, exist_ok=True)

    # ---------------- Prepare for Training ----------------
    autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, adv_loss_fn = accelerator.prepare(
        autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, adv_loss_fn
    )

    # Test at starter
    validate_model(autoencoder, test_loader, DEVICE,
                   save_root=None, max_batches=6, step_name="Start", image_key=image_key, image_key_str=image_key_str)

    for epoch in range(args.n_epochs):

        if DEVICE == "cuda":
            print("EPOCH: ", epoch)
            print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        autoencoder.train()
        discriminator.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            # if step > 100: break
            if args.DEBUG and step >= 5:
                break

            images = torch.cat([batch[key] for key in image_key], dim=1).to(DEVICE)
            mask   = batch["mask"].to(DEVICE)

            # Mask
            mask_images = perform_mask_image(images, c_num_keep=-1)  # TODO learn with other target or not
            # image = mask_images
            

            # with autocast(enabled=True):
            with accelerator.autocast():
                reconstruction, z_mu, z_sigma = autoencoder(mask_images)  # Masked
                logits_fake = discriminator(reconstruction.contiguous())[-1]

                rec_loss = l1_loss_fn(reconstruction, images)
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)

                # ---------- KL channel divergence loss ----------
                if mask.dim() == 4:
                    mask = mask.squeeze(1)  # convert to (B, H, W) if needed

                # Make sure mask is boolean and same device
                valid_mask = (mask >= 0).to(DEVICE)  


                use_chamnnel_loss = False

                if use_chamnnel_loss and len(image_key) > 1:
                    # Compute per-pixel KL divergence
                    recon_log_prob = F.log_softmax(reconstruction, dim=1)  # log(P)
                    images_prob    = F.softmax(images, dim=1)                 # Q

                    # Compute per-pixel KL divergence (shape: B, C, H, W)
                    kl_div = F.kl_div(recon_log_prob, images_prob, reduction='none', log_target=False)

                    kl_map = kl_div.sum(dim=1)

                    kl_map_masked = kl_map * valid_mask
                    masked_mean   = kl_map_masked.sum() / valid_mask.sum().clamp(min=1)

                    # Final weighted KL loss
                    kl_channel_weight = 0.1  # 0.1
                    channel_kl_loss = masked_mean * kl_channel_weight

                else:
                    channel_kl_loss = torch.tensor(0.0, device=DEVICE)



                # per_loss = perceptual_weight * perc_loss_fn(reconstruction, images)
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False) 

                loss_g = rec_loss + kld_loss + gen_loss + channel_kl_loss  # + per_loss


                progress_bar.set_postfix(loss_g=loss_g.item() if hasattr(loss_g, "item") else loss_g,
                                         rec_loss=rec_loss.item() if hasattr(rec_loss, "item") else rec_loss,
                                         kld_loss=kld_loss.item() if hasattr(kld_loss, "item") else kld_loss,
                                         gen_loss=gen_loss.item() if hasattr(gen_loss, "item") else gen_loss,
                                         channel_kl=channel_kl_loss.item() if hasattr(channel_kl_loss, "item") else channel_kl_loss)


                    # print(f"rec_loss: {rec_loss}, kld_loss: {kld_loss}, gen_loss: {gen_loss}")

            accelerator.backward(loss_g)
            optimizer_g.step()
            optimizer_g.zero_grad()


            # ⚠️ This is a workaround, but should be improved
            with accelerator.autocast():
                fake_images = reconstruction.detach()  # Detach to cut generator graph
                logits_real = discriminator(images.contiguous())[-1]   # .contiguous().detach()
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                discriminator_loss = (d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            optimizer_d.zero_grad()
            accelerator.backward(loss_d)


            with accelerator.autocast():
                fake_images = reconstruction.detach()  # Detach to cut generator graph
                logits_fake = discriminator(fake_images.contiguous())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                discriminator_loss = (d_loss_fake) * 0.5
                loss_d = adv_weight * discriminator_loss

            # optimizer_d.zero_grad()
            accelerator.backward(loss_d)
            optimizer_d.step()
            optimizer_d.zero_grad()



            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            # avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adverarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss', loss_d.item())

            total_counter += 1

        save_root = "./image_result/step1_ae_train/epoch_{}".format(epoch)
        os.makedirs(save_root, exist_ok=True)

        autoencoder.eval()
        validate_model(autoencoder, test_loader, DEVICE,
                       save_root=save_root, max_batches=6,
                       step_name="Epoch_{}".format(epoch), image_key=image_key)


  
        
        # 保存模型
        if (epoch + 1) % save_epoch == 0 and accelerator.is_main_process:
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir,
                                                                f'dis-{epoch + 1}-{image_key_str}.pth'))
            
            torch.save(autoencoder.state_dict(),
                       os.path.join(args.output_dir, f'ae-{epoch + 1}-{image_key_str}.pth'))

            print("Saving models to: ",
                  os.path.join(args.output_dir, f'ae-{epoch + 1}-{image_key_str}.pth'))
            

            try:
                os.remove(os.path.join(args.output_dir, f'dis-{epoch + 1 - 3}-{image_key_str}.pth'))
                os.remove(os.path.join(args.output_dir, f'ae-{epoch + 1 - 3}-{image_key_str}.pth'))
            except:
                pass

        # accelerator.wait_for_everyone() 
        gc.collect()
        torch.cuda.empty_cache()


torch.save(autoencoder.state_dict(),
           os.path.join(args.output_dir, f'ae-final-{image_key_str}.pth'))

print("Training finished.")

