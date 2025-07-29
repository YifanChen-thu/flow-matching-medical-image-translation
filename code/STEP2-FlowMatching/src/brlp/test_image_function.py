import torch

from torch.cuda.amp import autocast, GradScaler

from monai.networks.schedulers import DDPMScheduler
from monai.inferers import DiffusionInferer
from tqdm import tqdm
from monai.networks.schedulers import DDIMScheduler

from . import const
from . import utils
from . import networks
from . import (
    get_dataset_from_pd,
    sample_using_diffusion
)

import torch
import torch.nn.functional as F

from src.utils.image_utilis import standardize_images

def center_crop_or_pad_3d(img, target_shape):
    """
    Center crop or pad a 3D image to match the target shape.

    Args:
        img (Tensor): shape [B, C, D, H, W]
        target_shape (tuple): desired shape [D, H, W]

    Returns:
        Tensor: same B, C, but D, H, W adjusted
    """
    assert img.dim() == 5, "Expected 5D input (B, C, D, H, W): got {}".format(img.dim())
    _, _, d, h, w = img.shape
    td, th, tw = target_shape

    # Compute padding or cropping for each dim
    def get_crop_pad(current, target):
        pad = max(target - current, 0)
        crop = max(current - target, 0)
        pad_before = pad // 2
        pad_after = pad - pad_before
        crop_before = crop // 2
        crop_after = crop - crop_before
        return (pad_before, pad_after), (crop_before, crop_after)

    (pd1, pd2), (cd1, cd2) = get_crop_pad(d, td)
    (ph1, ph2), (ch1, ch2) = get_crop_pad(h, th)
    (pw1, pw2), (cw1, cw2) = get_crop_pad(w, tw)

    # Pad if needed
    img = F.pad(img, (pw1, pw2, ph1, ph2, pd1, pd2))  # Pad: W, H, D

    # Crop if needed
    img = img[:, :, cd1:cd1 + td, ch1:ch1 + th, cw1:cw1 + tw]

    return img


class BrLPDiffusion:
    def __init__(self, img_height=128, aekl_ckpt=None, diff_ckpt=None, cnet_ckpt=None, use_t=False):
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = DEVICE

        self.autoencoder = networks.init_autoencoder(aekl_ckpt)  #.to(DEVICE)
        self.diffusion   = networks.init_latent_diffusion(diff_ckpt) #.to(DEVICE)
        self.controlnet  = networks.init_controlnet(cnet_ckpt, use_t=use_t) #.to(DEVICE)
        self.use_t = use_t


        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            schedule='scaled_linear_beta',
            beta_start=0.0015,
            beta_end=0.0205
        )

        self.inferer = DiffusionInferer(scheduler=scheduler)
        self.scale_factor = 0.16933174431324005  #0.5

        # Evaluation mode
        self.autoencoder.eval()
        self.diffusion.eval()


        total_params = sum(p.numel() for p in self.diffusion.parameters())
        trainable_params = sum(p.numel() for p in self.diffusion.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    def sample_using_diffusion(
            self,
            # autoencoder: nn.Module,
            # diffusion: nn.Module,
            mri_latent: torch.Tensor,
            context: torch.Tensor,
            device: str,
            scale_factor: int = 1,
            num_training_steps: int = 1000,
            num_inference_steps: int = 50,
            schedule: str = 'scaled_linear_beta',
            beta_start: float = 0.0015,
            beta_end: float = 0.0205,
            verbose: bool = False,
            time_step=0
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
        # Using DDIM sampling from (Song et al., 2020) allowing for a
        # deterministic reverse diffusion process (except for the starting noise)
        # and a faster sampling with fewer denoising steps.
        scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                                  schedule=schedule,
                                  beta_start=beta_start,
                                  beta_end=beta_end,
                                  clip_sample=False)

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # the subject-specific variables and the progression-related
        # covariates are concatenated into a vector outside this function.
        context = context.unsqueeze(0).to(device).to(device)

        # drawing a random z_T ~ N(0,I)
        # print("const.LATENT_SHAPE_DM = ", const.LATENT_SHAPE_DM, mri_latent.shape, context.shape)

        mri_latent = mri_latent[0]
        z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)  # (122, 146, 122)
        z = torch.randn(mri_latent.shape).unsqueeze(0).to(device)
        fact = 0.2
        z = z * (1 - fact) + mri_latent * fact * self.scale_factor

        progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
        for t in progress_bar:
            with torch.no_grad():
                with autocast(enabled=True):
                    timestep = torch.tensor([t]).to(device)

                    # predict the noise
                    noise_pred = self.diffusion(
                        x=z.float(),
                        timesteps=timestep,
                        context=context.float(),
                    )

                    # the scheduler applies the formula to get the
                    # denoised step z_{t-1} from z_t and the predicted noise
                    z, _ = scheduler.step(noise_pred, t, z)

        # decode the latent
        z = z / scale_factor
        z = utils.to_vae_latent_trick(z.squeeze(0).cpu())  # ?
        x = self.autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
        x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
        return x

    def forward(self, data):
        # import torch
        torch.cuda.empty_cache()
        self.autoencoder.to(self.device)
        self.diffusion.to(self.device)

        # Image data processing
        imgs = data["followup_image"].to(self.device)
        # 1, 1, 128, 144, 128 -> 1, 1, 122, 146, 122
        imgs = center_crop_or_pad_3d(imgs, (128, 160, 128))

        tag_i = 0
        context = data["context"].to(self.device)[0]
        context = torch.tensor([[
            # (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age
            # (torch.randint(1, 2, (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            #
            (data['starting_age'][0] - const.AGE_MIN) / const.AGE_DELTA,
            (data['sex'][0] - const.SEX_MIN) / const.SEX_DELTA,
            # (data['starting_diagnosis'][0] - const.DIA_MIN) / const.DIA_DELTA,
            (torch.randint(1, 3, (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567,  # (mean) cerebral cortex
            0.539,  # (mean) hippocampus
            0.578,  # (mean) amygdala
            0.558,  # (mean) cerebral white matter
            0.30 * (tag_i + 1),  # variable size lateral ventricles
        ]]).float()

        imgs = standardize_images(imgs)

        mri_latent, _ = self.autoencoder.encode(imgs)

        # self.scale_factor = 1 / torch.std(mri_latent)

        # print(f"Scaling factor set to {self.scale_factor}")
        starting_a = data['starting_age'].to(self.device)
        target_a   = data['followup_age'].to(self.device)
        time_step = target_a - starting_a

        gen_img = self.sample_using_diffusion(
            # autoencoder=self.autoencoder,
            # diffusion=self.diffusion,
            context=context,
            mri_latent=mri_latent,
            device=self.device,
            scale_factor=self.scale_factor,

        )

        gen_img = gen_img.unsqueeze(0).unsqueeze(0)
        gen_img = center_crop_or_pad_3d(gen_img, (160, 160, 128))

        gen_img_cpu = gen_img.detach().cpu().numpy()

        self.autoencoder.cpu()
        self.diffusion.cpu()
        self.controlnet.cpu()

        return gen_img_cpu

    def forward_ae(self, data, mix_ratio=0):
        torch.cuda.empty_cache()
        self.autoencoder.to(self.device)
        # self.diffusion.to(self.device)

        # Image data processing
        imgs    = data["source"].to(self.device)
        targets = data["target"].to(self.device)

        imgs    = center_crop_or_pad_3d(imgs, (128, 160, 128))  # 128, 160, 128
        targets = center_crop_or_pad_3d(targets, (128, 160, 128))  # 128, 160, 128

        tag_i = 0
        # context = data["context"].to(self.device)[0]
        # imgs = standardize_images(imgs)
        # targets = standardize_images(targets)

        mri_latent, _ = self.autoencoder.encode(imgs)
        target_latent, _ = self.autoencoder.encode(targets)

        mri_latent = (1 - mix_ratio) * mri_latent + mix_ratio * target_latent

        # print("mri_latent ae = ", mri_latent.shape)

        x = self.autoencoder.decode_stage_2_outputs(mri_latent)
        # pred_targets = self.autoencoder.decode_stage_2_outputs(target_latent)

        # x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)

        gen_img = x # .unsqueeze(0) #.unsqueeze(0)

        # print("gen_img = ", gen_img.shape)

        gen_img = center_crop_or_pad_3d(gen_img, (160, 160, 128))  #128, 144, 128))

        gen_img_cpu = gen_img.detach().cpu().numpy()

        return gen_img_cpu

    def sample_using_controlnet_and_z(
            self,
            starting_z: torch.Tensor,
            starting_a: int,
            context: torch.Tensor,
            device: str,
            scale_factor: int = 1,
            average_over_n: int = 1,
            num_training_steps: int = 1000,
            num_inference_steps: int = 50,
            prediction_time=0,
            schedule: str = 'scaled_linear_beta',
            beta_start: float = 0.0015,
            beta_end: float = 0.0205,
            verbose: bool = False
    ) -> torch.Tensor:
        """
        The inference process described in the paper.

        Args:
            autoencoder (nn.Module): the KL autoencoder
            diffusion (nn.Module): the UNet
            controlnet (nn.Module): the ControlNet
            starting_z (torch.Tensor): the latent from the MRI of the starting visit
            starting_a (int): the starting age
            context (torch.Tensor): the covariates
            device (str): the device ('cuda' or 'cpu')
            scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
            average_over_n (int, optional): LAS parameter m. Defaults to 1.
            num_training_steps (int, optional): T parameter. Defaults to 1000.
            num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
            schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
            beta_start (float, optional): noise starting level. Defaults to 0.0015.
            beta_end (float, optional): noise ending level. Defaults to 0.0205.
            verbose (bool, optional): print progression bar. Defaults to True.

        Returns:
            torch.Tensor: the inferred follow-up MRI
        """
        # Using DDIM sampling from (Song et al., 2020) allowing for a
        # deterministic reverse diffusion process (except for the starting noise)
        # and a faster sampling with fewer denoising steps.
        scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                                  schedule=schedule,
                                  beta_start=beta_start,
                                  beta_end=beta_end,
                                  clip_sample=False)

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # preparing controlnet spatial condition.
        starting_z = starting_z.unsqueeze(0).to(device)
        concatenating_age = torch.tensor([starting_a]).view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(
            device)
        controlnet_condition = torch.cat([starting_z, concatenating_age], dim=1).to(device)

        # the subject-specific variables and the progression-related
        # covariates are concatenated into a vector outside this function.
        context = context.unsqueeze(0).unsqueeze(0).to(device)

        # if performing LAS, we repeat the inputs for the diffusion process
        # m times (as specified in the paper) and perform the reverse diffusion
        # process in parallel to avoid overheads.
        if average_over_n > 1:
            context = context.repeat(average_over_n, 1, 1)
            controlnet_condition = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1)

            # this is z_T - the starting noise.
        z = torch.randn(average_over_n, *starting_z.shape[1:]).to(device)

        progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

        for t in progress_bar:
            with torch.no_grad():
                with autocast(enabled=True):
                    # convert the timestep to a tensor.
                    timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                    # get the intermediate features from the ControlNet
                    # by feeding the starting latent, the covariates and the timestep
                    if self.use_t:
                        down_h, mid_h = self.controlnet(
                            x=z.float(),
                            timesteps=timestep,
                            context=context,
                            controlnet_cond=controlnet_condition.float(),
                        )
                    else:
                        down_h, mid_h = self.controlnet(
                            x=z.float(),
                            timesteps=timestep,
                            context=context,
                            controlnet_cond=controlnet_condition.float()
                        )

                    # the diffusion takes the intermediate features and predicts
                    # the noise. This is why we conceptualize the two networks as
                    # as a unified network.
                    noise_pred = self.diffusion(
                        x=z.float(),
                        timesteps=timestep,
                        context=context.float(),
                        down_block_additional_residuals=down_h,
                        mid_block_additional_residual=mid_h
                    )

                    # the scheduler applies the formula to get the
                    # denoised step z_{t-1} from z_t and the predicted noise
                    z, _ = scheduler.step(noise_pred, t, z)

        # Here we conclude Latent Average Stabilization by averaging
        # m different latents from m different samplings.
        z = (z / scale_factor).sum(axis=0) / average_over_n
        # z = utils.to_vae_latent_trick(z.squeeze(0).cpu())

        # print("z = ", z.shape)
        # decode the latent using the Decoder block from the KL autoencoder.
        x = self.autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))  #.squeeze(0)
        # x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
        return x

    def forward_cnet(self, data):
        torch.cuda.empty_cache()
        self.autoencoder.to(self.device)
        self.diffusion.to(self.device)
        self.controlnet.to(self.device)


        # Image data processing
        imgs = data["followup_image"].to(self.device)
        starting_a = data["starting_age"].to(self.device)[0]
        # print("starting_a = ", starting_a)

        # 1, 1, 128, 144, 128 -> 1, 1, 122, 146, 122
        imgs = center_crop_or_pad_3d(imgs, (128, 160, 128))  # 128, 160, 128

        tag_i = 0
        # context = data["context"].to(self.device)[0]
        context = torch.tensor([[
            # (torch.randint(60, 70, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age
            # (torch.randint(1, 2, (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            # (torch.randint(1, 3, (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            (data['starting_age'][0] - const.AGE_MIN) / const.AGE_DELTA,
            (data['sex'][0] - const.SEX_MIN) / const.SEX_DELTA,
            # (data['starting_diagnosis'][0] - const.DIA_MIN) / const.DIA_DELTA,
            (torch.randint(1, 3, (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567,  # (mean) cerebral cortex
            0.539,  # (mean) hippocampus
            0.578,  # (mean) amygdala
            0.558,  # (mean) cerebral white matter
            0.30 * (tag_i + 1),  # variable size lateral ventricles
        ]])[0].float()

        # context = torch.stack([
        #     (data['starting_age'][0] - const.AGE_MIN) / const.AGE_DELTA,
        #     (data['sex'][0] - const.SEX_MIN) / const.SEX_DELTA,
        #     (data['starting_diagnosis'][0] - const.DIA_MIN) / const.DIA_DELTA,
        #     data['starting_cerebral_cortex'][0],
        #     data['starting_hippocampus'][0],
        #     data['starting_amygdala'][0],
        #     data['starting_cerebral_white_matter'][0],
        #     data['starting_lateral_ventricle'][0]
        # ]).float()

        # context = context[0]
        imgs = standardize_images(imgs)
        mri_latent, _ = self.autoencoder.encode(imgs)
        mri_latent = mri_latent[0]

        # self.scale_factor = 1 / torch.std(mri_latent)
        # print(f"Scaling factor set to {self.scale_factor} with mri_latent=", mri_latent.shape)

        starting_a = data['starting_age'].to(self.device)
        target_a   = data['followup_age'].to(self.device)
        time_step = target_a - starting_a

        gen_img = self.sample_using_controlnet_and_z(
            starting_z=mri_latent,
            starting_a=starting_a,
            context=context,
            device=self.device,
            scale_factor=self.scale_factor,
            prediction_time = time_step
        )

        # gen_img = gen_img.unsqueeze(0).unsqueeze(0)
        gen_img = center_crop_or_pad_3d(gen_img, (160, 160, 128))

        gen_img_cpu = gen_img.detach().cpu().numpy()

        return gen_img_cpu