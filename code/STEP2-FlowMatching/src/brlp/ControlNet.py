from monai.networks.nets import (
    AutoencoderKL,
    PatchDiscriminator,
    DiffusionModelUNet,
    ControlNet
)
import torch

from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding
from collections.abc import Sequence
from torch import nn


class ControlNet_witht(ControlNet):
    def __init__(self, spatial_dims: int, in_channels: int, channels: Sequence[int] = (32, 64, 64, 64), **kwargs):
        super().__init__(spatial_dims, in_channels, channels=channels, **kwargs)
        self.in_channels = in_channels

        time_embed_dim = channels[0] * 4
        self.pred_time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            controlnet_cond: torch.Tensor,
            conditioning_scale: float = 1.0,
            context: torch.Tensor | None = None,
            class_labels: torch.Tensor | None = None,
            prediction_time: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: input tensor (N, C, H, W, [D]).
            timesteps: timestep tensor (N,).
            controlnet_cond: controlnet conditioning tensor (N, C, H, W, [D])
            conditioning_scale: conditioning scale.
            context: context tensor (N, 1, cross_attention_dim), where cross_attention_dim is specified in the model init.
            class_labels: context tensor (N, ).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        if prediction_time is not None:
            pred_time_emb = get_timestep_embedding(prediction_time,  self.block_out_channels[0])
            # t_emb = t_emb + pred_time_emb


        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if prediction_time is not None:
            pred_time_emb = pred_time_emb.to(dtype=x.dtype)
            pred_time_emb = self.pred_time_embed(pred_time_emb)

            # print("emb", emb.shape, pred_time_emb.shape)
            emb = emb + pred_time_emb


        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        h += controlnet_cond

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        # 6. Control net blocks
        controlnet_down_block_res_samples = []

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample: torch.Tensor = self.controlnet_mid_block(h)

        # 6. scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        return down_block_res_samples, mid_block_res_sample