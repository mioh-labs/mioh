# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Training-only adversarial components for MiohRestorer models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class TemporalPatchDiscriminator(nn.Module):
    """Judge spatial appearance and adjacent-frame motion inside the ROI."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        # RGB current frame + RGB temporal difference + one mask channel.
        channels = (7, base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        layers: list[nn.Module] = []
        for index in range(len(channels) - 1):
            layers.extend(
                (
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
        layers.extend(
            (
                nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channels * 8, 1, 3, padding=1),
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(values)
        return logits

    def forward_features(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return logits plus multiscale activations for feature matching."""
        if values.ndim != 4 or values.shape[1] != 7:
            raise ValueError("discriminator input must have shape [N,7,H,W]")
        features: list[torch.Tensor] = []
        output = values
        for layer in self.network:
            output = layer(output)
            if isinstance(layer, nn.LeakyReLU):
                features.append(output)
        return output, tuple(features)


class SpectralUNetDiscriminator(nn.Module):
    """Dense ROI discriminator with spectral-normalized U-Net skip paths.

    Unlike :class:`TemporalPatchDiscriminator`, this network returns one logit
    per input pixel.  The dense output prevents a small number of PatchGAN
    cells from averaging together unrelated high-frequency structures.  The
    topology follows the discriminator already used by the BasicVSR++ GAN
    training assets, while accepting Mioh's seven-channel conditioned input.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.conv_0 = nn.Conv2d(7, base_channels, 3, 1, 1)
        self.conv_1 = spectral_norm(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False)
        )
        self.conv_2 = spectral_norm(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False)
        )
        self.conv_3 = spectral_norm(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False)
        )
        self.conv_4 = spectral_norm(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, 1, 1, bias=False)
        )
        self.conv_5 = spectral_norm(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        )
        self.conv_6 = spectral_norm(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=False)
        )
        self.conv_7 = spectral_norm(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False)
        )
        self.conv_8 = spectral_norm(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False)
        )
        self.conv_9 = nn.Conv2d(base_channels, 1, 3, 1, 1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(values)
        return logits

    def forward_features(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if values.ndim != 4 or values.shape[1] != 7:
            raise ValueError("discriminator input must have shape [N,7,H,W]")
        feat_0 = self.activation(self.conv_0(values))
        feat_1 = self.activation(self.conv_1(feat_0))
        feat_2 = self.activation(self.conv_2(feat_1))
        feat_3 = self.activation(self.conv_3(feat_2))

        feat_4 = self.activation(self.conv_4(self.upsample(feat_3))) + feat_2
        feat_5 = self.activation(self.conv_5(self.upsample(feat_4))) + feat_1
        feat_6 = self.activation(self.conv_6(self.upsample(feat_5))) + feat_0
        feat_7 = self.activation(self.conv_7(feat_6))
        feat_8 = self.activation(self.conv_8(feat_7))
        return self.conv_9(feat_8), (
            feat_0,
            feat_1,
            feat_2,
            feat_3,
            feat_4,
            feat_5,
            feat_6,
            feat_7,
            feat_8,
        )


def temporal_discriminator_input(
    video: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    frame_stride: int = 4,
    image_size: int = 192,
) -> torch.Tensor:
    """Build masked current-frame/motion pairs with clean target context."""
    if video.shape != target.shape or video.ndim != 5:
        raise ValueError("video and target must have matching B,T,C,H,W shapes")
    if masks.shape != (video.shape[0], video.shape[1], 1, *video.shape[-2:]):
        raise ValueError("masks do not match video")
    if video.shape[1] < 2:
        raise ValueError("temporal discriminator requires at least two frames")
    if frame_stride <= 0 or image_size < 32:
        raise ValueError("invalid temporal discriminator sampling settings")

    composed = video * masks + target * (1.0 - masks)
    current = composed[:, 1::frame_stride]
    previous = composed[:, 0:-1:frame_stride]
    pair_count = min(current.shape[1], previous.shape[1])
    current = current[:, :pair_count]
    previous = previous[:, :pair_count]
    pair_mask = torch.minimum(
        masks[:, 1::frame_stride][:, :pair_count],
        masks[:, 0:-1:frame_stride][:, :pair_count],
    )
    motion = current - previous
    values = torch.cat((current, motion, pair_mask), dim=2)
    values = values.reshape(-1, 7, *values.shape[-2:])
    return F.interpolate(
        values,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def roi_temporal_discriminator_input(
    video: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    frame_stride: int = 1,
    image_size: int = 192,
    crop_padding: int = 16,
    minimum_crop_size: int = 96,
    include_motion: bool = False,
    normalize_secondary_rms: bool = False,
    condition_on_target_rgb: bool = True,
) -> torch.Tensor:
    """Build ROI-centred PatchGAN inputs without shrinking the whole frame.

    The primary RGB channels are clean target context for both real and fake
    samples. The generator therefore receives adversarial gradients only
    through the secondary high-frequency (Stage 4) or motion (Stage 5)
    channels; it cannot win by shifting colour, confidence, or low-frequency
    structure. The mask keeps the discriminator focused on the composited ROI.
    """

    if video.shape != target.shape or video.ndim != 5:
        raise ValueError("video and target must have matching B,T,C,H,W shapes")
    if masks.shape != (video.shape[0], video.shape[1], 1, *video.shape[-2:]):
        raise ValueError("masks do not match video")
    if video.shape[1] < 2:
        raise ValueError("ROI discriminator requires at least two frames")
    if frame_stride <= 0 or image_size < 32:
        raise ValueError("invalid ROI discriminator sampling settings")
    if crop_padding < 0 or minimum_crop_size < 1:
        raise ValueError("invalid ROI discriminator crop settings")

    composed = video * masks + target * (1.0 - masks)
    height, width = video.shape[-2:]
    crops: list[torch.Tensor] = []
    for batch_index in range(video.shape[0]):
        for current_index in range(1, video.shape[1], frame_stride):
            previous_index = current_index - 1
            pair_mask = torch.maximum(
                masks[batch_index, current_index],
                masks[batch_index, previous_index],
            )
            active = torch.nonzero(pair_mask[0] > 0, as_tuple=False)
            if not active.numel():
                continue
            top = int(active[:, 0].min()) - crop_padding
            bottom = int(active[:, 0].max()) + 1 + crop_padding
            left = int(active[:, 1].min()) - crop_padding
            right = int(active[:, 1].max()) + 1 + crop_padding
            crop_height = bottom - top
            crop_width = right - left
            side = min(max(crop_height, crop_width, minimum_crop_size), height, width)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            top = min(max(center_y - side // 2, 0), height - side)
            left = min(max(center_x - side // 2, 0), width - side)
            bottom = top + side
            right = left + side

            current = composed[
                batch_index, current_index, :, top:bottom, left:right
            ]
            target_condition = target[
                batch_index, current_index, :, top:bottom, left:right
            ]
            previous = composed[
                batch_index, previous_index, :, top:bottom, left:right
            ]
            secondary = (
                current - previous
                if include_motion
                else (
                    current
                    - F.avg_pool2d(
                        current[None],
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        count_include_pad=False,
                    )[0]
                )
            )
            if normalize_secondary_rms:
                rms = secondary.square().mean().add(1e-8).sqrt()
                secondary = (secondary / rms * 0.25).clamp(-1.0, 1.0)
            else:
                secondary = secondary * (4.0 if include_motion else 8.0)
            cropped_mask = pair_mask[:, top:bottom, left:right]
            primary = target_condition if condition_on_target_rgb else current
            normalized_current = primary * 2.0 - 1.0
            crop = torch.cat(
                (normalized_current, secondary, cropped_mask), dim=0
            )[None]
            crops.append(
                F.interpolate(
                    crop,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )

    if not crops:
        return video.new_empty((0, 7, image_size, image_size))
    return torch.stack(crops)


def discriminator_hinge_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    real_loss = F.relu(1.0 - real_logits)
    fake_loss = F.relu(1.0 + fake_logits)
    if weights is None:
        return real_loss.mean() + fake_loss.mean()
    if weights.shape != real_logits.shape or fake_logits.shape != real_logits.shape:
        raise ValueError("discriminator weights must match the logits")
    denominator = weights.sum().clamp_min(1.0)
    return (real_loss * weights).sum() / denominator + (
        fake_loss * weights
    ).sum() / denominator


def generator_hinge_loss(
    fake_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        return -fake_logits.mean()
    if weights.shape != fake_logits.shape:
        raise ValueError("generator weights must match the logits")
    return -(fake_logits * weights).sum() / weights.sum().clamp_min(1.0)


def discriminator_relativistic_pair_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """RpGAN discriminator loss for paired real/restored ROI logits.

    The critic is trained on the relative score ``real - fake`` rather than
    two independent absolute margins.  This is the relativistic paired loss
    used by SeedVR2 to reduce mode dropping during adversarial post-training.
    """

    if real_logits.shape != fake_logits.shape:
        raise ValueError("real/fake relativistic logits must match")
    loss = F.softplus(-(real_logits - fake_logits))
    if weights is None:
        return loss.mean()
    if weights.shape != loss.shape:
        raise ValueError("discriminator weights must match the logits")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def generator_relativistic_pair_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """RpGAN generator loss with the real critic branch held fixed."""

    if real_logits.shape != fake_logits.shape:
        raise ValueError("real/fake relativistic logits must match")
    loss = F.softplus(-(fake_logits - real_logits.detach()))
    if weights is None:
        return loss.mean()
    if weights.shape != loss.shape:
        raise ValueError("generator weights must match the logits")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def discriminator_feature_matching_loss(
    real_features: tuple[torch.Tensor, ...],
    fake_features: tuple[torch.Tensor, ...],
    discriminator_input: torch.Tensor,
) -> torch.Tensor:
    """Match ROI discriminator activations without chasing its output margin.

    The real branch is detached deliberately.  Each scale is averaged over
    channels first, then only PatchGAN cells overlapping the composited ROI
    contribute.  Averaging scales keeps the coefficient independent of the
    discriminator depth.
    """

    if not real_features or len(real_features) != len(fake_features):
        raise ValueError("real/fake discriminator features must be non-empty and match")
    losses: list[torch.Tensor] = []
    for real, fake in zip(real_features, fake_features, strict=True):
        if real.shape != fake.shape:
            raise ValueError("real/fake discriminator feature shapes must match")
        difference = (fake - real.detach()).abs().mean(dim=1, keepdim=True)
        weights = discriminator_roi_patch_weights(discriminator_input, difference)
        losses.append(
            (difference * weights).sum()
            / (weights.sum().clamp_min(1.0) * difference.shape[1])
        )
    return torch.stack(losses).mean()


def discriminator_roi_patch_weights(
    discriminator_input: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Select PatchGAN cells whose receptive fields overlap the edited ROI.

    Real and generated crops are identical outside the compositor mask. If
    those cells participate in the hinge mean, their opposing real/fake
    gradients cancel and drown the few cells that can distinguish restoration
    detail. Max-downsampling followed by one-cell dilation keeps the edited ROI
    and its seam while excluding unrelated clean context.
    """

    if discriminator_input.ndim != 4 or discriminator_input.shape[1] != 7:
        raise ValueError("discriminator input must have shape [N,7,H,W]")
    if logits.ndim != 4 or logits.shape[0] != discriminator_input.shape[0]:
        raise ValueError("discriminator logits do not match the input batch")
    mask = F.adaptive_max_pool2d(
        discriminator_input[:, 6:7],
        logits.shape[-2:],
    )
    mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return (mask > 0).to(dtype=logits.dtype)
