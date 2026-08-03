# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch.nn.functional as F

from lada.models.basicvsrpp.mmagic.registry import MODELS
from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
from lada.models.basicvsrpp.mmagic.real_basicvsr import RealBasicVSR

@MODELS.register_module()
class BasicVSRPlusPlusGanNet(BasicVSRPlusPlusNet):
    def __init__(self,
                trainable_modules=None,
                **kwargs):

        super().__init__(**kwargs)
        self.spynet.requires_grad_(False)

        self.trainable_module_names = None
        if trainable_modules is not None:
            requested = tuple(str(name) for name in trainable_modules)
            if not requested:
                raise ValueError('trainable_modules cannot be empty')
            if len(set(requested)) != len(requested):
                raise ValueError('trainable_modules contains duplicates')

            available = set(dict(self.named_children()))
            unknown = sorted(set(requested) - available)
            if unknown:
                raise ValueError(
                    'unknown BasicVSR++ trainable modules: '
                    + ', '.join(unknown)
                )
            if 'spynet' in requested:
                raise ValueError('SPyNet cannot be enabled by trainable_modules')

            # A real freeze is required here.  An lr multiplier of zero still
            # builds the backward graph through flow warp/DCNv2 and allocates
            # optimizer state.  Fine-tuning only the reconstruction tail keeps
            # the deployed alignment and four propagation branches bit-stable.
            self.requires_grad_(False)
            for name in requested:
                getattr(self, name).requires_grad_(True)
            self.trainable_module_names = requested

    def train(self, mode=True):
        super().train(mode)
        if self.trainable_module_names is not None:
            trainable = set(self.trainable_module_names)
            for name, module in self.named_children():
                if name not in trainable:
                    module.eval()
        return self


    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        outputs = super().forward(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs

@MODELS.register_module()
class BasicVSRPlusPlusGan(RealBasicVSR):
    """RealBasicVSR model for real-world video super-resolution.

    Ref:
    Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 is_use_ema=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            is_use_sharpened_gt_in_pixel=False,
            is_use_sharpened_gt_in_percep=False,
            is_use_sharpened_gt_in_gan=False,
            is_use_ema=is_use_ema,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        # EMA is an inference shadow, never an optimization target.  Keeping
        # requires_grad enabled doubles the apparent trainable count and can
        # accidentally admit EMA tensors into broad optimizer selectors.
        if self.generator_ema is not None:
            self.generator_ema.requires_grad_(False)


    def extract_gt_data(self, data_samples):
        gt = data_samples.gt_img
        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)

        return gt_pixel, gt_percep, gt_gan


@MODELS.register_module()
class BasicVSRPlusPlusSharpGan(BasicVSRPlusPlusGan):
    """Sharpness-focused fine-tuning with ROI-only perceptual/GAN gradients."""

    def __init__(
        self,
        generator,
        discriminator=None,
        gan_loss=None,
        pixel_loss=None,
        roi_pixel_loss=None,
        perceptual_loss=None,
        high_frequency_loss=None,
        temporal_loss=None,
        roi_dilation=4,
        **kwargs,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            **kwargs,
        )
        self.high_frequency_loss = (
            MODELS.build(high_frequency_loss) if high_frequency_loss else None
        )
        self.roi_pixel_loss = (
            MODELS.build(roi_pixel_loss) if roi_pixel_loss else None
        )
        self.temporal_loss = MODELS.build(temporal_loss) if temporal_loss else None
        self.roi_dilation = roi_dilation

    def _expanded_mask(self, mask):
        if self.roi_dilation <= 0:
            return mask
        kernel = self.roi_dilation * 2 + 1
        return F.max_pool2d(
            mask, kernel_size=kernel, stride=1, padding=self.roi_dilation
        )

    def extract_gt_data(self, data_samples):
        gt_sequence = data_samples.gt_img
        mask_sequence = data_samples.mask.to(dtype=gt_sequence.dtype).clamp(0, 1)
        n, t, c, h, w = gt_sequence.shape
        gt_flat = gt_sequence.reshape(n * t, c, h, w)
        mask_flat = mask_sequence.reshape(n * t, 1, h, w)
        return (
            gt_flat.clone(),
            gt_flat.clone(),
            gt_flat.clone(),
            mask_flat,
            gt_sequence,
            mask_sequence,
        )

    def _roi_composite(self, prediction, batch_gt_data):
        gt = batch_gt_data[2]
        mask = self._expanded_mask(batch_gt_data[3])
        return prediction * mask + gt * (1.0 - mask)

    def prepare_discriminator_fake(self, fake_output, batch_gt_data):
        return self._roi_composite(fake_output, batch_gt_data)

    def g_step(self, batch_outputs, batch_gt_data):
        gt_pixel, gt_percep, _, mask, gt_sequence, mask_sequence = batch_gt_data
        fake_output, _ = batch_outputs
        fake_output = fake_output.view(gt_pixel.shape)
        fake_roi = self._roi_composite(fake_output, batch_gt_data)

        losses = {}
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(fake_output, gt_pixel)
        if self.roi_pixel_loss:
            losses['loss_pixel_roi'] = self.roi_pixel_loss(
                fake_output, gt_pixel, mask
            )
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(fake_roi, gt_percep)
            if loss_percep is not None:
                losses['loss_perceptual_roi'] = loss_percep
            if loss_style is not None:
                losses['loss_style_roi'] = loss_style
        if self.high_frequency_loss:
            losses['loss_high_frequency_roi'] = self.high_frequency_loss(
                fake_output, gt_pixel, mask
            )
        if self.temporal_loss:
            losses['loss_temporal_roi'] = self.temporal_loss(
                fake_output.view_as(gt_sequence),
                gt_sequence,
                mask_sequence,
            )
        if self.gan_loss and self.discriminator:
            fake_prediction = self.discriminator(fake_roi)
            losses['loss_gan_roi'] = self.gan_loss(
                fake_prediction, target_is_real=True, is_disc=False
            )
        return losses
