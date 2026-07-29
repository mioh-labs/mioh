# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import glob
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

from lada.models.basicvsrpp.mmagic.data_sample import DataSample
from lada.models.basicvsrpp.mmagic.registry import DATASETS

import lada.utils.video_utils as video_utils
from lada.utils import random_utils, transforms as realesrgan_transforms, image_utils
from lada.utils.mosaic_utils import addmosaic_base, get_random_parameters_by_block_size
from lada.utils.image_utils import unpad_image, pad_image_by_pad, repad_image, scale_pad
from lada.datasetcreation.restoration_dataset_metadata import RestorationDatasetMetadataV2

from torchvision.transforms import transforms as torchvision_transforms

def create_degradation_pipeline(lq_size, resize=True):
    transforms = []
    if resize:
        transforms.append(realesrgan_transforms.ResizeFrames(lq_size))
    transforms.extend([
        realesrgan_transforms.VideoCompression(p=0.9, codecs=['libx264', 'libx265', 'libvpx-vp9', 'mpeg2video'], codec_probs=[0.3, 0.3, 0.3, 0.1],
                                               crf_ranges={'libx264': (16, 28), 'libx265': (20, 36)},
                                               bitrate_ranges={'libvpx-vp9': (6_000, 16_000), 'mpeg2video': (18_000, 40_000)}),
        realesrgan_transforms.GaussianBlur(sigma_range=[1., 4.], p=0.3),
        realesrgan_transforms.GaussianNoise(snr=50, p=0.2),
        realesrgan_transforms.VideoCompression(p=0.15, codecs=['libx264'],
                                               codec_probs=[1.],
                                               crf_ranges={'libx264': (24, 28)},
                                               bitrate_ranges={}),
    ])
    return torchvision_transforms.Compose(transforms)


def _as_single_channel_mask(mask):
    if mask.ndim == 3:
        return np.max(mask, axis=2)
    return mask


def _pad_frames_to_size(frames, size, mode='zero'):
    h, w = frames[0].shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    pad = (
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
    )
    if not any(pad):
        return frames, pad
    return [pad_image_by_pad(frame, pad, mode=mode) for frame in frames], pad


def _native_roi_crop(img_gts, img_lqs, masks, size, rng_random):
    """Crop one native-resolution tile shared by the whole temporal window.

    The crop is anchored on the union of the generated mosaic masks. This
    preserves native pixels and temporal alignment instead of resizing the
    complete tracked scene to a square.
    """
    img_gts, pad = _pad_frames_to_size(img_gts, size)
    img_lqs, _ = _pad_frames_to_size(img_lqs, size)
    masks, _ = _pad_frames_to_size(masks, size)

    h, w = img_gts[0].shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        union_mask = np.maximum(union_mask, _as_single_channel_mask(mask))

    ys, xs = np.where(union_mask > 0)
    if len(ys):
        selected = rng_random.randrange(len(ys))
        anchor_y, anchor_x = int(ys[selected]), int(xs[selected])
        jitter = max(1, size // 8)
        anchor_y += rng_random.randint(-jitter, jitter)
        anchor_x += rng_random.randint(-jitter, jitter)
    else:
        anchor_y, anchor_x = h // 2, w // 2

    top = max(0, min(h - size, anchor_y - size // 2))
    left = max(0, min(w - size, anchor_x - size // 2))
    # Keep the crop phase stable for pixel-unshuffle-compatible successors.
    top -= top % 2
    left -= left % 2

    def crop(frames):
        return [frame[top:top + size, left:left + size] for frame in frames]

    return crop(img_gts), crop(img_lqs), crop(masks)


def _rotate_mask(mask, degrees):
    h, w = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1)
    return cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

@DATASETS.register_module()
class MosaicVideoDataset(data.Dataset):
    def __init__(self, **opt):
        super(MosaicVideoDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.lq_size = opt.get('lq_size', 256)
        self.meta_root = Path(opt['metadata_root_dir'])
        self.use_hflip = opt.get('use_hflip', False)
        self.degrade = opt.get('degrade', False)
        self.max_frame_count = opt['num_frame']
        self.min_frame_count = opt['min_num_frame'] if 'min_num_frame' in opt else opt['num_frame']
        self.random_mosaic_params = opt.get('random_mosaic_params', True)
        self.repeatable_random = opt.get('repeatable_random', False)
        self.filter_watermark = opt.get('filter_watermark', False)
        self.filter_nudenet_nsfw = opt.get('filter_nudenet_nsfw', False)
        self.filter_video_quality = opt.get('filter_video_quality', False)
        self.native_roi_crop = opt.get('native_roi_crop', False)
        self.return_mosaic_mask = opt.get('return_mosaic_mask', False)
        self.rotation_probability = opt.get('rotation_probability', 0.3)
        self.filter_watermark_thresh = 0.1
        self.repad = True

        self.metadata = []
        for meta_path in glob.glob(os.path.join(opt['metadata_root_dir'], '*')):
            meta = RestorationDatasetMetadataV2.from_json_file(meta_path)
            if meta.frames_count < self.min_frame_count:
                continue
            if self.filter_watermark and meta.watermark_detected:
                continue
            if self.filter_nudenet_nsfw and not meta.nudenet_nsfw_detected:
                continue
            if self.filter_video_quality and meta.video_quality and meta.video_quality.overall < self.filter_video_quality:
                continue
            self.metadata.append(meta)

    def get_mosaic_params(self, meta: RestorationDatasetMetadataV2):
        if self.random_mosaic_params:
            mosaic_size, mosaic_mod, mosaic_rectangle_ratio, mosaic_feather_size = get_random_parameters_by_block_size(meta.base_mosaic_block_size.mosaic_size_v1_normal, randomize_size=True, repeatable_random=self.repeatable_random)
        else:
            mosaic_size, mosaic_mod, mosaic_rectangle_ratio, mosaic_feather_size = meta.mosaic.mosaic_size, meta.mosaic.mod, meta.mosaic.rect_ratio, meta.mosaic.feather_size
        return mosaic_size, mosaic_mod, mosaic_rectangle_ratio, mosaic_feather_size

    def get_end_frame_index(self, meta):
        rng_random, _ = random_utils.get_rngs(self.repeatable_random)
        if self.max_frame_count == -1:
            # select the full clip
            start_frame_idx = 0
            end_frame_idx = meta.frames_count - 1
        else:
            # randomly select shorter clip of length num_frame
            start_frame_idx = rng_random.randint(0, meta.frames_count - self.max_frame_count)
            end_frame_idx = start_frame_idx + self.max_frame_count
        return end_frame_idx, start_frame_idx

    def __getitem__(self, index):
        rng_random, _ = random_utils.get_rngs(self.repeatable_random)
        meta = self.metadata[index]

        end_frame_idx, start_frame_idx = self.get_end_frame_index(meta)

        pads = meta.pad[start_frame_idx:end_frame_idx]

        vid_gt_path = str(Path(self.meta_root).joinpath(meta.relative_nsfw_video_path))
        img_gts = video_utils.read_video_frames(vid_gt_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx)

        h, w = img_gts[0].shape[:2]
        scale_h = h / self.lq_size
        scale_w = w / self.lq_size
        scaled_pads = [scale_pad(pad, scale_h, scale_w) for pad in pads]
        mask_lqs = None

        if not self.random_mosaic_params:
            vid_lq_path = str(Path(self.meta_root).joinpath(meta.relative_mosaic_nsfw_video_path))
            img_lqs = video_utils.read_video_frames(vid_lq_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx)
            if self.return_mosaic_mask or self.native_roi_crop:
                vid_mask_gt_path = str(Path(self.meta_root).joinpath(meta.relative_mask_video_path))
                mask_lqs = video_utils.read_video_frames(
                    vid_mask_gt_path,
                    float32=False,
                    start_idx=start_frame_idx,
                    end_idx=end_frame_idx,
                    binary_frames=True,
                )
        else:
            vid_mask_gt_path = str(Path(self.meta_root).joinpath(meta.relative_mask_video_path))
            mask_gts = video_utils.read_video_frames(vid_mask_gt_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx, binary_frames=True)
            mosaic_size, mosaic_mod, mosaic_rectangle_ratio, mosaic_feather_size = self.get_mosaic_params(meta)

            img_lqs = []
            mask_lqs = []
            for img_gt, mask_gt, pad in zip(img_gts, mask_gts, pads):
                img_lq, mask_lq = addmosaic_base(unpad_image(img_gt, pad),
                                                 unpad_image(mask_gt, pad),
                                                 mosaic_size,
                                                 model=mosaic_mod,
                                                 rect_ratio=mosaic_rectangle_ratio,
                                                 feather=mosaic_feather_size)
                img_lqs.append(pad_image_by_pad(img_lq, pad))
                mask_lqs.append(pad_image_by_pad(mask_lq, pad))

        if self.native_roi_crop:
            if mask_lqs is None:
                raise RuntimeError('native_roi_crop requires a mask video')
            if self.repad:
                img_lqs = repad_image(img_lqs, pads, mode='zero')
                img_gts = repad_image(img_gts, pads, mode='zero')
                mask_lqs = repad_image(mask_lqs, pads, mode='zero')
            img_gts, img_lqs, mask_lqs = _native_roi_crop(
                img_gts,
                img_lqs,
                mask_lqs,
                self.lq_size,
                rng_random,
            )
            if self.degrade:
                degrade = create_degradation_pipeline(self.lq_size, resize=False)
                img_lqs = degrade(img_lqs)
        else:
            if self.degrade and self.random_mosaic_params:
                degrade = create_degradation_pipeline(self.lq_size)
                img_lqs = degrade(img_lqs)
            img_gts = video_utils.resize_video_frames(img_gts, self.lq_size)
            img_lqs = video_utils.resize_video_frames(img_lqs, self.lq_size)
            if mask_lqs is not None:
                mask_lqs = video_utils.resize_video_frames(mask_lqs, self.lq_size)

            if self.repad:
                img_lqs = repad_image(img_lqs, scaled_pads, mode='zero')
                img_gts = repad_image(img_gts, scaled_pads, mode='zero')
                if mask_lqs is not None:
                    mask_lqs = repad_image(mask_lqs, scaled_pads, mode='zero')

        if self.use_hflip and rng_random.random() < 0.5:
            img_gts = [np.fliplr(img) for img in img_gts]
            img_lqs = [np.fliplr(img) for img in img_lqs]
            if mask_lqs is not None:
                mask_lqs = [np.fliplr(mask) for mask in mask_lqs]

        if rng_random.random() < self.rotation_probability:
            rotation_deg = rng_random.choice([-2, -1, 1, 2])
            img_lqs = [image_utils.rotate(img, rotation_deg) for img in img_lqs]
            img_gts = [image_utils.rotate(img, rotation_deg) for img in img_gts]
            if mask_lqs is not None:
                mask_lqs = [_rotate_mask(mask, rotation_deg) for mask in mask_lqs]

        img_gts = image_utils.img2tensor(img_gts, float32=False, bgr2rgb=True)
        img_lqs = image_utils.img2tensor(img_lqs, float32=False, bgr2rgb=True)
        mask_tensor = None
        if mask_lqs is not None:
            mask_tensor = torch.stack([
                torch.from_numpy(
                    np.ascontiguousarray(_as_single_channel_mask(mask))
                ).float().unsqueeze(0) / 255.0
                for mask in mask_lqs
            ], dim=0)

        #print(f"selected from dataset: {clip_name}--({start_frame_idx:06d}-{end_frame_idx:06d})")

        data_sample = DataSample(gt_img=torch.stack(img_gts, dim=0))
        if self.return_mosaic_mask:
            if mask_tensor is None:
                raise RuntimeError('return_mosaic_mask requires a mask video')
            data_sample.mask = mask_tensor
        data_sample.set_predefined_data({
            'img': img_lqs,
            'img_channel_order': 'rgb',
            'img_color_type': 'color',
            'gt_img': img_gts,
            'gt_path': vid_gt_path,
            'gt_channel_order': 'rgb',
            'gt_color_type': 'color',
            'key': meta.name,
            'fps': meta.fps
        })
        inputs = torch.stack(img_lqs, dim=0)
        # inputs = tensor (T,C,H,W)
        return {'inputs': inputs, 'data_samples': data_sample}

    def __len__(self):
        return len(self.metadata)
