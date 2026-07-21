# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from .model import MiohRestorerV1
from .model_v2 import MiohRestorerV2, MultiScaleFusionRefiner
from .model_v3 import (
    CoreAIShiftAlignment,
    HierarchicalCoreAIShiftAlignment,
    MiohRestorerV3,
    SecondOrderShiftPropagation,
)
from .model_v4 import (
    MiohRestorerV4ExportWrapper,
    MiohRestorerV4Q,
    NormalizedShiftCorrelation,
)
from .losses_v4 import (
    MiohRestorerV4Loss,
    confidence_error_correlation,
    overlap_consistency_loss,
)
from .runner_v4 import MiohRestorerV4WindowRunner
from .model_v5 import (
    MiohRestorerV5,
    MiohRestorerV5Config,
    MiohRestorerV5DecoderExportWrapper,
    MiohRestorerV5Encoder,
    MiohRestorerV5EncoderExportWrapper,
    MiohRestorerV5ExportWrapper,
    MiohRestorerV5StatefulExportWrapper,
    V5_BUCKETS,
)
from .runner_v5 import (
    MiohRestorerV5StreamingRunner,
    NativeCrop,
    V5BucketHysteresis,
    cut_safe_window_indices,
    native_crop_for_center,
    native_tile_offsets,
    repair_isolated_mask_misses,
    required_v5_crop_size,
    select_v5_bucket,
    smooth_even_centers,
)
from .native_dataset_v5 import (
    MiohRestorerV5NativeDataset,
    V5BucketBatchSampler,
    V5NativeManifestEntry,
    crop_native_frame,
    read_v5_native_manifest,
)
from .curriculum_v5 import V5_STAGES, V5LossWeights, V5TrainingStage
from .losses_v5 import MiohRestorerV5Loss
from .adversarial import (
    TemporalPatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    temporal_discriminator_input,
)
from .training import (
    MaskedVGG16PerceptualLoss,
    RestorationLoss,
    masked_charbonnier_loss,
    masked_high_frequency_loss,
    masked_multiscale_structural_loss,
    masked_psnr,
    restoration_loss,
    run_training_sequence,
)

__all__ = [
    "MiohRestorerV1",
    "MiohRestorerV2",
    "MiohRestorerV3",
    "MultiScaleFusionRefiner",
    "CoreAIShiftAlignment",
    "HierarchicalCoreAIShiftAlignment",
    "SecondOrderShiftPropagation",
    "MiohRestorerV4Q",
    "MiohRestorerV4ExportWrapper",
    "NormalizedShiftCorrelation",
    "MiohRestorerV4Loss",
    "MiohRestorerV4WindowRunner",
    "MiohRestorerV5",
    "MiohRestorerV5Config",
    "MiohRestorerV5Encoder",
    "MiohRestorerV5ExportWrapper",
    "MiohRestorerV5EncoderExportWrapper",
    "MiohRestorerV5DecoderExportWrapper",
    "MiohRestorerV5StatefulExportWrapper",
    "MiohRestorerV5StreamingRunner",
    "V5_BUCKETS",
    "NativeCrop",
    "V5BucketHysteresis",
    "select_v5_bucket",
    "required_v5_crop_size",
    "smooth_even_centers",
    "native_crop_for_center",
    "native_tile_offsets",
    "cut_safe_window_indices",
    "repair_isolated_mask_misses",
    "MiohRestorerV5NativeDataset",
    "V5BucketBatchSampler",
    "V5NativeManifestEntry",
    "crop_native_frame",
    "read_v5_native_manifest",
    "V5_STAGES",
    "V5LossWeights",
    "V5TrainingStage",
    "MiohRestorerV5Loss",
    "confidence_error_correlation",
    "overlap_consistency_loss",
    "TemporalPatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "temporal_discriminator_input",
    "MaskedVGG16PerceptualLoss",
    "RestorationLoss",
    "masked_charbonnier_loss",
    "masked_high_frequency_loss",
    "masked_multiscale_structural_loss",
    "masked_psnr",
    "restoration_loss",
    "run_training_sequence",
]
