import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor
from lada.utils.mps_utils import serialized_mps_execution

class BasicvsrppMosaicRestorer:
    # Fixed-shape source models may overlap a small number of calls. Variable
    # Swift/Core AI already executes chunks sequentially, so retaining every
    # completed output until the final chunk only increases peak memory.
    stream_model_chunks = False

    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

    def restore(
        self,
        video: list[ImageTensor],
        max_frames=-1,
        temporal_overlap: int = 8,
        enable_crossfade: bool = True,
    ) -> list[ImageTensor]:
        if self.device.type == 'mps':
            with serialized_mps_execution():
                return self._restore_unlocked(
                    video,
                    max_frames=max_frames,
                    temporal_overlap=temporal_overlap,
                    enable_crossfade=enable_crossfade,
                )
        return self._restore_unlocked(
            video,
            max_frames=max_frames,
            temporal_overlap=temporal_overlap,
            enable_crossfade=enable_crossfade,
        )

    def _restore_unlocked(
        self,
        video: list[ImageTensor],
        max_frames=-1,
        temporal_overlap: int = 8,
        enable_crossfade: bool = True,
    ) -> list[ImageTensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            inference_view = torch.stack([x.permute(2, 0, 1) for x in video], dim=0).to(device=self.device).to(dtype=self.dtype).div_(255.0).unsqueeze(0)

            if max_frames > 0:
                result = self._restore_chunked_with_overlap(
                    inference_view,
                    max_frames,
                    temporal_overlap=temporal_overlap,
                    enable_crossfade=enable_crossfade,
                )
            else:
                result = self.model(inputs=inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
            result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result

    def _run_model_chunks(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.model(inputs=chunk) for chunk in chunks]

    def _restore_chunked_with_overlap(
        self,
        inference_view: torch.Tensor,
        max_frames: int,
        temporal_overlap: int = 8,
        enable_crossfade: bool = True,
    ) -> torch.Tensor:
        frame_count = inference_view.shape[1]
        if frame_count <= max_frames:
            return self.model(inputs=inference_view)

        overlap = min(max(0, temporal_overlap), max_frames // 2, frame_count - 1)
        if overlap <= 0:
            chunks = [
                inference_view[:, start:start + max_frames]
                for start in range(0, frame_count, max_frames)
            ]
            return torch.cat(self._run_model_chunks(chunks), dim=1)

        stride = max_frames - overlap
        output_accum = None
        weight_accum = None
        starts = []
        start = 0
        while start < frame_count:
            starts.append(start)
            if start + max_frames >= frame_count:
                break
            start += stride
        chunk_specs = []
        for start in starts:
            end = min(frame_count, start + max_frames)
            model_start = start
            if end - start < max_frames:
                model_start = max(0, end - max_frames)
            chunk_specs.append((
                inference_view[:, model_start:end],
                start - model_start,
                end - start,
            ))
        outputs = None
        if not self.stream_model_chunks:
            chunks = [chunk for chunk, _offset, _length in chunk_specs]
            outputs = self._run_model_chunks(chunks)
        for chunk_index, (start, chunk_spec) in enumerate(
            zip(starts, chunk_specs, strict=True)
        ):
            if outputs is None:
                output = self._run_model_chunks([chunk_spec[0]])[0]
            else:
                output = outputs[chunk_index]
            end = min(frame_count, start + max_frames)
            _chunk, output_offset, output_length = chunk_spec
            output = output[:, output_offset:output_offset + output_length]
            if output_accum is None:
                output_shape = (output.shape[0], frame_count, *output.shape[2:])
                output_accum = torch.zeros(output_shape, dtype=output.dtype, device=output.device)
                weight_accum = torch.zeros((output.shape[0], frame_count, 1, 1, 1), dtype=output.dtype, device=output.device)

            weights = torch.ones((output.shape[1],), dtype=output.dtype, device=output.device)
            if enable_crossfade and chunk_index > 0:
                ramp = torch.arange(1, min(overlap, output.shape[1]) + 1, dtype=output.dtype, device=output.device) / (overlap + 1)
                weights[: ramp.numel()] = ramp
            if enable_crossfade and end < frame_count:
                ramp_len = min(overlap, output.shape[1])
                ramp = torch.arange(ramp_len, 0, -1, dtype=output.dtype, device=output.device) / (overlap + 1)
                weights[-ramp_len:] = torch.minimum(weights[-ramp_len:], ramp)

            weight_view = weights.view(1, output.shape[1], 1, 1, 1)
            output_accum[:, start:end] += output * weight_view
            weight_accum[:, start:end] += weight_view

        assert output_accum is not None and weight_accum is not None
        return output_accum / weight_accum.clamp_min(torch.finfo(weight_accum.dtype).eps)
