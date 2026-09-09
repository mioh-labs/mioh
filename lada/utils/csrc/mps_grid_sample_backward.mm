// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

/**
 * Native MPS implementation of aten::grid_sampler_2d_backward.
 *
 * PyTorch 2.12 ships an MPS forward kernel for grid_sampler_2d but no MPS
 * backward dispatch.  This extension follows the CUDA/CPU derivative and the
 * command-buffer integration used by mps-deform-conv.  Accumulation is done in
 * FP32 because Metal has no atomic add for half/bfloat values.
 */

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

std::mutex g_grid_sample_init_mutex;
std::atomic<bool> g_grid_sample_initialized{false};
id<MTLDevice> g_grid_sample_device = nil;
id<MTLLibrary> g_grid_sample_library = nil;
id<MTLComputePipelineState> g_grid_sample_backward_fp32 = nil;

NSString* grid_sample_metal_source() {
  return @R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int INTERPOLATION_NEAREST = 1;
constant int INTERPOLATION_BICUBIC = 2;
constant int PADDING_ZEROS = 0;
constant int PADDING_BORDER = 1;
constant int PADDING_REFLECTION = 2;

inline bool within_bounds_2d(int y, int x, int height, int width) {
  return y >= 0 && y < height && x >= 0 && x < width;
}

inline float safe_coordinate(float value) {
  // Keep float-to-int conversion defined for NaN, Inf and huge grid values.
  return (!isfinite(value) || value > 2147483520.0f || value < -2147483648.0f)
      ? -100.0f
      : value;
}

inline float unnormalize_set_grad(
    float coordinate,
    int size,
    bool align_corners,
    thread float& gradient) {
  if (align_corners) {
    gradient = float(size - 1) * 0.5f;
    return (coordinate + 1.0f) * float(size - 1) * 0.5f;
  }
  gradient = float(size) * 0.5f;
  return ((coordinate + 1.0f) * float(size) - 1.0f) * 0.5f;
}

inline float clip_set_grad(float value, int size, thread float& gradient) {
  if (value <= 0.0f) {
    gradient = 0.0f;
    return 0.0f;
  }
  float maximum = float(size - 1);
  if (value >= maximum) {
    gradient = 0.0f;
    return maximum;
  }
  gradient = 1.0f;
  return value;
}

inline float reflect_set_grad(
    float value,
    int twice_low,
    int twice_high,
    thread float& gradient) {
  if (twice_low == twice_high) {
    gradient = 0.0f;
    return 0.0f;
  }

  float minimum = float(twice_low) * 0.5f;
  float span = float(twice_high - twice_low) * 0.5f;
  value -= minimum;

  float sign = 1.0f;
  if (value < 0.0f) {
    sign = -1.0f;
    value = -value;
  }

  float extra = fmod(value, span);
  float flips = floor(value / span);
  if (fmod(flips, 2.0f) == 0.0f) {
    gradient = sign;
    return extra + minimum;
  }
  gradient = -sign;
  return span - extra + minimum;
}

inline float source_index_set_grad(
    float coordinate,
    int size,
    int padding_mode,
    bool align_corners,
    thread float& gradient) {
  float value = unnormalize_set_grad(coordinate, size, align_corners, gradient);

  if (padding_mode == PADDING_BORDER) {
    float clip_gradient;
    value = clip_set_grad(value, size, clip_gradient);
    gradient *= clip_gradient;
  } else if (padding_mode == PADDING_REFLECTION) {
    float reflection_gradient;
    if (align_corners) {
      value = reflect_set_grad(
          value, 0, 2 * (size - 1), reflection_gradient);
    } else {
      value = reflect_set_grad(
          value, -1, 2 * size - 1, reflection_gradient);
    }
    float clip_gradient;
    value = clip_set_grad(value, size, clip_gradient);
    gradient *= reflection_gradient * clip_gradient;
  }

  return safe_coordinate(value);
}

inline float unnormalize_only_set_grad(
    float coordinate,
    int size,
    bool align_corners,
    thread float& gradient) {
  return safe_coordinate(
      unnormalize_set_grad(coordinate, size, align_corners, gradient));
}

inline int bounded_index(
    int index,
    int size,
    int padding_mode,
    bool align_corners) {
  if (padding_mode == PADDING_ZEROS) {
    return index >= 0 && index < size ? index : -1;
  }
  if (padding_mode == PADDING_BORDER) {
    return clamp(index, 0, size - 1);
  }

  if (size <= 1) {
    return 0;
  }
  float ignored_gradient;
  float reflected;
  if (align_corners) {
    reflected = reflect_set_grad(
        float(index), 0, 2 * (size - 1), ignored_gradient);
  } else {
    reflected = reflect_set_grad(
        float(index), -1, 2 * size - 1, ignored_gradient);
  }
  return clamp(int(reflected), 0, size - 1);
}

inline float bounded_value(
    device const float* input,
    int y,
    int x,
    int height,
    int width,
    int padding_mode,
    bool align_corners) {
  int bounded_y = bounded_index(y, height, padding_mode, align_corners);
  int bounded_x = bounded_index(x, width, padding_mode, align_corners);
  if (bounded_y < 0 || bounded_x < 0) {
    return 0.0f;
  }
  return input[bounded_y * width + bounded_x];
}

inline void atomic_add_bounded(
    device atomic_float* output,
    int y,
    int x,
    int height,
    int width,
    int padding_mode,
    bool align_corners,
    float value) {
  int bounded_y = bounded_index(y, height, padding_mode, align_corners);
  int bounded_x = bounded_index(x, width, padding_mode, align_corners);
  if (bounded_y >= 0 && bounded_x >= 0) {
    atomic_fetch_add_explicit(
        &output[bounded_y * width + bounded_x],
        value,
        memory_order_relaxed);
  }
}

inline void get_cubic_coefficients(thread float coefficients[4], float t) {
  constexpr float a = -0.75f;
  float x1 = t;
  float x2 = 1.0f - t;
  coefficients[0] = ((a * (x1 + 1.0f) - 5.0f * a) *
                         (x1 + 1.0f) +
                     8.0f * a) *
                        (x1 + 1.0f) -
                    4.0f * a;
  coefficients[1] = ((a + 2.0f) * x1 - (a + 3.0f)) * x1 * x1 + 1.0f;
  coefficients[2] = ((a + 2.0f) * x2 - (a + 3.0f)) * x2 * x2 + 1.0f;
  coefficients[3] = ((a * (x2 + 1.0f) - 5.0f * a) *
                         (x2 + 1.0f) +
                     8.0f * a) *
                        (x2 + 1.0f) -
                    4.0f * a;
}

inline void get_cubic_coefficients_grad(
    thread float coefficients[4],
    float t) {
  constexpr float a = -0.75f;
  float x;
  x = -1.0f - t;
  coefficients[0] = (-3.0f * a * x - 10.0f * a) * x - 8.0f * a;
  x = -t;
  coefficients[1] =
      (-3.0f * (a + 2.0f) * x - 2.0f * (a + 3.0f)) * x;
  x = 1.0f - t;
  coefficients[2] =
      (3.0f * (a + 2.0f) * x - 2.0f * (a + 3.0f)) * x;
  x = 2.0f - t;
  coefficients[3] = (3.0f * a * x - 10.0f * a) * x + 8.0f * a;
}

kernel void grid_sampler_2d_backward_fp32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device atomic_float* grad_input [[buffer(3)]],
    device float* grad_grid [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& input_height [[buffer(7)]],
    constant int& input_width [[buffer(8)]],
    constant int& output_height [[buffer(9)]],
    constant int& output_width [[buffer(10)]],
    constant int& interpolation_mode [[buffer(11)]],
    constant int& padding_mode [[buffer(12)]],
    constant int& align_corners_value [[buffer(13)]],
    constant int& input_requires_grad_value [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  int count = batch_size * output_height * output_width;
  if (int(gid) >= count) {
    return;
  }

  int index = int(gid);
  int output_x = index % output_width;
  int output_y = (index / output_width) % output_height;
  int batch = index / (output_height * output_width);
  bool align_corners = align_corners_value != 0;
  bool input_requires_grad = input_requires_grad_value != 0;

  int grid_offset = index * 2;
  float normalized_x = grid[grid_offset];
  float normalized_y = grid[grid_offset + 1];
  float source_x_gradient;
  float source_y_gradient;

  if (interpolation_mode == INTERPOLATION_BICUBIC) {
    float source_x = unnormalize_only_set_grad(
        normalized_x, input_width, align_corners, source_x_gradient);
    float source_y = unnormalize_only_set_grad(
        normalized_y, input_height, align_corners, source_y_gradient);
    int northwest_x = int(floor(source_x));
    int northwest_y = int(floor(source_y));
    float tx = source_x - float(northwest_x);
    float ty = source_y - float(northwest_y);

    float x_coefficients[4];
    float y_coefficients[4];
    float x_coefficient_gradients[4];
    float y_coefficient_gradients[4];
    get_cubic_coefficients(x_coefficients, tx);
    get_cubic_coefficients(y_coefficients, ty);
    get_cubic_coefficients_grad(x_coefficient_gradients, tx);
    get_cubic_coefficients_grad(y_coefficient_gradients, ty);

    float grid_x_gradient = 0.0f;
    float grid_y_gradient = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
      int output_offset =
          ((batch * channels + channel) * output_height + output_y) *
              output_width +
          output_x;
      int input_channel_offset =
          (batch * channels + channel) * input_height * input_width;
      float output_gradient = grad_output[output_offset];
      device const float* input_channel = input + input_channel_offset;
      device atomic_float* grad_input_channel = input_requires_grad
          ? grad_input + input_channel_offset
          : grad_input;

      for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
          int input_y = northwest_y - 1 + row;
          int input_x = northwest_x - 1 + column;
          if (input_requires_grad) {
            atomic_add_bounded(
                grad_input_channel,
                input_y,
                input_x,
                input_height,
                input_width,
                padding_mode,
                align_corners,
                output_gradient * x_coefficients[column] *
                    y_coefficients[row]);
          }

          float value = bounded_value(
              input_channel,
              input_y,
              input_x,
              input_height,
              input_width,
              padding_mode,
              align_corners);
          grid_x_gradient -= value * x_coefficient_gradients[column] *
              y_coefficients[row] * output_gradient;
          grid_y_gradient -= value * y_coefficient_gradients[row] *
              x_coefficients[column] * output_gradient;
        }
      }
    }
    grad_grid[grid_offset] = source_x_gradient * grid_x_gradient;
    grad_grid[grid_offset + 1] = source_y_gradient * grid_y_gradient;
    return;
  }

  float source_x = source_index_set_grad(
      normalized_x,
      input_width,
      padding_mode,
      align_corners,
      source_x_gradient);
  float source_y = source_index_set_grad(
      normalized_y,
      input_height,
      padding_mode,
      align_corners,
      source_y_gradient);

  if (interpolation_mode == INTERPOLATION_NEAREST) {
    int nearest_x = int(rint(source_x));
    int nearest_y = int(rint(source_y));
    if (input_requires_grad &&
        within_bounds_2d(
            nearest_y, nearest_x, input_height, input_width)) {
      for (int channel = 0; channel < channels; ++channel) {
        int output_offset =
            ((batch * channels + channel) * output_height + output_y) *
                output_width +
            output_x;
        int input_offset =
            ((batch * channels + channel) * input_height + nearest_y) *
                input_width +
            nearest_x;
        atomic_fetch_add_explicit(
            &grad_input[input_offset],
            grad_output[output_offset],
            memory_order_relaxed);
      }
    }
    grad_grid[grid_offset] = 0.0f;
    grad_grid[grid_offset + 1] = 0.0f;
    return;
  }

  int northwest_x = int(floor(source_x));
  int northwest_y = int(floor(source_y));
  int northeast_x = northwest_x + 1;
  int northeast_y = northwest_y;
  int southwest_x = northwest_x;
  int southwest_y = northwest_y + 1;
  int southeast_x = northwest_x + 1;
  int southeast_y = northwest_y + 1;

  float northwest_weight =
      (float(southeast_x) - source_x) *
      (float(southeast_y) - source_y);
  float northeast_weight =
      (source_x - float(southwest_x)) *
      (float(southwest_y) - source_y);
  float southwest_weight =
      (float(northeast_x) - source_x) *
      (source_y - float(northeast_y));
  float southeast_weight =
      (source_x - float(northwest_x)) *
      (source_y - float(northwest_y));

  float grid_x_gradient = 0.0f;
  float grid_y_gradient = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    int output_offset =
        ((batch * channels + channel) * output_height + output_y) *
            output_width +
        output_x;
    int input_channel_offset =
        (batch * channels + channel) * input_height * input_width;
    float output_gradient = grad_output[output_offset];
    device const float* input_channel = input + input_channel_offset;
    device atomic_float* grad_input_channel = input_requires_grad
        ? grad_input + input_channel_offset
        : grad_input;

    if (input_requires_grad) {
      if (within_bounds_2d(
              northwest_y,
              northwest_x,
              input_height,
              input_width)) {
        atomic_fetch_add_explicit(
            &grad_input_channel[northwest_y * input_width + northwest_x],
            northwest_weight * output_gradient,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              northeast_y,
              northeast_x,
              input_height,
              input_width)) {
        atomic_fetch_add_explicit(
            &grad_input_channel[northeast_y * input_width + northeast_x],
            northeast_weight * output_gradient,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              southwest_y,
              southwest_x,
              input_height,
              input_width)) {
        atomic_fetch_add_explicit(
            &grad_input_channel[southwest_y * input_width + southwest_x],
            southwest_weight * output_gradient,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              southeast_y,
              southeast_x,
              input_height,
              input_width)) {
        atomic_fetch_add_explicit(
            &grad_input_channel[southeast_y * input_width + southeast_x],
            southeast_weight * output_gradient,
            memory_order_relaxed);
      }
    }

    if (within_bounds_2d(
            northwest_y, northwest_x, input_height, input_width)) {
      float value =
          input_channel[northwest_y * input_width + northwest_x];
      grid_x_gradient -=
          value * (float(southeast_y) - source_y) * output_gradient;
      grid_y_gradient -=
          value * (float(southeast_x) - source_x) * output_gradient;
    }
    if (within_bounds_2d(
            northeast_y, northeast_x, input_height, input_width)) {
      float value =
          input_channel[northeast_y * input_width + northeast_x];
      grid_x_gradient +=
          value * (float(southwest_y) - source_y) * output_gradient;
      grid_y_gradient -=
          value * (source_x - float(southwest_x)) * output_gradient;
    }
    if (within_bounds_2d(
            southwest_y, southwest_x, input_height, input_width)) {
      float value =
          input_channel[southwest_y * input_width + southwest_x];
      grid_x_gradient -=
          value * (source_y - float(northeast_y)) * output_gradient;
      grid_y_gradient +=
          value * (float(northeast_x) - source_x) * output_gradient;
    }
    if (within_bounds_2d(
            southeast_y, southeast_x, input_height, input_width)) {
      float value =
          input_channel[southeast_y * input_width + southeast_x];
      grid_x_gradient +=
          value * (source_y - float(northwest_y)) * output_gradient;
      grid_y_gradient +=
          value * (source_x - float(northwest_x)) * output_gradient;
    }
  }

  grad_grid[grid_offset] = source_x_gradient * grid_x_gradient;
  grad_grid[grid_offset + 1] = source_y_gradient * grid_y_gradient;
}
)METAL";
}

void initialize_grid_sample_metal() {
  if (g_grid_sample_initialized.load(std::memory_order_acquire)) {
    return;
  }

  std::lock_guard<std::mutex> lock(g_grid_sample_init_mutex);
  if (g_grid_sample_initialized.load(std::memory_order_relaxed)) {
    return;
  }

  @autoreleasepool {
    g_grid_sample_device = MTLCreateSystemDefaultDevice();
    if (!g_grid_sample_device) {
      throw std::runtime_error("Failed to create Metal device for grid_sample backward");
    }

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    g_grid_sample_library = [g_grid_sample_device
        newLibraryWithSource:grid_sample_metal_source()
        options:options
        error:&error];
    if (!g_grid_sample_library) {
      throw std::runtime_error(
          "Failed to compile grid_sample backward Metal library: " +
          std::string([[error localizedDescription] UTF8String]));
    }

    id<MTLFunction> function = [g_grid_sample_library
        newFunctionWithName:@"grid_sampler_2d_backward_fp32"];
    if (!function) {
      throw std::runtime_error("Failed to find grid_sample backward Metal function");
    }
    g_grid_sample_backward_fp32 = [g_grid_sample_device
        newComputePipelineStateWithFunction:function
        error:&error];
    if (!g_grid_sample_backward_fp32) {
      throw std::runtime_error(
          "Failed to create grid_sample backward Metal pipeline: " +
          std::string([[error localizedDescription] UTF8String]));
    }
  }

  g_grid_sample_initialized.store(true, std::memory_order_release);
}

void check_int32_size(int64_t value, const char* name) {
  TORCH_CHECK(
      value <= std::numeric_limits<int32_t>::max(),
      name,
      " exceeds the int32 limit: ",
      value);
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward_mps(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  TORCH_CHECK(input.device().is_mps(), "grid_sample backward input must be on MPS");
  TORCH_CHECK(grid.device().is_mps(), "grid_sample backward grid must be on MPS");
  TORCH_CHECK(
      grad_output.device().is_mps(),
      "grid_sample backward grad_output must be on MPS");
  at::native::check_grid_sampler_common(input, grid);
  at::native::check_grid_sampler_2d(input, grid);
  TORCH_CHECK(
      input.scalar_type() == grid.scalar_type(),
      "grid_sample backward input and grid must have the same dtype");
  TORCH_CHECK(
      grad_output.scalar_type() == input.scalar_type(),
      "grid_sample backward grad_output and input must have the same dtype");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
          input.scalar_type() == at::kBFloat16,
      "grid_sample backward supports float32, float16 and bfloat16 on MPS");
  TORCH_CHECK(
      interpolation_mode >= 0 && interpolation_mode <= 2,
      "grid_sample backward received invalid interpolation mode ",
      interpolation_mode);
  TORCH_CHECK(
      padding_mode >= 0 && padding_mode <= 2,
      "grid_sample backward received invalid padding mode ",
      padding_mode);

  const int64_t batch_size = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  const int64_t output_height = grid.size(1);
  const int64_t output_width = grid.size(2);
  TORCH_CHECK(
      grad_output.dim() == 4 && grad_output.size(0) == batch_size &&
          grad_output.size(1) == channels &&
          grad_output.size(2) == output_height &&
          grad_output.size(3) == output_width,
      "grid_sample backward grad_output has incompatible shape ",
      grad_output.sizes());

  const int64_t thread_count = batch_size * output_height * output_width;
  check_int32_size(thread_count, "grid_sample backward thread count");
  check_int32_size(input.numel(), "grid_sample backward input size");
  check_int32_size(grid.numel(), "grid_sample backward grid size");
  check_int32_size(grad_output.numel(), "grid_sample backward grad_output size");

  const at::ScalarType input_dtype = input.scalar_type();
  const at::ScalarType grid_dtype = grid.scalar_type();
  auto grad_output_work = grad_output.to(at::kFloat).contiguous();
  auto input_work = input.to(at::kFloat).contiguous();
  auto grid_work = grid.to(at::kFloat).contiguous();

  at::Tensor grad_input;
  at::Tensor grad_input_work;
  if (output_mask[0]) {
    grad_input_work = at::zeros_like(input_work);
  } else {
    grad_input_work = at::zeros({1}, input_work.options());
  }
  auto grad_grid_work = at::zeros_like(grid_work);

  if (thread_count > 0) {
    initialize_grid_sample_metal();
    if (output_mask[0]) {
      at::globalContext().alertNotDeterministic(
          "grid_sampler_2d_backward_mps");
    }

    // Resolve Metal storage before requesting the command encoder.  This is
    // required to remain on PyTorch's current stream without a forced sync.
    id<MTLBuffer> grad_output_buffer =
        at::native::mps::getMTLBufferStorage(grad_output_work);
    id<MTLBuffer> input_buffer =
        at::native::mps::getMTLBufferStorage(input_work);
    id<MTLBuffer> grid_buffer =
        at::native::mps::getMTLBufferStorage(grid_work);
    id<MTLBuffer> grad_input_buffer =
        at::native::mps::getMTLBufferStorage(grad_input_work);
    id<MTLBuffer> grad_grid_buffer =
        at::native::mps::getMTLBufferStorage(grad_grid_work);

    @autoreleasepool {
      auto stream = at::mps::getCurrentMPSStream();
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      [encoder setComputePipelineState:g_grid_sample_backward_fp32];
      [encoder
          setBuffer:grad_output_buffer
          offset:grad_output_work.storage_offset() *
              grad_output_work.element_size()
          atIndex:0];
      [encoder
          setBuffer:input_buffer
          offset:input_work.storage_offset() * input_work.element_size()
          atIndex:1];
      [encoder
          setBuffer:grid_buffer
          offset:grid_work.storage_offset() * grid_work.element_size()
          atIndex:2];
      [encoder
          setBuffer:grad_input_buffer
          offset:grad_input_work.storage_offset() *
              grad_input_work.element_size()
          atIndex:3];
      [encoder
          setBuffer:grad_grid_buffer
          offset:grad_grid_work.storage_offset() *
              grad_grid_work.element_size()
          atIndex:4];

      int32_t batch_size_value = static_cast<int32_t>(batch_size);
      int32_t channels_value = static_cast<int32_t>(channels);
      int32_t input_height_value = static_cast<int32_t>(input_height);
      int32_t input_width_value = static_cast<int32_t>(input_width);
      int32_t output_height_value = static_cast<int32_t>(output_height);
      int32_t output_width_value = static_cast<int32_t>(output_width);
      int32_t interpolation_mode_value =
          static_cast<int32_t>(interpolation_mode);
      int32_t padding_mode_value = static_cast<int32_t>(padding_mode);
      int32_t align_corners_value = align_corners ? 1 : 0;
      int32_t input_requires_grad_value = output_mask[0] ? 1 : 0;
      [encoder setBytes:&batch_size_value length:sizeof(int32_t) atIndex:5];
      [encoder setBytes:&channels_value length:sizeof(int32_t) atIndex:6];
      [encoder setBytes:&input_height_value length:sizeof(int32_t) atIndex:7];
      [encoder setBytes:&input_width_value length:sizeof(int32_t) atIndex:8];
      [encoder setBytes:&output_height_value length:sizeof(int32_t) atIndex:9];
      [encoder setBytes:&output_width_value length:sizeof(int32_t) atIndex:10];
      [encoder
          setBytes:&interpolation_mode_value
          length:sizeof(int32_t)
          atIndex:11];
      [encoder
          setBytes:&padding_mode_value
          length:sizeof(int32_t)
          atIndex:12];
      [encoder
          setBytes:&align_corners_value
          length:sizeof(int32_t)
          atIndex:13];
      [encoder
          setBytes:&input_requires_grad_value
          length:sizeof(int32_t)
          atIndex:14];

      const NSUInteger threadgroup_width = std::min<NSUInteger>(
          256, g_grid_sample_backward_fp32.maxTotalThreadsPerThreadgroup);
      MTLSize grid_size = MTLSizeMake(
          static_cast<NSUInteger>(thread_count), 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, 1, 1);
      [encoder
          dispatchThreads:grid_size
          threadsPerThreadgroup:threadgroup_size];
      // PyTorch owns encoder finalization and command-buffer submission.
    }
  }

  if (output_mask[0]) {
    grad_input = input_dtype == at::kFloat
        ? grad_input_work
        : grad_input_work.to(input_dtype);
  }
  at::Tensor grad_grid = grid_dtype == at::kFloat
      ? grad_grid_work
      : grad_grid_work.to(grid_dtype);
  return std::make_tuple(grad_input, grad_grid);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl(
      "grid_sampler_2d_backward",
      TORCH_FN(grid_sampler_2d_backward_mps));
}
