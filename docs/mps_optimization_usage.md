# LADA MPS 优化使用指南

本文档介绍如何在 macOS 上使用 LADA 的 MPS (Metal Performance Shaders) 优化功能。

## 概述

MPS 优化为 macOS 用户提供了显著的性能提升，特别是在使用 Apple Silicon (M1/M2/M3) 芯片的设备上。这些优化包括：

- 自动批处理大小调整
- 内存管理优化
- 异步数据传输
- 队列大小动态调整
- 性能监控和报告

## 系统要求

- macOS 12.3 或更高版本
- Apple Silicon (M1/M2/M3) 或支持 Metal 的 Intel Mac
- PyTorch 1.12 或更高版本（支持 MPS）

## 快速开始

### 1. 检查 MPS 可用性

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 2. 使用 MPS 设备运行 LADA

```bash
# 基本用法
python -m lada.cli.main --device mps --input input_video.mp4

# 使用自定义输出路径
python -m lada.cli.main --device mps --input input_video.mp4 --output output_video.mp4

# 批量处理
python -m lada.cli.main --device mps --input /path/to/videos/ --output /path/to/output/
```

### 3. 运行优化测试

```bash
python test_mps_optimization.py
```

## 配置选项

### 性能配置文件

LADA 提供三种预设的性能配置文件：

#### 实时模式 (realtime)
- 优化延迟，适合实时处理
- 较小的批处理大小和队列
- 较低的内存使用

#### 质量模式 (quality)
- 优化质量，适合离线处理
- 较大的批处理大小和队列
- 较高的内存使用

#### 平衡模式 (balanced) - 默认
- 平衡性能和质量
- 中等的批处理大小和队列

### 自定义配置

您可以通过修改 `lada/config/mps_config.py` 来自定义 MPS 配置：

```python
from lada.config.mps_config import MPSConfig

# 创建自定义配置
custom_config = MPSConfig(
    memory_fraction=0.8,
    detection_batch_size=4,
    restoration_batch_size=2,
    max_frames_per_batch=8,
    enable_async_transfer=True
)
```

## 性能监控

### 实时监控

MPS 优化包含内置的性能监控功能，会在处理完成后自动显示性能报告：

```
MPS PERFORMANCE REPORT
======================================================
Device: mps
Monitoring Duration: 150 samples

Current Metrics:
  FPS: 12.34
  Avg Inference Time: 81.23 ms
  CPU Memory: 45.2%
  MPS Utilization: 78.5%
  MPS Allocated: 1024.0 MB

Aggregate Metrics:
  Average FPS: 11.89
  Max FPS: 15.67
  Avg Inference Time: 84.12 ms
  Min Inference Time: 65.43 ms
  Max Inference Time: 120.89 ms

Queue Metrics:
  frame_restoration_queue:
    Avg Size: 8.5
    Max Size: 16
    Avg Wait Time: 12.34 ms
```

### 手动性能监控

```python
from lada.lib.mps_performance_monitor import start_performance_monitoring, print_performance_report

# 开始监控
start_performance_monitoring(torch.device('mps'))

# 您的代码...

# 打印报告
print_performance_report()
```

## 故障排除

### 常见问题

#### 1. MPS 不可用
```
MPS selected but torch.backends.mps is not available
```

**解决方案：**
- 确保使用 PyTorch 1.12+
- 确保 macOS 版本 12.3+
- 更新到最新的 PyTorch 版本

#### 2. 内存不足错误
```
RuntimeError: MPS backend out of memory
```

**解决方案：**
- 减少批处理大小
- 使用 'realtime' 配置文件
- 关闭其他占用 GPU 内存的应用

#### 3. 性能较差

**解决方案：**
- 检查系统资源使用情况
- 尝试不同的配置文件
- 确保没有其他重负载应用运行

### 调试模式

启用详细日志以获取更多调试信息：

```bash
export LADA_LOG_LEVEL=DEBUG
python -m lada.cli.main --device mps --input input_video.mp4
```

### 环境变量

MPS 优化会自动设置以下环境变量：

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPS_FORCE_ENABLE_GPU=1
```

## 性能优化建议

### 1. 硬件优化
- 确保充足的系统内存 (16GB+)
- 关闭不必要的后台应用
- 使用 SSD 存储以提高 I/O 性能

### 2. 软件优化
- 使用最新版本的 PyTorch
- 定期更新 macOS 系统
- 使用适当的视频编码器设置

### 3. 配置优化
- 根据硬件能力选择合适的配置文件
- 监控内存使用情况
- 调整批处理大小以平衡性能和内存使用

## 基准测试

在不同硬件上的典型性能表现：

| 设备 | 分辨率 | FPS | 内存使用 |
|------|--------|-----|----------|
| M1 MacBook Air | 1080p | 8-12 | 2-4GB |
| M1 Pro MacBook Pro | 1080p | 12-18 | 4-6GB |
| M1 Max MacBook Pro | 1080p | 18-25 | 6-8GB |
| M2 MacBook Air | 1080p | 10-15 | 2-4GB |
| M2 Pro Mac mini | 1080p | 15-22 | 4-6GB |

*注意：实际性能可能因视频内容、系统配置和其他因素而有所不同。*

## 贡献

如果您发现性能问题或有改进建议，请：

1. 运行 `test_mps_optimization.py` 收集诊断信息
2. 在 GitHub 上创建 issue
3. 包含系统信息和性能报告

## 相关文档

- [MPS 优化技术指南](mps_optimization_guide.md)
- [macOS 安装指南](macos_install.md)
- [性能调优指南](pipeline.md)