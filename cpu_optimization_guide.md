# CPU瓶颈优化指南

## 概述

本指南详细说明了针对LADA视频处理中CPU瓶颈问题的优化方案。通过系统性的分析和优化，我们解决了CPU使用率过高（100%-200%）而GPU使用率偏低（75%）的问题。

## 问题分析

### 原始问题
- **CPU使用率**: 100%-200%（严重瓶颈）
- **GPU使用率**: 75%左右（未充分利用）
- **症状**: 视频播放卡顿，处理速度慢

### 根本原因
1. **图像处理操作CPU密集**: 大量的`cv2.cvtColor`、`cv2.resize`等操作
2. **数据预处理效率低**: 逐个处理图像，缺乏批处理优化
3. **CPU-GPU负载不平衡**: GPU资源未充分利用
4. **线程池配置不当**: 并发处理机制不够高效

## 优化方案

### 1. 图像处理优化

#### 创建GPU加速图像处理器
- **文件**: `lada/lib/gpu_image_utils.py`
- **功能**: 
  - GPU批量图像缩放
  - GPU颜色空间转换
  - GPU归一化处理

#### 优化图像转换函数
- **文件**: `lada/lib/image_utils.py`
- **改进**:
  - 添加`batch_resize`函数支持批量缩放
  - 优化`img2tensor`函数支持GPU加速
  - 使用切片操作替代`cv2.cvtColor`进行BGR到RGB转换

### 2. 数据预处理优化

#### BasicVSR++推理优化
- **文件**: `lada/basicvsrpp/inference.py`
- **改进**:
  - GPU加速的批量预处理
  - 优化的张量转换
  - 异步数据传输优化

#### DeepMosaics推理优化
- **文件**: `lada/deepmosaics/inference.py`
- **改进**:
  - 向量化操作替代循环处理
  - 优化的归一化计算
  - GPU加速的数据预处理

### 3. CPU-GPU负载平衡

#### 负载平衡器
- **文件**: `lada/lib/cpu_gpu_balancer.py`
- **功能**:
  - 实时资源监控
  - 自适应任务调度
  - 动态工作线程调整
  - 任务类型智能分配

#### 核心特性
```python
# 使用示例
from lada.lib.cpu_gpu_balancer import get_global_balancer, Task, TaskType

balancer = get_global_balancer()
task = Task(
    func=image_processing_function,
    args=(image,),
    task_type=TaskType.CPU_INTENSIVE,
    priority=0
)
future = balancer.submit_task(task)
result = future.result()
```

### 4. 并发处理优化

#### 帧恢复器优化
- **文件**: `lada/lib/frame_restorer.py`
- **改进**:
  - 集成负载平衡器
  - 并行剪辑处理
  - 批量图像操作
  - 异步任务执行

## 使用方法

### 1. 启用优化功能

```python
# 在创建FrameRestorer时启用负载平衡
restorer = FrameRestorer(
    model_name="your_model",
    device="mps",  # 或 "cuda"
    enable_load_balancing=True
)
```

### 2. 配置负载平衡器

```python
from lada.lib.cpu_gpu_balancer import get_global_balancer

balancer = get_global_balancer()

# 调整工作线程数量
balancer.adjust_workers(cpu_workers=8, gpu_workers=4)

# 获取资源统计
stats = balancer.get_resource_stats()
print(f"CPU使用率: {stats['cpu_usage']:.1f}%")
print(f"GPU使用率: {stats['gpu_usage']:.1f}%")
```

### 3. 批量图像处理

```python
from lada.lib import image_utils

# 批量缩放
resized_images = image_utils.batch_resize(images, target_size=(512, 512))

# GPU加速张量转换
tensors = image_utils.img2tensor(images, device="mps", batch_process=True)
```

## 性能测试

### 运行测试脚本
```bash
python test_cpu_optimization.py
```

### 测试内容
1. **顺序处理基准测试**
2. **负载平衡处理测试**
3. **GPU加速处理测试**
4. **性能对比分析**

### 预期改进效果
- **CPU使用率降低**: 30-50%
- **处理速度提升**: 2-4倍
- **GPU利用率提升**: 提升至85-95%
- **内存效率改善**: 减少内存碎片

## 监控和调优

### 1. 实时监控

```python
# 获取资源使用情况
balancer = get_global_balancer()
stats = balancer.get_resource_stats()

if stats['is_cpu_bottleneck']:
    print("检测到CPU瓶颈，建议增加GPU工作线程")
    balancer.adjust_workers(gpu_workers=stats['gpu_workers'] + 1)

if stats['is_gpu_underutilized']:
    print("GPU未充分利用，建议调整任务分配策略")
```

### 2. 性能调优建议

#### CPU密集型任务优化
- 增加CPU工作线程数量
- 使用批处理减少函数调用开销
- 优化数据结构和算法

#### GPU加速优化
- 确保批处理大小适中（4-16）
- 避免频繁的CPU-GPU数据传输
- 使用异步传输重叠计算和传输

#### 内存优化
- 及时释放不需要的张量
- 使用内存池减少分配开销
- 监控内存使用避免OOM

## 故障排除

### 常见问题

1. **负载平衡器初始化失败**
   - 检查设备可用性
   - 确认依赖库安装正确
   - 查看错误日志

2. **GPU加速不生效**
   - 验证设备支持（CUDA/MPS）
   - 检查张量设备放置
   - 确认批处理大小合理

3. **性能提升不明显**
   - 调整工作线程数量
   - 检查任务类型分配
   - 优化批处理大小

### 调试方法

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查设备状态
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# 监控资源使用
balancer = get_global_balancer()
print(balancer.get_resource_stats())
```

## 配置建议

### 硬件配置建议
- **CPU**: 8核以上，支持多线程
- **GPU**: 8GB显存以上，支持CUDA 11.0+或MPS
- **内存**: 16GB以上系统内存

### 软件配置建议
- **批处理大小**: 4-16（根据显存调整）
- **CPU工作线程**: CPU核心数 + 4
- **GPU工作线程**: 2-4个
- **队列大小**: 32-64

## 总结

通过实施这些优化措施，我们成功解决了CPU瓶颈问题：

1. **图像处理优化**: 使用GPU加速和批处理减少CPU负载
2. **数据预处理优化**: 向量化操作和异步传输提升效率
3. **负载平衡**: 智能任务调度充分利用GPU资源
4. **并发优化**: 并行处理和线程池优化提升吞吐量

这些优化措施不仅解决了当前的性能问题，还为未来的扩展提供了良好的基础架构。