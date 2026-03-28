# LADA MPS 性能优化指南

## 概述
本文档提供了在 macOS MPS (Metal Performance Shaders) 模式下优化 LADA 性能的详细建议和实现方案。

## 当前性能瓶颈

### 1. MPS 设备特性限制
- **FP16 禁用**: MPS 下强制禁用半精度浮点运算
- **兼容性问题**: 某些 PyTorch 操作需要 CPU 回退
- **内存带宽限制**: 统一内存架构的带宽约束

### 2. 数据传输瓶颈
- **同步数据传输**: CPU-MPS 数据移动阻塞执行
- **频繁内存拷贝**: 每帧都需要设备间数据传输
- **缺乏数据预取**: 没有提前加载下一批数据

### 3. 内存管理问题
- **固定队列大小**: 512MB 限制可能过于保守
- **内存碎片**: 频繁的内存分配/释放
- **缓存未优化**: 没有充分利用 MPS 内存缓存

## 优化方案

### 1. 数据传输优化

#### 1.1 异步数据传输
```python
# 在 MosaicDetectionModel 中实现异步数据传输
class AsyncDataTransfer:
    def __init__(self, device):
        self.device = device
        self.stream = torch.mps.Stream() if device.type == 'mps' else None
    
    def async_to_device(self, tensor):
        if self.stream:
            with torch.mps.stream(self.stream):
                return tensor.to(self.device, non_blocking=True)
        return tensor.to(self.device)
```

#### 1.2 数据预取机制
```python
# 在 FrameRestorer 中实现数据预取
class DataPrefetcher:
    def __init__(self, device, prefetch_size=2):
        self.device = device
        self.prefetch_size = prefetch_size
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        
    def prefetch_frames(self, frame_generator):
        # 预取下一批帧到 MPS 设备
        pass
```

### 2. 内存管理优化

#### 2.1 动态队列大小调整
```python
def calculate_optimal_queue_size(device, video_resolution, available_memory_gb):
    """根据设备性能和视频分辨率动态计算队列大小"""
    if device.type == 'mps':
        # MPS 设备使用统一内存，可以更激进
        base_memory_mb = available_memory_gb * 1024 * 0.6  # 使用60%可用内存
    else:
        base_memory_mb = 512  # 保守默认值
    
    frame_size_mb = (video_resolution[0] * video_resolution[1] * 3) / (1024 * 1024)
    optimal_queue_size = int(base_memory_mb / frame_size_mb)
    return max(1, optimal_queue_size)
```

#### 2.2 内存池管理
```python
class MPSMemoryPool:
    def __init__(self, device, pool_size_mb=1024):
        self.device = device
        self.pool_size = pool_size_mb * 1024 * 1024
        self.allocated_tensors = []
        
    def get_tensor(self, shape, dtype=torch.float32):
        """从内存池获取张量，避免频繁分配"""
        # 实现内存池逻辑
        pass
        
    def return_tensor(self, tensor):
        """归还张量到内存池"""
        pass
```

### 3. 批处理优化

#### 3.1 自适应批处理大小
```python
def get_optimal_batch_size(device, model_type, video_resolution):
    """根据设备和模型类型确定最优批处理大小"""
    if device.type == 'mps':
        # 根据 Apple Silicon 芯片类型调整
        if 'M1' in platform.processor():
            base_batch_size = 6
        elif 'M2' in platform.processor():
            base_batch_size = 8
        elif 'M3' in platform.processor():
            base_batch_size = 12
        else:
            base_batch_size = 8
            
        # 根据分辨率调整
        if video_resolution[0] * video_resolution[1] > 1920 * 1080:
            base_batch_size = max(2, base_batch_size // 2)
            
        return base_batch_size
    return 4  # 默认值
```

### 4. 模型优化

#### 4.1 MPS 特定优化
```python
def optimize_model_for_mps(model, device):
    """为 MPS 设备优化模型"""
    if device.type == 'mps':
        # 启用 MPS 特定优化
        model = torch.jit.script(model)  # JIT 编译
        model = model.to(device)
        
        # 预热模型
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
            
    return model
```

#### 4.2 计算图优化
```python
def enable_mps_optimizations():
    """启用 MPS 特定的 PyTorch 优化"""
    if torch.backends.mps.is_available():
        # 启用 MPS 内存高效模式
        torch.backends.mps.enable_memory_efficient_attention()
        
        # 设置 MPS 特定环境变量
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
```

### 5. 线程模型优化

#### 5.1 并行推理管道
```python
class ParallelInferencePipeline:
    def __init__(self, detection_model, restoration_model, device):
        self.detection_model = detection_model
        self.restoration_model = restoration_model
        self.device = device
        
        # 为不同模型创建独立的 MPS 流
        self.detection_stream = torch.mps.Stream()
        self.restoration_stream = torch.mps.Stream()
        
    def parallel_inference(self, frames):
        """并行执行检测和还原推理"""
        # 实现并行推理逻辑
        pass
```

## 实施优先级

### 高优先级 (立即实施)
1. **自适应批处理大小**: 根据设备性能调整 batch_size
2. **动态队列大小**: 根据可用内存调整队列限制
3. **MPS 模型预热**: 减少首次推理延迟

### 中优先级 (短期实施)
1. **异步数据传输**: 实现非阻塞数据移动
2. **内存池管理**: 减少内存分配开销
3. **计算图优化**: 启用 MPS 特定优化

### 低优先级 (长期实施)
1. **并行推理管道**: 重构线程模型
2. **模型量化**: 探索 MPS 兼容的量化方案
3. **自定义 MPS 内核**: 针对特定操作优化

## 性能监控

### 关键指标
- **推理延迟**: 单帧处理时间
- **内存使用**: 峰值和平均内存占用
- **队列等待时间**: 线程间同步开销
- **设备利用率**: MPS GPU 使用率

### 监控工具
```python
class MPSPerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        
    def start_timing(self, operation):
        self.metrics[operation] = time.time()
        
    def end_timing(self, operation):
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]
            logger.info(f"{operation}: {duration:.3f}s")
```

## 预期性能提升

基于优化方案，预期可以获得以下性能提升：

- **推理速度**: 提升 30-50%
- **内存效率**: 减少 20-30% 内存占用
- **播放流畅度**: 减少 60-80% 卡顿现象
- **导出速度**: 提升 25-40%

## 注意事项

1. **兼容性**: 确保优化不影响其他设备的性能
2. **稳定性**: 逐步实施优化，避免引入新的 bug
3. **可配置性**: 提供配置选项让用户调整优化参数
4. **监控**: 实施性能监控以验证优化效果