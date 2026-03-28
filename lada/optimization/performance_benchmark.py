"""
Performance Benchmark Script for LADA Optimization

Features:
- Compare performance metrics before and after optimization
- Generate detailed performance reports
- Produce visualization charts of performance data
- Validate optimization effectiveness
"""

import torch
import time
import psutil
import json
import os
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy and Torch types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)

# Import optimization components
from .integrated_optimization_manager import (
    IntegratedOptimizationManager, 
    OptimizationMode, 
    get_optimization_manager
)


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    test_name: str
    duration_seconds: float
    fps: float
    cpu_utilization_avg: float
    cpu_utilization_max: float
    memory_usage_avg: float
    memory_usage_max: float
    gpu_memory_usage: float
    frame_drop_rate: float
    latency_avg_ms: float
    latency_max_ms: float
    throughput_mbps: float
    optimization_enabled: bool
    additional_metrics: Dict[str, Any]


@dataclass
class ComparisonReport:
    """Comparison report"""
    baseline_result: BenchmarkResult
    optimized_result: BenchmarkResult
    improvements: Dict[str, float]
    performance_gain_percent: float
    efficiency_score: float
    recommendation: str


class PerformanceBenchmark:
    """Performance benchmarking helper"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.optimization_manager = None
        self.test_data = []
        self.monitoring_active = False
        self.performance_data = []
        
        # Test configurations
        self.test_configs = {
            'light_load': {
                'frame_count': 100,
                'frame_size': (720, 480),
                'processing_complexity': 'low',
                'concurrent_tasks': 1
            },
            'medium_load': {
                'frame_count': 200,
                'frame_size': (1280, 720),
                'processing_complexity': 'medium',
                'concurrent_tasks': 2
            },
            'heavy_load': {
                'frame_count': 300,
                'frame_size': (1920, 1080),
                'processing_complexity': 'high',
                'concurrent_tasks': 4
            },
            'stress_test': {
                'frame_count': 100,  # Fewer frames
                'frame_size': (1920, 1080),  # Fixed resolution
                'processing_complexity': 'extreme',
                'concurrent_tasks': 4  # Reduced concurrency
            }
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, ComparisonReport]:
        """Run the comprehensive benchmark suite"""
        print("Starting comprehensive performance benchmark...")
        
        results = {}
        
        for test_name, config in self.test_configs.items():
            print(f"\nRunning {test_name} test...")
            
            # Baseline test (no optimization)
            print(f"  Running baseline test for {test_name}...")
            baseline_result = self._run_single_test(test_name, config, optimization_enabled=False)
            
            # Optimized test
            print(f"  Running optimized test for {test_name}...")
            optimized_result = self._run_single_test(test_name, config, optimization_enabled=True)
            
            # Generate comparison report
            comparison = self._generate_comparison_report(baseline_result, optimized_result)
            results[test_name] = comparison
            
            print(f"  {test_name} completed - Performance gain: {comparison.performance_gain_percent:.1f}%")
        
        # Save results
        self._save_benchmark_results(results)
        
        # Generate visualization report
        self._generate_visualization_report(results)
        
        print(f"\nBenchmark completed. Results saved to: {self.output_dir}")
        return results
    
    def _run_single_test(self, test_name: str, config: Dict[str, Any], optimization_enabled: bool) -> BenchmarkResult:
        """Run a single benchmark test"""
        # Initialize optimization manager
        if optimization_enabled:
            self.optimization_manager = get_optimization_manager()
            self.optimization_manager.initialize_optimization(
                mode=OptimizationMode.PERFORMANCE,
                target_fps=30.0
            )
        else:
            self.optimization_manager = None
        
        # Prepare test data
        test_frames = self._generate_test_frames(config)
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        # Run test loop
        start_time = time.time()
        processed_frames = 0
        frame_times = []
        dropped_frames = 0
        
        try:
            for i, frame in enumerate(test_frames):
                frame_start = time.time()
                
                # Process frame
                if optimization_enabled and self.optimization_manager:
                    # Use optimized processing
                    from .realtime_gui_optimizer import FrameMetadata, PlaybackQuality
                    metadata = FrameMetadata(
                        frame_id=i,
                        timestamp=time.time(),
                        processing_time=0.0,
                        quality_level=PlaybackQuality.HIGH,
                        is_keyframe=(i % 10 == 0),
                        size_bytes=frame.numel() * 4,  # Assume float32
                        resolution=frame.shape[-2:]
                    )
                    
                    result = self.optimization_manager.process_frame_optimized(frame, metadata)
                    if result is not None:
                        processed_frames += 1
                    else:
                        dropped_frames += 1
                else:
                    # Baseline processing
                    result = self._baseline_frame_processing(frame, config['processing_complexity'])
                    processed_frames += 1
                
                frame_end = time.time()
                frame_times.append((frame_end - frame_start) * 1000)  # ms
                
                # Simulate real-time processing interval
                time.sleep(0.001)
        
        except Exception as e:
            print(f"Test error: {e}")
        
        finally:
            # Stop performance monitoring
            self._stop_performance_monitoring()
            
            # Cleanup optimization manager
            if self.optimization_manager:
                self.optimization_manager.shutdown()
                self.optimization_manager = None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Compute performance metrics
        fps = processed_frames / duration if duration > 0 else 0
        frame_drop_rate = (dropped_frames / len(test_frames)) * 100 if test_frames else 0
        
        # Analyze performance data
        perf_summary = self._analyze_performance_data()
        
        return BenchmarkResult(
            test_name=test_name,
            duration_seconds=duration,
            fps=fps,
            cpu_utilization_avg=perf_summary['cpu_avg'],
            cpu_utilization_max=perf_summary['cpu_max'],
            memory_usage_avg=perf_summary['memory_avg'],
            memory_usage_max=perf_summary['memory_max'],
            gpu_memory_usage=perf_summary['gpu_memory'],
            frame_drop_rate=frame_drop_rate,
            latency_avg_ms=np.mean(frame_times) if frame_times else 0,
            latency_max_ms=np.max(frame_times) if frame_times else 0,
            throughput_mbps=self._calculate_throughput(processed_frames, config['frame_size'], duration),
            optimization_enabled=optimization_enabled,
            additional_metrics={
                'processed_frames': processed_frames,
                'dropped_frames': dropped_frames,
                'total_frames': len(test_frames),
                'frame_times_p95': np.percentile(frame_times, 95) if frame_times else 0,
                'frame_times_p99': np.percentile(frame_times, 99) if frame_times else 0
            }
        )
    
    def _generate_test_frames(self, config: Dict[str, Any]) -> List[torch.Tensor]:
        """Generate test frames"""
        frames = []
        height, width = config['frame_size']
        
        # Determine device
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        for i in range(config['frame_count']):
            # Generate random frame data directly on the target device
            frame = torch.randn(3, height, width, dtype=torch.float32, device=device)
            
            # Add different processing demands based on complexity
            if config['processing_complexity'] == 'high':
                # Add noise and complex patterns
                noise = torch.randn_like(frame) * 0.1
                frame = frame + noise
            elif config['processing_complexity'] == 'extreme':
                # More complex data patterns
                for _ in range(2):  # Fewer loops for speed
                    noise = torch.randn_like(frame) * 0.05
                    frame = frame + noise
                    frame = torch.clamp(frame, -1, 1)
            
            frames.append(frame)
        
        return frames
    
    def _baseline_frame_processing(self, frame: torch.Tensor, complexity: str) -> torch.Tensor:
        """Baseline frame processing (no optimization)"""
        device = frame.device
        
        # Simulate processing at different complexity levels
        if complexity == 'low':
            # Simple processing
            result = frame * 0.9 + 0.1
        elif complexity == 'medium':
            # Medium processing
            kernel = torch.ones(3, 3, 3, 3, device=device) / 9
            result = torch.nn.functional.conv2d(
                frame.unsqueeze(0), 
                kernel, 
                padding=1
            ).squeeze(0)
        elif complexity == 'high':
            # Complex processing
            result = frame
            for _ in range(3):
                kernel = torch.randn(3, 3, 3, 3, device=device)
                result = torch.nn.functional.conv2d(
                    result.unsqueeze(0), 
                    kernel, 
                    padding=1
                ).squeeze(0)
                result = torch.relu(result)
        else:  # extreme
            # Extreme complexity processing — reduced loop count
            result = frame
            for _ in range(3):  # Reduced from 5 to 3
                kernel = torch.randn(3, 3, 3, 3, device=device)  # Reduced from 5x5 to 3x3
                result = torch.nn.functional.conv2d(
                    result.unsqueeze(0), 
                    kernel, 
                    padding=1
                ).squeeze(0)
                result = torch.relu(result)
                # Simplify pooling and interpolation operations
                if result.shape[-2:] != frame.shape[-2:]:
                    result = torch.nn.functional.interpolate(
                        result.unsqueeze(0), 
                        size=frame.shape[-2:], 
                        mode='bilinear'
                    ).squeeze(0)
        
        return result
    
    def _start_performance_monitoring(self):
        """Start performance monitoring"""
        self.performance_data = []
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                data = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'gpu_memory': 0
                }
                
                # GPU memory monitoring
                if torch.backends.mps.is_available():
                    try:
                        data['gpu_memory'] = torch.mps.driver_allocated_memory()
                    except:
                        pass
                
                self.performance_data.append(data)
                time.sleep(0.1)  # 100ms interval
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _analyze_performance_data(self) -> Dict[str, float]:
        """Analyze performance data"""
        if not self.performance_data:
            return {
                'cpu_avg': 0, 'cpu_max': 0,
                'memory_avg': 0, 'memory_max': 0,
                'gpu_memory': 0
            }
        
        cpu_values = [d['cpu_percent'] for d in self.performance_data]
        memory_values = [d['memory_percent'] for d in self.performance_data]
        gpu_memory_values = [d['gpu_memory'] for d in self.performance_data]
        
        return {
            'cpu_avg': np.mean(cpu_values),
            'cpu_max': np.max(cpu_values),
            'memory_avg': np.mean(memory_values),
            'memory_max': np.max(memory_values),
            'gpu_memory': np.max(gpu_memory_values) if gpu_memory_values else 0
        }
    
    def _calculate_throughput(self, frames: int, frame_size: Tuple[int, int], duration: float) -> float:
        """Calculate throughput (Mbps)"""
        if duration <= 0:
            return 0
        
        # Assume 3 bytes per pixel (RGB)
        bytes_per_frame = frame_size[0] * frame_size[1] * 3
        total_bytes = frames * bytes_per_frame
        bits_per_second = (total_bytes * 8) / duration
        return bits_per_second / (1024 * 1024)  # Mbps
    
    def _generate_comparison_report(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> ComparisonReport:
        """Generate a comparison report"""
        improvements = {}
        
        # Calculate improvement percentages
        metrics = [
            'fps', 'cpu_utilization_avg', 'memory_usage_avg', 
            'frame_drop_rate', 'latency_avg_ms', 'throughput_mbps'
        ]
        
        for metric in metrics:
            baseline_val = getattr(baseline, metric)
            optimized_val = getattr(optimized, metric)
            
            if baseline_val > 0:
                if metric in ['cpu_utilization_avg', 'memory_usage_avg', 'frame_drop_rate', 'latency_avg_ms']:
                    # These metrics are better when lower
                    improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                else:
                    # These metrics are better when higher
                    improvement = ((optimized_val - baseline_val) / baseline_val) * 100
                
                improvements[metric] = improvement
            else:
                improvements[metric] = 0
        
        # Compute overall performance gain
        performance_gain = np.mean([
            improvements.get('fps', 0),
            improvements.get('throughput_mbps', 0),
            -improvements.get('latency_avg_ms', 0),  # Lower latency is good
            -improvements.get('frame_drop_rate', 0)  # Lower frame drop rate is good
        ])
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(baseline, optimized)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(improvements, performance_gain)
        
        return ComparisonReport(
            baseline_result=baseline,
            optimized_result=optimized,
            improvements=improvements,
            performance_gain_percent=performance_gain,
            efficiency_score=efficiency_score,
            recommendation=recommendation
        )
    
    def _calculate_efficiency_score(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> float:
        """Calculate efficiency score (0–100)"""
        # Performance improvement weights
        fps_improvement = (optimized.fps - baseline.fps) / baseline.fps if baseline.fps > 0 else 0
        latency_improvement = (baseline.latency_avg_ms - optimized.latency_avg_ms) / baseline.latency_avg_ms if baseline.latency_avg_ms > 0 else 0
        
        # Resource usage improvement weights
        cpu_improvement = (baseline.cpu_utilization_avg - optimized.cpu_utilization_avg) / baseline.cpu_utilization_avg if baseline.cpu_utilization_avg > 0 else 0
        memory_improvement = (baseline.memory_usage_avg - optimized.memory_usage_avg) / baseline.memory_usage_avg if baseline.memory_usage_avg > 0 else 0
        
        # Composite score
        score = (
            fps_improvement * 30 +
            latency_improvement * 25 +
            cpu_improvement * 25 +
            memory_improvement * 20
        ) * 100
        
        return max(0, min(100, score))
    
    def _generate_recommendation(self, improvements: Dict[str, float], performance_gain: float) -> str:
        """Generate optimization recommendation"""
        if performance_gain > 20:
            return "Significant optimization benefits; enable all optimizations in production."
        elif performance_gain > 10:
            return "Good optimization results; selectively enable optimizations based on scenarios."
        elif performance_gain > 0:
            return "Moderate optimization effects; adjust parameters or target specific bottlenecks."
        else:
            return "No clear benefits or negative impact; check optimization configuration and environment."
    
    def _save_benchmark_results(self, results: Dict[str, ComparisonReport]):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = {}
        for test_name, report in results.items():
            detailed_results[test_name] = {
                'baseline': asdict(report.baseline_result),
                'optimized': asdict(report.optimized_result),
                'improvements': report.improvements,
                'performance_gain_percent': report.performance_gain_percent,
                'efficiency_score': report.efficiency_score,
                'recommendation': report.recommendation
            }
        
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, cls=NumpyEncoder)
        
        # Save summary report
        summary = self._generate_summary_report(results)
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    
    def _generate_summary_report(self, results: Dict[str, ComparisonReport]) -> str:
        """Generate a summary report"""
        report_lines = [
            "LADA Performance Optimization Benchmark Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Test Results Summary:",
            "-" * 30
        ]
        
        for test_name, comparison in results.items():
            baseline = comparison.baseline_result
            optimized = comparison.optimized_result
            
            report_lines.extend([
                f"\n{test_name.upper()} TEST:",
                f"  Performance Gain: {comparison.performance_gain_percent:.1f}%",
                f"  Efficiency Score: {comparison.efficiency_score:.1f}/100",
                f"  FPS: {baseline.fps:.1f} → {optimized.fps:.1f} ({comparison.improvements.get('fps', 0):.1f}%)",
                f"  CPU Usage: {baseline.cpu_utilization_avg:.1f}% → {optimized.cpu_utilization_avg:.1f}% ({comparison.improvements.get('cpu_utilization_avg', 0):.1f}%)",
                f"  Memory Usage: {baseline.memory_usage_avg:.1f}% → {optimized.memory_usage_avg:.1f}% ({comparison.improvements.get('memory_usage_avg', 0):.1f}%)",
                f"  Frame Drop Rate: {baseline.frame_drop_rate:.1f}% → {optimized.frame_drop_rate:.1f}% ({comparison.improvements.get('frame_drop_rate', 0):.1f}%)",
                f"  Avg Latency: {baseline.latency_avg_ms:.1f}ms → {optimized.latency_avg_ms:.1f}ms ({comparison.improvements.get('latency_avg_ms', 0):.1f}%)",
                f"  Recommendation: {comparison.recommendation}"
            ])
        
        # Overall analysis
        avg_performance_gain = np.mean([r.performance_gain_percent for r in results.values()])
        avg_efficiency_score = np.mean([r.efficiency_score for r in results.values()])
        
        report_lines.extend([
            "\nOverall Analysis:",
            "-" * 20,
            f"Average Performance Gain: {avg_performance_gain:.1f}%",
            f"Average Efficiency Score: {avg_efficiency_score:.1f}/100",
            "",
            "Key Findings:",
            "- " + self._get_key_findings(results)
        ])
        
        return "\n".join(report_lines)
    
    def _get_key_findings(self, results: Dict[str, ComparisonReport]) -> str:
        """Get key findings"""
        findings = []
        
        # Analyze the largest improvement
        best_test = max(results.items(), key=lambda x: x[1].performance_gain_percent)
        findings.append(f"Best optimization effect in {best_test[0]} test, performance improved by {best_test[1].performance_gain_percent:.1f}%")
        
        # Analyze CPU usage improvement
        cpu_improvements = [r.improvements.get('cpu_utilization_avg', 0) for r in results.values()]
        avg_cpu_improvement = np.mean(cpu_improvements)
        if avg_cpu_improvement > 10:
            findings.append(f"Average CPU usage decreased by {avg_cpu_improvement:.1f}%, improving resource efficiency")
        
        # Analyze latency improvement
        latency_improvements = [r.improvements.get('latency_avg_ms', 0) for r in results.values()]
        avg_latency_improvement = np.mean(latency_improvements)
        if avg_latency_improvement > 15:
            findings.append(f"Average latency reduced by {avg_latency_improvement:.1f}%, significantly improving real-time performance")
        
        return "\n- ".join(findings) if findings else "Optimization effects require further analysis"
    
    def _generate_visualization_report(self, results: Dict[str, ComparisonReport]):
        """Generate visualization report"""
        try:
            # Create performance comparison charts
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('LADA Performance Optimization Results', fontsize=16)
            
            test_names = list(results.keys())
            
            # FPS comparison
            baseline_fps = [results[name].baseline_result.fps for name in test_names]
            optimized_fps = [results[name].optimized_result.fps for name in test_names]
            
            x = np.arange(len(test_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, baseline_fps, width, label='Baseline', alpha=0.8)
            axes[0, 0].bar(x + width/2, optimized_fps, width, label='Optimized', alpha=0.8)
            axes[0, 0].set_xlabel('Test Cases')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].set_title('Frame Rate Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(test_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # CPU usage comparison
            baseline_cpu = [results[name].baseline_result.cpu_utilization_avg for name in test_names]
            optimized_cpu = [results[name].optimized_result.cpu_utilization_avg for name in test_names]
            
            axes[0, 1].bar(x - width/2, baseline_cpu, width, label='Baseline', alpha=0.8)
            axes[0, 1].bar(x + width/2, optimized_cpu, width, label='Optimized', alpha=0.8)
            axes[0, 1].set_xlabel('Test Cases')
            axes[0, 1].set_ylabel('CPU Utilization (%)')
            axes[0, 1].set_title('CPU Usage Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(test_names, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Latency comparison
            baseline_latency = [results[name].baseline_result.latency_avg_ms for name in test_names]
            optimized_latency = [results[name].optimized_result.latency_avg_ms for name in test_names]
            
            axes[1, 0].bar(x - width/2, baseline_latency, width, label='Baseline', alpha=0.8)
            axes[1, 0].bar(x + width/2, optimized_latency, width, label='Optimized', alpha=0.8)
            axes[1, 0].set_xlabel('Test Cases')
            axes[1, 0].set_ylabel('Average Latency (ms)')
            axes[1, 0].set_title('Latency Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(test_names, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Performance gain percentage
            performance_gains = [results[name].performance_gain_percent for name in test_names]
            colors = ['green' if gain > 0 else 'red' for gain in performance_gains]
            
            axes[1, 1].bar(test_names, performance_gains, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Test Cases')
            axes[1, 1].set_ylabel('Performance Gain (%)')
            axes[1, 1].set_title('Overall Performance Improvement')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = self.output_dir / f"performance_comparison_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {chart_file}")
            
        except ImportError:
            print("Matplotlib not available, skipping visualization generation")
        except Exception as e:
            print(f"Error generating visualization: {e}")


def run_benchmark_suite():
    """Run the full benchmark suite"""
    print("Starting LADA Performance Benchmark Suite...")
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\nBenchmark Suite Completed!")
    print("=" * 50)
    
    # Print brief results
    for test_name, report in results.items():
        print(f"{test_name}: {report.performance_gain_percent:.1f}% improvement")
    
    avg_gain = np.mean([r.performance_gain_percent for r in results.values()])
    print(f"\nAverage Performance Gain: {avg_gain:.1f}%")
    
    return results


if __name__ == "__main__":
    run_benchmark_suite()