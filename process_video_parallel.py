#!/usr/bin/env python3
"""
process_video_parallel.py - 真の並列処理版 🚀 (LADA v0.11.0 MPS完全対応版)

複数セグメントを同時にLADA処理
- 8GB: 1並列（デフォルト）
- 16GB: 2並列推奨
- 32GB+: 3-4並列推奨

期待速度向上: 並列数 × 倍速

LADA v0.11.0 MPS Complete 統合版対応:
- MPS完全最適化対応（メモリ使用量-50%、処理速度+30-50%）
- メモリリーク対策統合
- 動的キューサイズ対応
- flow_warp MPS対応
- エラー出力をキャプチャして表示

トラブルシューティング:
- まず --parallel-workers 1 でテスト
- DEBUG_LADA_CMD=1 で詳細表示
"""
# LADA v0.11.0では不要になったモジュール
# import lada.auto_optimize

import subprocess
import os
import argparse
from collections import deque
from pathlib import Path
import json
import gc
import time
import math
import queue
import re
import signal
import sys
import atexit
import multiprocessing as mp
from multiprocessing import resource_tracker
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
_spawn_semaphore = threading.Semaphore(1)
import psutil
from dataclasses import dataclass

from lada.utils.mps_utils import (
    configure_mps_runtime,
    get_mps_available_memory_gb,
    get_mps_memory_stats,
)
from lada.utils.video_utils import get_default_preset_name

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

def suppress_resource_tracker_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
    warnings.filterwarnings(
        'ignore',
        message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
    )


suppress_resource_tracker_warnings()


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    device: str
    fp16: bool
    mps_memory_fraction: float | None
    log_mps_memory: bool
    encoding_preset: str | None
    encoder: str | None
    encoder_options: str | None
    optimal_encoder_options: str | None
    mp4_fast_start: bool
    mosaic_restoration_model: str
    max_clip_length: int
    mosaic_detection_model: str
    detect_face_mosaics: bool
    lada_temp_dir: str | None
    overwrite: bool
    restore_max_frames: int | None = None
    mosaic_detection_empty_lookahead: int = 0
    restore_sharpen_strength: float = 0.0
    restore_detail_boost: float = 0.0
    restore_blend_feather: float = 1.0
    restore_texture_mix: float = 0.0
    restore_smooth_strength: float = 0.0
    restore_roi_enhancer: str = "none"
    restore_roi_enhancer_model_path: str | None = None
    restore_roi_enhancer_scale: int = 2
    restore_roi_enhancer_strength: float = 0.0
    restore_roi_enhancer_tile: int = 0
    restore_effect_upscale: int = 1


REPO_ROOT = Path(__file__).resolve().parent
COREAI_RESTORATION_MODELS = {
    'basicvsrpp-v1.2-coreai',
    'basicvsrpp-v1.2-coreai-t36',
    'basicvsrpp-v1.2-coreai-t90',
}
DEFAULT_MAX_CLIP_LENGTH = 180
COREAI_STREAMING_CLIP_LENGTHS = {
    'basicvsrpp-v1.2-coreai': 98,
    'basicvsrpp-v1.2-coreai-t36': 104,
    'basicvsrpp-v1.2-coreai-t90': 178,
}
COREAI_PYTHON = Path(
    os.environ.get('LADA_COREAI_PYTHON', REPO_ROOT / '.venv-coreai' / 'bin' / 'python')
)
COREAI_T36_MODEL_PATH = (
    REPO_ROOT / 'model_weights' / 'basicvsrpp-v1.2-t36-fp16.aimodel'
)
COREAI_DETECTION_MODELS = {'v4-fast-coreai'}
COREAI_ENHANCER_MODELS = {
    'realesr-general-x4v3-coreai',
    'realesrgan-x4-coreai',
}
APP_PROGRESS_PREFIX = "@@LADA_PROGRESS@@"
_PROGRESS_PERCENT_RE = re.compile(
    r"(?:Processing video|ビデオの処理中):\s+(\d+(?:\.\d+)?)%"
)


def parse_progress_line(line: str) -> float | None:
    match = _PROGRESS_PERCENT_RE.search(line)
    return float(match.group(1)) if match else None


class ParallelProgressRenderer:
    def __init__(
        self,
        stream=sys.stdout,
        app_protocol: bool = False,
        min_interval: float = 0.25,
    ):
        self.stream = stream
        self.app_protocol = app_protocol
        self.min_interval = min_interval
        self.active_lanes: dict[str, dict] = {}
        self._last_emit: dict[str, float] = {}
        self._rendered_line_count = 0
        self._interactive = bool(
            not app_protocol
            and hasattr(stream, "isatty")
            and stream.isatty()
        )

    def progress(self, event: dict) -> None:
        lane = str(event["lane"])
        self.active_lanes[lane] = dict(event)
        now = time.monotonic()
        last_emit = self._last_emit.get(lane)
        if last_emit is not None and now - last_emit < self.min_interval:
            return
        self._last_emit[lane] = now
        if self.app_protocol:
            self._emit_protocol(event)
        elif self._interactive:
            self._redraw()
        else:
            self.stream.write(self._display_line(event) + "\n")
            self.stream.flush()

    def complete(self, lane: str, message: str) -> None:
        self.active_lanes.pop(lane, None)
        self._last_emit.pop(lane, None)
        event = {"kind": "complete", "lane": lane, "text": message}
        if self.app_protocol:
            self._emit_protocol(event)
        elif self._interactive:
            self._clear_rendered_lines()
            self.stream.write(message + "\n")
            self._rendered_line_count = 0
            self._redraw()
        else:
            self.stream.write(message + "\n")
            self.stream.flush()

    def log(self, message: str) -> None:
        if self._interactive:
            self._clear_rendered_lines()
            self.stream.write(message + "\n")
            self._rendered_line_count = 0
            self._redraw()
        else:
            self.stream.write(message + "\n")
            self.stream.flush()

    def close(self) -> None:
        if self._interactive and self._rendered_line_count:
            self.stream.write("\n")
            self.stream.flush()
        self._rendered_line_count = 0

    def _emit_protocol(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.stream.write(APP_PROGRESS_PREFIX + payload + "\n")
        self.stream.flush()

    def _display_line(self, event: dict) -> str:
        segment = event.get("segment", "?")
        return f"[segment {segment}] {event.get('text', '')}"

    def _clear_rendered_lines(self) -> None:
        if self._rendered_line_count:
            self.stream.write(f"\x1b[{self._rendered_line_count}A")
            for _ in range(self._rendered_line_count):
                self.stream.write("\r\x1b[2K\n")
            self.stream.write(f"\x1b[{self._rendered_line_count}A")

    def _redraw(self) -> None:
        self._clear_rendered_lines()
        for event in self.active_lanes.values():
            self.stream.write("\r\x1b[2K" + self._display_line(event) + "\n")
        self._rendered_line_count = len(self.active_lanes)
        self.stream.flush()


def monitor_progress_events(progress_queue, renderer: ParallelProgressRenderer) -> None:
    while True:
        event = progress_queue.get()
        kind = event.get('kind')
        try:
            if kind == 'stop':
                return
            if kind == 'progress':
                renderer.progress(event)
            elif kind == 'complete':
                renderer.complete(str(event['lane']), str(event.get('text', '')))
            elif kind == 'log':
                renderer.log(str(event.get('text', '')))
        except Exception:
            text = str(event.get('text', ''))
            if text:
                print(text, flush=True)
COREAI_V4_FAST_MODEL_PATH = (
    REPO_ROOT / 'model_weights' / 'lada_mosaic_detection_model_v4_fast-fp16.aimodel'
)


def get_default_mosaic_restoration_model() -> str:
    return 'basicvsrpp-v1.2'


def get_default_mosaic_detection_model() -> str:
    if COREAI_PYTHON.is_file() and COREAI_V4_FAST_MODEL_PATH.is_dir():
        return 'v4-fast-coreai'
    return 'v4-fast'


def get_effective_max_clip_length(
    mosaic_restoration_model: str,
    requested_length: int | None,
) -> int:
    if requested_length is not None:
        return requested_length
    return COREAI_STREAMING_CLIP_LENGTHS.get(
        mosaic_restoration_model,
        DEFAULT_MAX_CLIP_LENGTH,
    )


def get_memory_safe_parallel_workers(
    mosaic_restoration_model: str,
    requested_workers: int,
) -> int:
    """Keep fixed-T90 Core AI restoration within unified-memory limits."""
    model = mosaic_restoration_model.lower()
    is_t90_coreai = model == 'basicvsrpp-v1.2-coreai-t90' or (
        't90' in Path(model).name
        and Path(model).suffix in {'.aimodel', '.aimodelc'}
    )
    return 1 if is_t90_coreai else requested_workers


def lada_cli_command_prefix(
    mosaic_restoration_model: str | None = None,
    mosaic_detection_model: str | None = None,
    roi_enhancer_model: str | None = None,
) -> list[str]:
    """
    Launch lada-cli with this interpreter and the code tree next to this
    script. A `lada-cli` on PATH resolves through the editable install and
    would run a different checkout when this wrapper lives in a worktree.
    """
    interpreter = (
        COREAI_PYTHON
        if mosaic_restoration_model in COREAI_RESTORATION_MODELS
        or str(mosaic_restoration_model).endswith(('.aimodel', '.aimodelc'))
        or mosaic_detection_model in COREAI_DETECTION_MODELS
        or str(mosaic_detection_model).endswith(('.aimodel', '.aimodelc'))
        or roi_enhancer_model in COREAI_ENHANCER_MODELS
        or str(roi_enhancer_model).endswith(('.aimodel', '.aimodelc'))
        else Path(sys.executable)
    )
    return [str(interpreter), '-m', 'lada.cli.main']


def build_worker_env(config: WorkerRuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    if config.device == 'mps':
        env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        env['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        env['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
        if config.mps_memory_fraction is not None:
            env['LADA_MPS_MEMORY_FRACTION'] = str(config.mps_memory_fraction)
            if config.log_mps_memory:
                env['LADA_LOG_MPS_MEMORY'] = '1'

    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '2'
    env['PYTHONMALLOC'] = 'malloc'
    env['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    env['PYTHONWARNINGS'] = ",".join(filter(None, [
        env.get('PYTHONWARNINGS'),
        "ignore::UserWarning:multiprocessing.resource_tracker",
    ]))
    return env


def build_lada_cli_command(config: WorkerRuntimeConfig, input_video: Path, output_video: Path) -> list[str]:
    cmd = [
        *lada_cli_command_prefix(
            config.mosaic_restoration_model,
            config.mosaic_detection_model,
            config.restore_roi_enhancer_model_path,
        ),
        '--input', str(input_video),
        '--output', str(output_video),
        '--device', config.device,
    ]

    cmd.append('--fp16' if config.fp16 else '--no-fp16')

    if config.device == 'mps' and config.mps_memory_fraction is not None:
        cmd.extend(['--mps-memory-fraction', str(config.mps_memory_fraction)])

    if config.encoding_preset:
        cmd.extend(['--encoding-preset', config.encoding_preset])
    elif config.encoder:
        cmd.extend(['--encoder', config.encoder])
        if config.optimal_encoder_options:
            cmd.extend(['--encoder-options', str(config.optimal_encoder_options)])
        elif config.encoder_options:
            cmd.extend(['--encoder-options', str(config.encoder_options)])
    elif config.optimal_encoder_options and config.device == 'mps':
        cmd.extend(['--encoder', 'hevc_videotoolbox'])
        cmd.extend(['--encoder-options', str(config.optimal_encoder_options)])
    elif config.encoder_options:
        cmd.extend(['--encoder', 'hevc_videotoolbox' if config.device == 'mps' else 'libx264'])
        cmd.extend(['--encoder-options', str(config.encoder_options)])

    if config.mp4_fast_start:
        cmd.append('--mp4-fast-start')

    cmd.extend(['--mosaic-restoration-model', config.mosaic_restoration_model])
    cmd.extend(['--max-clip-length', str(config.max_clip_length)])
    if config.restore_max_frames is not None:
        cmd.extend(['--restore-max-frames', str(config.restore_max_frames)])
    if config.restore_sharpen_strength > 0:
        cmd.extend(['--restore-sharpen-strength', str(config.restore_sharpen_strength)])
    if config.restore_detail_boost > 0:
        cmd.extend(['--restore-detail-boost', str(config.restore_detail_boost)])
    cmd.extend(['--restore-blend-feather', str(config.restore_blend_feather)])
    if config.restore_texture_mix > 0:
        cmd.extend(['--restore-texture-mix', str(config.restore_texture_mix)])
    if config.restore_smooth_strength > 0:
        cmd.extend(['--restore-smooth-strength', str(config.restore_smooth_strength)])
    if config.restore_effect_upscale > 1:
        cmd.extend(['--restore-effect-upscale', str(config.restore_effect_upscale)])
    if config.restore_roi_enhancer != "none":
        cmd.extend(['--restore-roi-enhancer', config.restore_roi_enhancer])
        if config.restore_roi_enhancer_model_path:
            cmd.extend(['--restore-roi-enhancer-model-path', str(config.restore_roi_enhancer_model_path)])
        cmd.extend(['--restore-roi-enhancer-scale', str(config.restore_roi_enhancer_scale)])
        cmd.extend(['--restore-roi-enhancer-strength', str(config.restore_roi_enhancer_strength)])
        cmd.extend(['--restore-roi-enhancer-tile', str(config.restore_roi_enhancer_tile)])
    cmd.extend(['--mosaic-detection-model', config.mosaic_detection_model])
    if config.mosaic_detection_empty_lookahead > 0:
        cmd.extend(['--mosaic-detection-empty-lookahead', str(config.mosaic_detection_empty_lookahead)])
    cmd.append('--detect-face-mosaics' if config.detect_face_mosaics else '--no-detect-face-mosaics')

    if config.lada_temp_dir:
        cmd.extend(['--temporary-directory', str(config.lada_temp_dir)])

    return cmd


def get_lada_encoding_preset(args, optimal_encoder_options: str | None) -> str | None:
    """
    Mirror lada-cli's Apple VideoToolbox default when this wrapper has no
    custom encoding settings to pass through.
    """
    if getattr(args, 'encoding_preset', None):
        return args.encoding_preset
    if getattr(args, 'encoder', None) or getattr(args, 'encoder_options', None) or optimal_encoder_options:
        return None
    if getattr(args, 'device', None) == 'mps':
        preset = get_default_preset_name()
        if 'apple' in preset or 'videotoolbox' in preset:
            return preset
    return None


def build_lada_cli_list_command(args) -> list[str] | None:
    list_flags = [
        "list_devices",
        "list_encoding_presets",
        "list_encoders",
        "list_mosaic_restoration_models",
        "list_mosaic_detection_models",
    ]
    for attr in list_flags:
        if getattr(args, attr, False):
            return [*lada_cli_command_prefix(), f"--{attr.replace('_', '-')}"]
    if getattr(args, "list_encoder_options", None):
        return [*lada_cli_command_prefix(), "--list-encoder-options", str(args.list_encoder_options)]
    return None


def run_lada_cli_list_command(args) -> int | None:
    cmd = build_lada_cli_list_command(args)
    if cmd is None:
        return None
    return subprocess.run(cmd).returncode


def aggressive_memory_cleanup_for_device(device: str):
    try:
        import torch

        if device == 'mps' and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif 'cuda' in device and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()
    time.sleep(0.05)


def process_segment_worker(segment_info, config: WorkerRuntimeConfig, progress_queue=None):
    idx, input_path, output_path = segment_info
    input_path = Path(input_path)
    output_path = Path(output_path)
    lane = f"worker-{os.getpid()}"

    with _spawn_semaphore:
        time.sleep(2.0)

    if output_path.exists() and not config.overwrite:
        return {
            'idx': idx,
            'lane': lane,
            'output_path': str(output_path),
            'status': 'skipped',
            'elapsed': 0.0,
            'error': None,
        }

    start_time = time.time()
    worker_name = f"PID:{os.getpid()}"
    start_message = f"[並列処理] セグメント #{idx} 開始 (ワーカー: {worker_name})"
    if progress_queue is None:
        print(start_message, flush=True)
    else:
        progress_queue.put({'kind': 'log', 'text': start_message})

    cmd = build_lada_cli_command(config, input_path, output_path)
    env = build_worker_env(config)
    recent_lines = deque(maxlen=15)

    try:
        if os.environ.get('DEBUG_LADA_CMD'):
            print(f"[DEBUG] 実行コマンド: {' '.join(cmd)}", flush=True)

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if 'Processing video:' in line:
                percent = parse_progress_line(line)
                event = {
                    'kind': 'progress',
                    'lane': lane,
                    'segment': idx,
                    'text': line,
                    'percent': percent,
                }
                if progress_queue is None:
                    print(f"\r  {line}", end='', flush=True)
                else:
                    try:
                        progress_queue.put(event)
                    except Exception:
                        print(f"\r  {line}", end='', flush=True)
            else:
                recent_lines.append(line)
                if 'error' in line.lower() or 'warning' in line.lower():
                    if progress_queue is None:
                        print(f"\n  {line}", flush=True)
                    else:
                        progress_queue.put({'kind': 'log', 'text': f"  {line}"})

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        elapsed = time.time() - start_time
        if progress_queue is None:
            print()
            print(f"[並列処理] セグメント #{idx} 完了 ({elapsed:.1f}秒)", flush=True)
        return {
            'idx': idx,
            'lane': lane,
            'output_path': str(output_path),
            'status': 'success',
            'elapsed': elapsed,
            'error': None,
        }
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] lada-cli実行エラー:", flush=True)
        print(f"  コマンド: {' '.join(cmd)}", flush=True)
        print(f"  終了コード: {e.returncode}", flush=True)
        if recent_lines:
            print(f"  --- worker出力(末尾) ---", flush=True)
            for output_line in recent_lines:
                print(f"  | {output_line}", flush=True)
        return {
            'idx': idx,
            'lane': lane,
            'output_path': None,
            'status': 'error',
            'elapsed': time.time() - start_time,
            'error': f"Command '{' '.join(cmd)}' returned non-zero exit status {e.returncode}.",
        }
    except Exception as e:
        print(f"\n[ERROR] 予期しないエラー: {e}", flush=True)
        return {
            'idx': idx,
            'lane': lane,
            'output_path': None,
            'status': 'error',
            'elapsed': time.time() - start_time,
            'error': str(e),
        }
    finally:
        aggressive_memory_cleanup_for_device(config.device)


def create_parallel_executor(args):
    if args.executor == "thread":
        return ThreadPoolExecutor(max_workers=args.parallel_workers)

    mp_context = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=args.parallel_workers,
        mp_context=mp_context,
        max_tasks_per_child=1,
    )

# ===== グローバルクリーンアップ (MPS v0.11.0最適化版) =====
def cleanup_resources():
    """プロセス終了時のリソースクリーンアップ（MPS v0.11.0最適化版）"""
    try:
        import torch
        
        # CUDA クリーンアップ
        if torch.cuda.is_available():
            # 全デバイスをクリア
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            
            # IPC収集
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
        
        # MPS クリーンアップ（v0.11.0最適化版）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS最適化: empty_cache + 明示的なメモリ解放
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
            # v0.11.0で追加されたMPS最適化機能を使用
            try:
                from lada.utils.mps_utils import optimize_mps_memory
                optimize_mps_memory()
            except ImportError:
                # mps_utilsが利用できない場合は従来の方法で継続
                pass
        
        # PyTorchのガベージコレクション
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            torch._C._cuda_clearCublasWorkspaces()
            
    except Exception:
        pass
    
    # Pythonのガベージコレクション
    gc.collect()
    

def get_effective_mps_memory_fraction(args) -> float | None:
    """Resolve process-level MPS memory fraction."""
    if getattr(args, 'mps_memory_fraction', None) is not None:
        return args.mps_memory_fraction
    if getattr(args, 'device', None) != 'mps':
        return None
    if getattr(args, 'parallel_workers', 1) <= 1:
        return 0.82
    return max(0.35, min(0.70, 0.92 / max(args.parallel_workers, 1)))


def format_mps_memory_stats() -> str | None:
    """Format torch.mps memory stats for logs."""
    stats = get_mps_memory_stats()
    if not stats.get("available"):
        return None
    def _gb(value):
        if value is None:
            return "n/a"
        return f"{value / (1024 ** 3):.1f}GB"
    pressure = stats.get("pressure_ratio")
    pressure_str = f"{pressure:.2f}" if pressure is not None else "n/a"
    return (
        f"MPS memory: current={_gb(stats.get('current_allocated_bytes'))} "
        f"driver={_gb(stats.get('driver_allocated_bytes'))} "
        f"recommended={_gb(stats.get('recommended_max_bytes'))} "
        f"system_available={_gb(stats.get('system_available_bytes'))} "
        f"pressure={pressure_str}"
    )

# ===== シグナルハンドリング（Ctrl+C即座停止対応） =====
# グローバル停止フラグ
_shutdown_requested = False
_force_shutdown = False
_active_executor = None
_active_executor_lock = threading.Lock()


def _set_active_executor(executor):
    global _active_executor
    with _active_executor_lock:
        _active_executor = executor


def _shutdown_active_executor():
    with _active_executor_lock:
        executor = _active_executor
    if executor is None:
        return
    _shutdown_executor(executor, wait=False, cancel_futures=True)


def _shutdown_executor(executor, *, wait: bool, cancel_futures: bool = False):
    if executor is None:
        return
    try:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    except RuntimeError:
        pass
    except Exception:
        pass


def _terminate_descendant_processes():
    try:
        current = psutil.Process(os.getpid())
        for child in current.children(recursive=True):
            try:
                child.terminate()
            except Exception:
                pass
    except Exception:
        pass

def signal_handler(signum, frame):
    """Ctrl+CまたはGUI停止時のハンドリング"""
    global _shutdown_requested, _force_shutdown
    
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"

    if _shutdown_requested or _force_shutdown:
        print(f"\n\n⚠️  強制終了します...")
    else:
        print(f"\n\n🛑 {signal_name}を受信しました。即座に停止します...")

    _shutdown_requested = True
    _force_shutdown = True
    _shutdown_active_executor()
    _terminate_descendant_processes()
    os._exit(130)

atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== ユーティリティ関数 =====
def get_video_duration(video_path):
    """動画の長さを取得"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    try:
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        duration = float(result)
        return duration
    except Exception:
        return None

def calculate_segment_count(duration, segment_duration):
    """セグメント数を計算"""
    return math.ceil(duration / segment_duration)


def resolve_segment_duration(duration, segment_duration, segment_count=None):
    if segment_count is None:
        return segment_duration
    return duration / segment_count


def get_video_resolution(video_path):
    """動画の解像度を取得"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        if data.get('streams'):
            stream = data['streams'][0]
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            return width, height
        return None, None
    except Exception:
        return None, None

def get_video_bitrate(video_path):
    """動画のビットレートを取得（kbps）"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=bit_rate',
        '-of', 'json',
        str(video_path)
    ]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        if data.get('streams'):
            stream = data['streams'][0]
            bit_rate = stream.get('bit_rate')
            if bit_rate:
                return int(bit_rate) // 1000  # bps → kbps
        return None
    except Exception:
        return None

def get_video_fps(video_path):
    """動画のフレームレートを取得"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        str(video_path)
    ]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        if data.get('streams'):
            stream = data['streams'][0]
            fps_str = stream.get('r_frame_rate')
            if fps_str and '/' in fps_str:
                # "30000/1001" のような形式を処理
                num, den = fps_str.split('/')
                return round(int(num) / int(den))
            elif fps_str:
                return round(float(fps_str))
        return None
    except Exception:
        return None

def has_audio_stream(video_path):
    """動画に音声ストリームがあるか確認"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'json',
        str(video_path)
    ]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        return bool(data.get('streams'))
    except Exception:
        return False

def get_optimal_encoder_options(video_path, user_options=None, auto_optimize=True, fps=None, bitrate_multiplier=3.0, qmin=None, qmax=None, quality=None, pre_fps_conversion=False):
    """
    元のファイルと同程度のファイルサイズになるようにエンコーダーオプションを設定
    
    元のビットレートを取得し、同程度のビットレートを使用
    fps: 出力フレームレート（Noneの場合は元のfpsを維持）
    user_options: ユーザー指定オプション（デフォルトに追加/上書き）
    bitrate_multiplier: ビットレート倍率（デフォルト: 3.0）
    qmin: 最小量子化値（0-51、PyAV 16.0+）
    qmax: 最大量子化値（0-51、PyAV 16.0+）
    quality: 品質値 -q:v（0-100、VideoToolbox専用）
    pre_fps_conversion: セグメント分割時にfps変換済みの場合True
    """
    # 自動最適化が無効の場合
    if not auto_optimize:
        return user_options if user_options else ""
    
    # 解像度を取得（表示用）
    width, height = get_video_resolution(video_path)
    
    # ビットレートを取得
    original_bitrate = get_video_bitrate(video_path)
    
    if original_bitrate is None:
        print("⚠️  ビットレートを検出できません。解像度ベースの設定を使用します")
        
        # 解像度ベースのフォールバック
        if height and height <= 1080:
            bitrate_mbps = 8
            resolution_name = "FHD"
        elif height and height <= 1440:
            bitrate_mbps = 16
            resolution_name = "2K"
        elif height and height <= 2160:
            bitrate_mbps = 25
            resolution_name = "4K"
        else:
            bitrate_mbps = 50
            resolution_name = "8K"
    else:
        # 元のビットレートをMbpsに変換
        bitrate_mbps = original_bitrate / 1000
        
        # 指定された倍率で増やす（デフォルト: 3.0倍）
        bitrate_mbps = bitrate_mbps * bitrate_multiplier
        
        # 解像度名を決定
        if height and height <= 1080:
            resolution_name = "FHD"
        elif height and height <= 1440:
            resolution_name = "2K"
        elif height and height <= 2160:
            resolution_name = "4K"
        else:
            resolution_name = "8K"
    
    # レベルを解像度に応じて決定
    if height and height <= 1080:
        level = "4.1"
    elif height and height <= 1440:
        level = "5.0"
    elif height and height <= 2160:
        level = "5.1"
    else:
        level = "5.2"
    
    # エンコードモードの決定
    # 優先順位: quality > qmin/qmax > bitrate
    use_quality_mode = quality is not None
    use_qminmax_mode = (qmin is not None or qmax is not None) and quality is None
    
    if use_quality_mode:
        # -q:vモード（VideoToolbox専用、FFmpeg 4.4+/Apple Silicon）
        # ビットレート制御は使用しない
        options_parts = [
            '-q:v', str(quality),             # 品質値（0-100）
            '-b:v', '0',                      # ビットレート制御を無効化
            '-qmin', '-1',                    # qmin無効化（FFmpeg 7.0未満対策）
            '-qmax', '-1',                    # qmax無効化（FFmpeg 7.0未満対策）
            '-pix_fmt yuv420p',               # ピクセルフォーマット
            '-profile:v high',                # プロファイル: High Profile (高品質)
            '-coder cabac',                   # エントロピー符号化: CABAC (高効率)
            '-a53cc 1',                       # A53クローズドキャプション有効
            '-power_efficient 1',             # 電力効率優先
            '-realtime 0',                    # リアルタイム無効（品質優先）
            '-frames_before 0',               # 連結最適化: 前フレームなし
            '-frames_after 0',                # 連結最適化: 後フレームなし
            '-prio_speed 0',                  # 速度より品質優先
        ]
        
        # 解像度に応じてレベルを設定
        if height and height <= 720:
            options_parts.append('-level 4.0')
        elif height and height <= 1080:
            options_parts.append('-level 4.1')
        else:
            options_parts.append('-level 5.1')
        
        encoding_mode = f"品質ベース (-q:v {quality})"
        
    elif use_qminmax_mode:
        # qmin/qmaxモード
        # ビットレート制御は使用しない
        options_parts = [
            '-pix_fmt yuv420p',               # ピクセルフォーマット
            '-profile:v high',                # プロファイル: High Profile (高品質)
            '-coder cabac',                   # エントロピー符号化: CABAC (高効率)
            '-a53cc 1',                       # A53クローズドキャプション有効
            '-power_efficient 1',             # 電力効率優先
            '-realtime 0',                    # リアルタイム無効（品質優先）
            '-frames_before 0',               # 連結最適化: 前フレームなし
            '-frames_after 0',                # 連結最適化: 後フレームなし
            '-prio_speed 0',                  # 速度より品質優先
        ]
        
        # 解像度に応じてレベルを設定
        if height and height <= 720:
            options_parts.append('-level 4.0')
        elif height and height <= 1080:
            options_parts.append('-level 4.1')
        else:
            options_parts.append('-level 5.1')
        
        # qmin/qmaxを追加
        if qmin is not None:
            options_parts.append(f'-qmin {qmin}')
        if qmax is not None:
            options_parts.append(f'-qmax {qmax}')
        
        encoding_mode = "品質ベース (qmin/qmax)"
    else:
        # ビットレートベースのエンコード（従来の動作）
        # VideoToolboxを使用（macOSのハードウェアエンコーダ）
        # ビットレート制御（-b:v, -maxrate, -bufsize）
        options_parts = [
            f'-b:v {bitrate_mbps}M',          # ビットレート: 元 × {bitrate_multiplier}
            f'-maxrate {bitrate_mbps}M',      # 最大ビットレート
            f'-bufsize {bitrate_mbps * 2}M',  # バッファサイズ: ビットレートの2倍
            '-pix_fmt yuv420p',               # ピクセルフォーマット
            '-profile:v high',                # プロファイル: High Profile (高品質)
            '-coder cabac',                   # エントロピー符号化: CABAC (高効率)
            '-a53cc 1',                       # A53クローズドキャプション有効
            '-power_efficient 1',             # 電力効率優先
            '-realtime 0',                    # リアルタイム無効（品質優先）
            '-frames_before 0',               # 連結最適化: 前フレームなし
            '-frames_after 0',                # 連結最適化: 後フレームなし
            '-prio_speed 0',                  # 速度より品質優先
        ]
        
        # 解像度に応じてレベルを設定
        if height and height <= 720:
            options_parts.append('-level 4.0')
        elif height and height <= 1080:
            options_parts.append('-level 4.1')
        else:
            options_parts.append('-level 5.1')
        
        encoding_mode = "ビットレートベース"
    
    options = ' '.join(options_parts)
    
    # fpsが指定されている場合は追加
    # ただし、pre_fps_conversion=Trueの場合は既にセグメント分割時に変換済み
    if fps and not pre_fps_conversion:
        options += f' -r {fps}'
    
    # user_optionsが指定されている場合は、デフォルトに追加/上書き
    if user_options:
        # user_optionsをパース
        user_opts_dict = {}
        if user_options:
            parts = user_options.strip().split()
            i = 0
            while i < len(parts):
                if parts[i].startswith('-'):
                    key = parts[i]
                    # 次の要素が値かどうか確認
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        user_opts_dict[key] = parts[i + 1]
                        i += 2
                    else:
                        user_opts_dict[key] = ''
                        i += 1
                else:
                    i += 1
        
        # デフォルトoptionsをパース
        default_opts_dict = {}
        parts = options.strip().split()
        i = 0
        while i < len(parts):
            if parts[i].startswith('-'):
                key = parts[i]
                if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                    default_opts_dict[key] = parts[i + 1]
                    i += 2
                else:
                    default_opts_dict[key] = ''
                    i += 1
            else:
                i += 1
        
        # user_optionsでデフォルトを上書き
        default_opts_dict.update(user_opts_dict)
        
        # 再構築
        options_list = []
        for key, value in default_opts_dict.items():
            if value:
                options_list.append(f"{key} {value}")
            else:
                options_list.append(key)
        options = ' '.join(options_list)
        
        print(f"📝 ユーザーオプション適用: {user_options}")
    
    # 情報表示
    if width and height:
        print(f"🎬 解像度: {width}x{height} ({resolution_name})")
    
    print(f"⚡ エンコーダー: VideoToolbox (H.264 ハードウェアエンコード)")
    print(f"📝 エンコードモード: {encoding_mode}")
    
    if quality is not None:
        # -q:vモード
        print(f"🎚️  品質値: -q:v {quality} (0=最低品質, 100=最高品質)")
        print(f"💡 注意: FFmpeg 4.4+とApple Siliconが必須です")
        print(f"💡 注意: 品質優先モードではファイルサイズが可変になります")
    elif use_qminmax_mode:
        # 品質ベースモード
        qmin_str = str(qmin) if qmin is not None else "デフォルト"
        qmax_str = str(qmax) if qmax is not None else "デフォルト"
        print(f"🎚️  品質範囲: qmin={qmin_str}, qmax={qmax_str}")
        print(f"💡 注意: 品質優先モードではファイルサイズが可変になります")
    else:
        # ビットレートベースモード
        if original_bitrate:
            print(f"📊 元のビットレート: {original_bitrate}kbps ({original_bitrate/1000:.1f}Mbps)")
            print(f"📊 設定ビットレート: {bitrate_mbps:.1f}Mbps (元の{bitrate_multiplier:.1f}倍)")
        print(f"📝 ビットレート制御: -b:v {bitrate_mbps:.1f}M -maxrate {bitrate_mbps:.1f}M -bufsize {bitrate_mbps * 2:.1f}M")
    
    if fps:
        print(f"🎞️  フレームレート: {fps}fps")
    
    print(f"📝 プロファイル: High Profile, CABAC, 品質優先モード")
    print(f"📝 最終設定: {options}")
    
    return options

def get_pre_fps_encoder_options(encoder_options: str | None) -> list[str]:
    """Keep rate control while making the intermediate fps pass quality-first."""
    import shlex

    if encoder_options:
        try:
            tokens = shlex.split(encoder_options)
        except ValueError:
            tokens = encoder_options.split()
    else:
        tokens = []

    replaced_options = {
        '-power_efficient',
        '-realtime',
        '-prio_speed',
        '-spatial_aq',
        '-pix_fmt',
        '-frames_before',
        '-frames_after',
    }
    filtered = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in replaced_options:
            index += 2
            continue
        filtered.append(token)
        index += 1

    filtered.extend([
        '-power_efficient', '0',
        '-realtime', '0',
        '-prio_speed', '0',
        '-spatial_aq', '1',
    ])
    return filtered


def convert_fps_segments_parallel(
    input_segments,
    output_dir,
    fps,
    encoder_options=None,
    max_workers=2,
):
    """Convert already-split segments with at most two concurrent FFmpeg jobs."""
    input_segments = [Path(segment) for segment in input_segments]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_segments:
        return []

    encoder_opts = get_pre_fps_encoder_options(encoder_options)
    output_segments = [
        output_dir / f"segment_{index:03d}.mp4"
        for index in range(len(input_segments))
    ]

    def convert_one(input_segment, output_segment):
        partial_output = output_segment.with_name(
            f"{output_segment.stem}.partial{output_segment.suffix}"
        )
        partial_output.unlink(missing_ok=True)
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'videotoolbox',
            '-hwaccel_output_format', 'videotoolbox_vld',
            '-i', str(input_segment),
            '-c:v', 'h264_videotoolbox',
            '-r', str(fps),
            *encoder_opts,
            '-c:a', 'copy',
            '-map', '0:v:0',
            '-map', '0:a?',
            str(partial_output),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            os.replace(partial_output, output_segment)
        except Exception:
            partial_output.unlink(missing_ok=True)
            raise
        return output_segment

    worker_count = min(max_workers, len(input_segments))
    print(f"fps変換開始: {len(input_segments)}個 × FFmpeg {worker_count}並列")
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(convert_one, input_segment, output_segment): output_segment
            for input_segment, output_segment in zip(input_segments, output_segments)
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"\rfps変換: {completed}/{len(input_segments)}", end='', flush=True)
    print()
    print(f"✓ {len(output_segments)}個のfps変換完了")
    return output_segments


def split_video(input_video, output_dir, segment_duration=60, force_split=False, pre_fps=None, encoder_options=None, segment_count=None):
    """
    動画を指定時間ごとに分割
    
    既存のセグメントがある場合はスキップ
    force_split=True で強制的に再分割
    pre_fps: セグメント分割時にfps変換を行う（Noneの場合はコピーモード）
    encoder_options: pre_fps使用時のエンコーダーオプション
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pattern = output_dir / "segment_%03d.mp4"
    
    duration = get_video_duration(input_video)
    effective_segment_duration = resolve_segment_duration(duration, segment_duration, segment_count) if duration else segment_duration

    # 既存セグメントをチェック
    existing_segments = sorted(output_dir.glob("segment_*.mp4"))
    
    if existing_segments and not force_split:
        # 既存セグメントがある場合
        # 期待されるセグメント数を計算
        if duration:
            expected_count = segment_count if segment_count is not None else calculate_segment_count(duration, effective_segment_duration)
            
            if len(existing_segments) == expected_count:
                # セグメント数が一致 → スキップ
                print(f"✓ 既存のセグメントを使用: {len(existing_segments)}個")
                return existing_segments
            else:
                # セグメント数が不一致 → 再分割
                print(f"⚠️  セグメント数が不一致 (既存: {len(existing_segments)}, 期待: {expected_count})")
                print(f"既存セグメントを削除して再分割します...")
                for seg in existing_segments:
                    seg.unlink()
        else:
            # 長さ取得失敗 → 既存セグメントを使用
            print(f"✓ 既存のセグメントを使用: {len(existing_segments)}個")
            return existing_segments
    
    # fps変換時は、まずコピーで分割してから各セグメントを並列変換する。
    if pre_fps:
        if segment_count is not None:
            print(f"動画を{segment_count}個にコピー分割後、{pre_fps}fpsへ2並列変換します")
        else:
            print(f"動画を{segment_duration}秒ごとにコピー分割後、{pre_fps}fpsへ2並列変換します")

        source_dir = output_dir.parent / f"{output_dir.name}_pre_fps_source"
        source_segments = split_video(
            input_video,
            source_dir,
            segment_duration=segment_duration,
            force_split=force_split,
            segment_count=segment_count,
        )
        return convert_fps_segments_parallel(
            source_segments,
            output_dir,
            pre_fps,
            encoder_options=encoder_options,
            max_workers=2,
        )

    # コピーモード（従来の動作）
    if segment_count is not None:
        print(f"動画を{segment_count}個に均等分割中（目標: {effective_segment_duration:.3f}秒/segment）...")
    else:
        print(f"動画を{segment_duration}秒ごとに分割中...")
    cmd = [
        'ffmpeg', '-i', str(input_video),
        '-map', '0:v',              # ビデオストリーム
        '-map', '0:a?',             # 音声ストリーム（あれば）
        '-c:v', 'copy',             # ビデオをコピー
        '-c:a', 'copy',             # 音声をコピー
        '-segment_time', str(effective_segment_duration),
        '-f', 'segment',
        '-reset_timestamps', '1',
        str(output_pattern)
    ]
    
    print(f"分割コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # 結果を確認
    if result.stderr:
        if 'Audio:' in result.stderr:
            print("✓ 音声ストリームを検出しました")
        else:
            print("⚠️  音声ストリームが見つかりませんでした")
    
    segments = sorted(output_dir.glob("segment_*.mp4"))
    print(f"✓ {len(segments)}個のセグメントに分割完了")
    
    return segments


def is_valid_processed_segment(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 100 * 1024


def has_pending_segment_work(temp_dir: Path) -> bool:
    """
    Return True when reusable split segments exist but matching processed files
    are missing or incomplete.
    """
    temp_dir = Path(temp_dir)
    segments = sorted((temp_dir / "segments").glob("segment_*.mp4"))
    if not segments:
        return False

    processed_dir = temp_dir / "processed"
    for i, _segment in enumerate(segments):
        if not is_valid_processed_segment(processed_dir / f"processed_{i:03d}.mp4"):
            return True
    return False


def resolve_single_output_path(input_path: Path, output_path: Path) -> Path:
    """
    For single-file runs, allow --output to be a directory-like path.
    This mirrors batch naming and prevents ffmpeg from receiving an
    extensionless directory path as the final muxer output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{input_path.stem}-UC{input_path.suffix}"
    return output_path


def merge_videos(segment_paths, output_path, encoder='copy'):
    """複数の動画を結合（音声も含む）"""
    print(f"\n動画を結合中... ({len(segment_paths)}個のセグメント)")
    
    concat_file = output_path.parent / "concat_list.txt"
    
    with open(concat_file, 'w', encoding='utf-8') as f:
        for segment in segment_paths:
            f.write(f"file '{segment.resolve().as_posix()}'\n")
    
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(concat_file)]
    
    if encoder == 'copy':
        # 全ストリームをコピー（ビデオ、音声、字幕など）
        cmd.extend([
            '-map', '0',           # 全ストリームを含める
            '-c:v', 'copy',        # ビデオをコピー
            '-c:a', 'copy',        # 音声をコピー
            '-c:s', 'copy',        # 字幕をコピー（あれば）
        ])
    else:
        cmd.extend([
            '-map', '0',           # 全ストリームを含める
            '-c:v', encoder,       # ビデオを再エンコード
            '-c:a', 'copy',        # 音声はコピー
            '-c:s', 'copy',        # 字幕はコピー（あれば）
        ])
    
    cmd.extend(['-y', str(output_path)])
    
    print(f"結合コマンド: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # 結果を確認
    if result.stderr:
        # FFmpegの出力から音声情報を確認
        if 'Audio:' in result.stderr:
            print("✓ 音声ストリームを検出しました")
        else:
            print("⚠️  音声ストリームが見つかりませんでした")
    
    concat_file.unlink()
    
    print(f"✓ 結合完了: {output_path}")

# ===== 並列処理クラス =====
class ParallelVideoProcessor:
    """真の並列処理プロセッサ"""
    
    def __init__(self, args):
        self.args = args
        
        # lada-cliコマンドの存在確認
        self._check_lada_cli()
        
        # 統計情報
        self.stats = {
            'total_segments': 0,
            'processed': 0,
            'skipped': 0,
            'errors': []
        }
        
        # ロック
        self.stats_lock = threading.Lock()
        self.print_lock = threading.Lock()
        
        # 開始時刻
        self.start_time = None
        
        # 解像度最適化されたエンコーダーオプション
        self.optimal_encoder_options = None
        
        # GUIモード検出
        self.gui_mode = os.environ.get('LADA_GUI_MODE') == '1'
    
    def _check_lada_cli(self):
        """lada-cliが利用可能か確認"""
        cli_main = REPO_ROOT / 'lada' / 'cli' / 'main.py'
        if not cli_main.is_file():
            print("\n" + "=" * 70)
            print("⚠️  エラー: ladaパッケージが見つかりません")
            print("=" * 70)
            print(f"\n{cli_main} が存在しません。")
            print("このスクリプトはリポジトリルートに置いたまま実行してください。\n")
            print("=" * 70)
            raise FileNotFoundError(f"lada package not found at {REPO_ROOT}")
        prefix = lada_cli_command_prefix(
            self.args.mosaic_restoration_model,
            self.args.mosaic_detection_model,
        )
        print(f"✓ lada-cli検出: {' '.join(prefix)} ({REPO_ROOT})")
    
    def safe_print(self, msg):
        """スレッドセーフなprint"""
        with self.print_lock:
            print(msg)
    
    def update_stats(self, key, value=1):
        """統計情報を更新"""
        with self.stats_lock:
            if isinstance(value, list):
                self.stats[key].extend(value)
            else:
                self.stats[key] += value

    def _build_worker_runtime_config(self) -> WorkerRuntimeConfig:
        return WorkerRuntimeConfig(
            device=self.args.device,
            fp16=self.args.fp16,
            mps_memory_fraction=get_effective_mps_memory_fraction(self.args),
            log_mps_memory=self.args.log_mps_memory,
            encoding_preset=get_lada_encoding_preset(self.args, self.optimal_encoder_options),
            encoder=self.args.encoder,
            encoder_options=self.args.encoder_options,
            optimal_encoder_options=self.optimal_encoder_options,
            mp4_fast_start=self.args.mp4_fast_start,
            mosaic_restoration_model=self.args.mosaic_restoration_model,
            max_clip_length=self.args.max_clip_length,
            restore_max_frames=self.args.restore_max_frames,
            mosaic_detection_model=self.args.mosaic_detection_model,
            detect_face_mosaics=self.args.detect_face_mosaics,
            lada_temp_dir=str(self.args.lada_temp_dir) if getattr(self.args, 'lada_temp_dir', None) else None,
            overwrite=self.args.overwrite,
            mosaic_detection_empty_lookahead=self.args.mosaic_detection_empty_lookahead,
            restore_sharpen_strength=self.args.restore_sharpen_strength,
            restore_detail_boost=self.args.restore_detail_boost,
            restore_blend_feather=self.args.restore_blend_feather,
            restore_texture_mix=self.args.restore_texture_mix,
            restore_smooth_strength=self.args.restore_smooth_strength,
            restore_roi_enhancer=self.args.restore_roi_enhancer,
            restore_roi_enhancer_model_path=self.args.restore_roi_enhancer_model_path,
            restore_roi_enhancer_scale=self.args.restore_roi_enhancer_scale,
            restore_roi_enhancer_strength=self.args.restore_roi_enhancer_strength,
            restore_roi_enhancer_tile=self.args.restore_roi_enhancer_tile,
            restore_effect_upscale=self.args.restore_effect_upscale,
        )
    
    def process_segment(self, segment_info):
        """
        1つのセグメントを処理
        segment_info: (index, input_path, output_path)
        """
        idx, input_path, output_path = segment_info
        
        # 既存チェック
        if output_path.exists() and not self.args.overwrite:
            self.update_stats('skipped')
            return (idx, output_path, 'skipped')
        
        try:
            start_time = time.time()
            self.safe_print(f"[並列処理] セグメント #{idx} 開始 (ワーカー: {threading.current_thread().name})")
            
            # LADA処理
            self._run_lada_cli(input_path, output_path)
            
            elapsed = time.time() - start_time
            self.safe_print(f"[並列処理] セグメント #{idx} 完了 ({elapsed:.1f}秒)")
            
            self.update_stats('processed')
            
            # 並列処理では積極的にメモリクリーンアップ
            self._aggressive_memory_cleanup()
            
            return (idx, output_path, 'success')
            
        except Exception as e:
            self.safe_print(f"[並列処理] セグメント #{idx} エラー: {e}")
            self.update_stats('errors', [f"Segment {idx}: {e}"])
            # エラー時もメモリクリーンアップ
            self._aggressive_memory_cleanup()
            return (idx, None, 'error')
    
    def _aggressive_memory_cleanup(self):
        """
        積極的なメモリクリーンアップ
        並列処理では各セグメント処理後に必ず実行
        """
        try:
            import torch
            
            if self.args.device == 'mps' and torch.backends.mps.is_available():
                # MPS: 1回クリアで十分（CPU負荷軽減）
                torch.mps.empty_cache()
                
            elif 'cuda' in self.args.device and torch.cuda.is_available():
                # CUDA: 1回クリアで十分（CPU負荷軽減）
                torch.cuda.empty_cache()
            
        except Exception as e:
            # エラーは無視（ベストエフォート）
            pass
        
        # Python GCを1回実行（CPU負荷軽減）
        gc.collect()
        
        # OSにメモリ返却の時間を与える（短縮）
        time.sleep(0.05)

    def _memory_pressure_high(self) -> bool:
        """MPSでは torch.mps の pressure を優先してメモリ圧迫を判定"""
        try:
            if self.args.device == 'mps':
                mps_stats = get_mps_memory_stats()
                pressure_ratio = mps_stats.get("pressure_ratio")
                available_gb = get_mps_available_memory_gb()
                threshold_gb = self.args.cleanup_trigger_gb
                if pressure_ratio is not None and pressure_ratio >= 0.82:
                    return True
                return available_gb < threshold_gb
            vm = psutil.virtual_memory()
            available_gb = vm.available / (1024 ** 3)
            return available_gb < self.args.cleanup_trigger_gb
        except Exception:
            return True
    
    def _run_lada_cli(self, input_video: Path, output_video: Path):
        """
        lada-cliを実行
        
        注意: lada-cliは自動的に音声をコピーします
        """
        # 環境変数設定
        env = os.environ.copy()
        if self.args.device == 'mps':
            env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            env['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
            env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            env['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
        
        env['OMP_NUM_THREADS'] = '2'
        env['MKL_NUM_THREADS'] = '2'
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['NUMEXPR_NUM_THREADS'] = '2'
        env['PYTHONMALLOC'] = 'malloc'
        env['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        effective_mps_fraction = get_effective_mps_memory_fraction(self.args)
        if self.args.device == 'mps' and effective_mps_fraction is not None:
            env['LADA_MPS_MEMORY_FRACTION'] = str(effective_mps_fraction)
            if self.args.log_mps_memory:
                env['LADA_LOG_MPS_MEMORY'] = '1'
        
        # コマンド構築
        cmd = [
            *lada_cli_command_prefix(
                self.args.mosaic_restoration_model,
                self.args.mosaic_detection_model,
                self.args.restore_roi_enhancer_model_path,
            ),
            '--input', str(input_video),
            '--output', str(output_video),
            '--device', self.args.device,
        ]
        
        if self.args.fp16:
            cmd.append('--fp16')
        else:
            cmd.append('--no-fp16')
        if self.args.device == 'mps':
            effective_mps_fraction = get_effective_mps_memory_fraction(self.args)
            if effective_mps_fraction is not None:
                cmd.extend(['--mps-memory-fraction', str(effective_mps_fraction)])
        
        # エンコーディング設定（解像度最適化）
        lada_encoding_preset = get_lada_encoding_preset(self.args, self.optimal_encoder_options)
        if lada_encoding_preset:
            # プリセット指定時はそのまま使用
            cmd.extend(['--encoding-preset', lada_encoding_preset])
        elif self.args.encoder:
            # エンコーダー指定時
            cmd.extend(['--encoder', self.args.encoder])
            # 最適化されたオプションを使用（ユーザー指定がある場合はそちらを優先）
            if self.optimal_encoder_options:
                # encoder-optionsは文字列として1つの引数で渡す
                cmd.extend(['--encoder-options', str(self.optimal_encoder_options)])
        elif self.optimal_encoder_options and self.args.device == 'mps':
            # MPS環境で最適化オプションがある場合はVideoToolboxを明示
            cmd.extend(['--encoder', 'hevc_videotoolbox'])
            # encoder-optionsは文字列として1つの引数で渡す
            cmd.extend(['--encoder-options', str(self.optimal_encoder_options)])
        elif self.args.encoder_options:
            # ユーザーがencoder-optionsのみ指定した場合はデバイスに応じて既定エンコーダーを補完
            if not self.args.encoder:
                if self.args.device == 'mps':
                    cmd.extend(['--encoder', 'hevc_videotoolbox'])
                else:
                    cmd.extend(['--encoder', 'libx264'])
            cmd.extend(['--encoder-options', str(self.args.encoder_options)])
        else:
            # 明示設定がなければ、LADA本体側の既定プリセット選択に委ねる
            pass
        
        if self.args.mp4_fast_start:
            cmd.append('--mp4-fast-start')
        
        cmd.extend(['--mosaic-restoration-model', self.args.mosaic_restoration_model])
        cmd.extend(['--max-clip-length', str(self.args.max_clip_length)])
        if self.args.restore_max_frames is not None:
            cmd.extend(['--restore-max-frames', str(self.args.restore_max_frames)])
        if self.args.restore_sharpen_strength > 0:
            cmd.extend(['--restore-sharpen-strength', str(self.args.restore_sharpen_strength)])
        if self.args.restore_detail_boost > 0:
            cmd.extend(['--restore-detail-boost', str(self.args.restore_detail_boost)])
        cmd.extend(['--restore-blend-feather', str(self.args.restore_blend_feather)])
        if self.args.restore_texture_mix > 0:
            cmd.extend(['--restore-texture-mix', str(self.args.restore_texture_mix)])
        if self.args.restore_smooth_strength > 0:
            cmd.extend(['--restore-smooth-strength', str(self.args.restore_smooth_strength)])
        if self.args.restore_effect_upscale > 1:
            cmd.extend(['--restore-effect-upscale', str(self.args.restore_effect_upscale)])
        if self.args.restore_roi_enhancer != "none":
            cmd.extend(['--restore-roi-enhancer', self.args.restore_roi_enhancer])
            if self.args.restore_roi_enhancer_model_path:
                cmd.extend(['--restore-roi-enhancer-model-path', str(self.args.restore_roi_enhancer_model_path)])
            cmd.extend(['--restore-roi-enhancer-scale', str(self.args.restore_roi_enhancer_scale)])
            cmd.extend(['--restore-roi-enhancer-strength', str(self.args.restore_roi_enhancer_strength)])
            cmd.extend(['--restore-roi-enhancer-tile', str(self.args.restore_roi_enhancer_tile)])
        cmd.extend(['--mosaic-detection-model', self.args.mosaic_detection_model])
        if self.args.mosaic_detection_empty_lookahead > 0:
            cmd.extend(['--mosaic-detection-empty-lookahead', str(self.args.mosaic_detection_empty_lookahead)])
        
        # 顔モザイク検出の設定
        if self.args.detect_face_mosaics:
            cmd.append('--detect-face-mosaics')
        else:
            cmd.append('--no-detect-face-mosaics')
        
        if hasattr(self.args, 'lada_temp_dir') and self.args.lada_temp_dir:
            cmd.extend(['--temporary-directory', str(self.args.lada_temp_dir)])
        
        # デバッグ: 実際のコマンドを表示
        if os.environ.get('DEBUG_LADA_CMD'):
            self.safe_print(f"[DEBUG] 実行コマンド: {' '.join(cmd)}")
        
        # 実行（進捗バーを表示）
        recent_lines = deque(maxlen=15)
        try:
            # サブプロセスをパイプで実行
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # 出力を逐次処理
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # "Processing video:" の行のみ表示（進捗バー）
                if 'Processing video:' in line:
                    # スレッドセーフに1行で表示（上書き）
                    print(f"\r  {line}", end='', flush=True)
                else:
                    recent_lines.append(line)
                    # エラーメッセージは表示
                    if 'error' in line.lower() or 'warning' in line.lower():
                        print(f"\n  {line}")

            # プロセスの終了を待つ
            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            # 進捗バー表示後に改行
            print()

        except subprocess.CalledProcessError as e:
            # エラー時に詳細情報を表示
            self.safe_print(f"\n[ERROR] lada-cli実行エラー:")
            self.safe_print(f"  コマンド: {' '.join(cmd)}")
            self.safe_print(f"  終了コード: {e.returncode}")
            if recent_lines:
                self.safe_print(f"  --- worker出力(末尾) ---")
                for output_line in recent_lines:
                    self.safe_print(f"  | {output_line}")
            raise
        except Exception as e:
            self.safe_print(f"\n[ERROR] 予期しないエラー: {e}")
            raise
    
    def process(self, input_video: Path, output_path: Path, temp_dir: Path):
        """
        並列処理を実行
        """
        print("\n" + "=" * 70)
        print(f"真の並列処理モード 🚀 ({self.args.parallel_workers}並列)")
        print("=" * 70)
        print(f"入力: {input_video}")
        print(f"出力: {output_path}")
        print(f"デバイス: {self.args.device}")
        print(f"並列数: {self.args.parallel_workers}")
        segment_count = getattr(self.args, 'segment_count', None)
        if segment_count is not None:
            print(f"セグメント数: {segment_count}個")
        else:
            print(f"セグメント長: {self.args.segment_duration}秒")
        print("=" * 70 + "\n")
        
        # 一時ディレクトリ作成
        temp_dir.mkdir(parents=True, exist_ok=True)
        segments_dir = temp_dir / "segments"
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpegの一時ディレクトリを設定（グローバル）
        ffmpeg_temp = Path(self.args.ffmpeg_temp_dir) if self.args.ffmpeg_temp_dir else (temp_dir / 'ffmpeg_temp')
        ffmpeg_temp.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(ffmpeg_temp)  # macOS/Linux
        os.environ['TEMP'] = str(ffmpeg_temp)    # Windows互換
        os.environ['TMP'] = str(ffmpeg_temp)     # Windows互換
        print(f"📁 FFmpeg一時ディレクトリ: {ffmpeg_temp}\n")
        if self.args.device == 'mps':
            effective_mps_fraction = get_effective_mps_memory_fraction(self.args)
            if effective_mps_fraction is not None:
                print(f"🧠 MPS メモリ上限: {effective_mps_fraction:.2f} / process")
            if self.args.log_mps_memory:
                stats_line = format_mps_memory_stats()
                if stats_line:
                    print(f"📊 {stats_line}\n")
        
        # fps変換の検証
        use_pre_fps_conversion = getattr(self.args, 'pre_fps_conversion', False)
        
        if self.args.fps and use_pre_fps_conversion:
            original_fps = get_video_fps(input_video)
            if original_fps:
                # fps比較（同じ値かチェック）
                if original_fps == self.args.fps:
                    print(f"⚠️  警告: 元動画のfps ({original_fps}fps) と指定されたfps ({self.args.fps}fps) が同じです")
                    print(f"⚠️  --pre-fps-conversion オプションを無視します（fps変換不要）\n")
                    use_pre_fps_conversion = False
                else:
                    print(f"📊 fps変換: {original_fps}fps → {self.args.fps}fps")
                    print(f"⚡ セグメント分割時にfps変換を実行（処理前モード）\n")
            else:
                print(f"⚠️  警告: 元動画のfpsを検出できませんでした")
                print(f"⚠️  --pre-fps-conversion オプションを無視します\n")
                use_pre_fps_conversion = False
        
        # 解像度検出と最適化設定
        print("\n" + "=" * 70)
        print("解像度検出と最適化設定")
        print("=" * 70)
        if self.args.device == 'mps':
            self.optimal_encoder_options = get_optimal_encoder_options(
                input_video,
                self.args.encoder_options,
                self.args.auto_optimize,
                self.args.fps,
                self.args.bitrate_multiplier,
                self.args.qmin,
                self.args.qmax,
                getattr(self.args, 'quality', None),
                use_pre_fps_conversion  # ローカル変数を使用
            )
        else:
            self.optimal_encoder_options = None
            print("ℹ️  非MPS環境のためVideoToolbox最適化オプションは適用しません（LADA既定設定を使用）")
        print("=" * 70 + "\n")
        
        # 開始時刻
        self.start_time = time.time()
        total_start_time = self.start_time
        
        # 各ステップの時間計測用
        step_times = {}
        
        try:
            # ステップ1: セグメント分割
            step_start = time.time()
            force_split = getattr(self.args, 'force_split', False)
            
            # pre_fps変換が有効な場合（ローカル変数を使用）
            if use_pre_fps_conversion and self.args.fps:
                # セグメント分割時にfps変換を実行
                segments = split_video(
                    input_video, 
                    segments_dir, 
                    self.args.segment_duration, 
                    force_split,
                    pre_fps=self.args.fps,
                    encoder_options=self.optimal_encoder_options,
                    segment_count=segment_count,
                )
            else:
                # 従来の動作（コピーモード）
                segments = split_video(
                    input_video, 
                    segments_dir, 
                    self.args.segment_duration, 
                    force_split,
                    segment_count=segment_count,
                )
            
            self.stats['total_segments'] = len(segments)
            step_times['分割'] = time.time() - step_start
            
            # ステップ2: 処理タスクを準備（差分処理対応）
            step_start = time.time()
            print("\n既存の処理済みセグメントをチェック中...")
            tasks = []
            existing_processed = {}
            
            for i, segment in enumerate(segments):
                output_path_seg = processed_dir / f"processed_{i:03d}.mp4"
                
                # 既存の処理済みセグメントをチェック
                if output_path_seg.exists() and not self.args.overwrite:
                    # ファイルサイズをチェック（100KB以上なら有効）
                    if is_valid_processed_segment(output_path_seg):
                        existing_processed[i] = output_path_seg
                        self.update_stats('skipped')
                    else:
                        # サイズが小さい場合は再処理
                        tasks.append((i, segment, output_path_seg))
                else:
                    tasks.append((i, segment, output_path_seg))
            
            # 差分処理の情報を表示
            if existing_processed:
                print(f"✓ 既存の処理済みセグメント: {len(existing_processed)}個")
                print(f"⚡ 新規処理が必要なセグメント: {len(tasks)}個")
            else:
                print(f"⚡ 全セグメントを処理: {len(tasks)}個")
            
            print(f"\n{'=' * 70}")
            if tasks:
                print(f"並列処理開始: {len(tasks)}個のセグメント × {self.args.parallel_workers}並列")
            else:
                print(f"すべてのセグメントが処理済みです")
            print(f"{'=' * 70}\n")
            
            # ステップ3: 並列処理実行
            results = {}
            
            # 既存の処理済みセグメントを結果に追加
            for idx, path in existing_processed.items():
                results[idx] = path
            
            # 未処理のセグメントのみを並列処理
            if tasks:
                worker_config = self._build_worker_runtime_config()
                progress_manager = None
                if self.args.executor == "thread":
                    progress_queue = queue.Queue()
                else:
                    progress_manager = mp.get_context("spawn").Manager()
                    progress_queue = progress_manager.Queue()
                progress_renderer = ParallelProgressRenderer(
                    app_protocol=os.environ.get("LADA_APP_PROGRESS") == "1"
                )
                progress_monitor = threading.Thread(
                    target=monitor_progress_events,
                    args=(progress_queue, progress_renderer),
                    name="lada-progress-renderer",
                    daemon=True,
                )
                progress_monitor.start()
                executor = create_parallel_executor(self.args)
                _set_active_executor(executor)
                try:
                    # 全タスクを投入
                    future_to_task = {
                        executor.submit(
                            process_segment_worker,
                            task,
                            worker_config,
                            progress_queue,
                        ): task
                        for task in tasks
                    }
                    
                    # 進捗モニタリング
                    completed = 0
                    for future in as_completed(future_to_task):
                        # 停止フラグチェック
                        if _shutdown_requested:
                            print("\n\n⚠️  停止要求を検出しました。処理を中断します...")
                            raise KeyboardInterrupt("停止要求")
                        
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            idx = result['idx']
                            lane = result['lane']
                            output_path_seg = Path(result['output_path']) if result['output_path'] else None
                            status = result['status']
                            if output_path_seg is not None:
                                results[idx] = output_path_seg
                            completed += 1
                            if status == 'success':
                                self.update_stats('processed')
                            elif status == 'skipped':
                                self.update_stats('skipped')
                            else:
                                self.update_stats('errors', [f"Segment {idx}: {result['error']}"])
                            if completed % self.args.memory_cleanup_interval == 0 and self._memory_pressure_high():
                                self._aggressive_memory_cleanup()
                            
                            # 進捗表示
                            with self.stats_lock:
                                progress = (self.stats['processed'] + self.stats['skipped']) / self.stats['total_segments'] * 100
                                elapsed = time.time() - self.start_time
                                avg_time = elapsed / completed if completed > 0 else 0
                                remaining = len(tasks) - completed
                                eta = avg_time * remaining / self.args.parallel_workers
                                
                                eta_str = f"{eta/60:.1f}分" if eta >= 60 else f"{eta:.0f}秒"
                                
                                progress_msg = (f"🚀 進捗: {progress:5.1f}% [{self.stats['processed'] + self.stats['skipped']:3d}/{self.stats['total_segments']:3d}] "
                                      f"(新規: {self.stats['processed']:2d}, スキップ: {self.stats['skipped']:2d}) "
                                      f"| 推定残り: {eta_str}")

                                completion_message = (
                                    f"[並列処理] セグメント #{idx} 完了"
                                    if status != 'error'
                                    else f"[並列処理] セグメント #{idx} エラー"
                                )
                                progress_queue.put({
                                    'kind': 'complete',
                                    'lane': lane,
                                    'text': completion_message,
                                })
                                progress_queue.put({'kind': 'log', 'text': progress_msg})
                            if status == 'error':
                                progress_queue.put({
                                    'kind': 'log',
                                    'text': f"[並列処理] セグメント #{idx} エラー: {result['error']}",
                                })
                            
                        except Exception as e:
                            progress_queue.put({
                                'kind': 'log',
                                'text': f"[エラー] タスク {task[0]} 失敗: {e}",
                            })
                
                except KeyboardInterrupt:
                    progress_queue.put({
                        'kind': 'log',
                        'text': "🛑 KeyboardInterruptを受信。並列処理を即座に停止します...",
                    })
                    # 即座にシャットダウン（実行中タスクをキャンセル）
                    _shutdown_executor(executor, wait=False, cancel_futures=True)
                    raise
                
                finally:
                    # 必ずexecutorをクリーンアップ
                    _shutdown_executor(executor, wait=True)
                    _set_active_executor(None)
                    progress_queue.put({'kind': 'stop'})
                    progress_monitor.join(timeout=5)
                    progress_renderer.close()
                    if progress_manager is not None:
                        progress_manager.shutdown()
                
                # 並列処理完了後、強制的にメモリクリーンアップ
                print("\n並列処理完了、メモリを解放中...")
                self._aggressive_memory_cleanup()
                time.sleep(0.5)  # 確実に解放されるまで待機
                print("メモリ解放完了")
                step_times['LADA処理'] = time.time() - step_start
            else:
                # 全セグメントが処理済みの場合
                print("すべてのセグメントが既に処理済みです。結合処理に進みます...")
                step_times['LADA処理'] = 0

            
            # ステップ4: 結合
            step_start = time.time()
            sorted_segments = [results[i] for i in sorted(results.keys()) if results[i] is not None]
            
            if not sorted_segments:
                raise ValueError("処理済みセグメントがありません")
            
            # 音声ストリームの確認
            print(f"\n音声ストリームの確認中...")
            audio_count = 0
            for idx, seg in enumerate(sorted_segments[:3], 1):  # 最初の3つをチェック
                if has_audio_stream(seg):
                    audio_count += 1
                    print(f"  セグメント #{idx}: ✓ 音声あり")
                else:
                    print(f"  セグメント #{idx}: ⚠️  音声なし")
            
            if audio_count == 0:
                print(f"\n⚠️  警告: セグメントに音声が含まれていません")
                print(f"    元の動画に音声がない、またはLADA処理で音声が失われた可能性があります")
            else:
                print(f"\n✓ 音声ストリームを確認しました ({audio_count}/3 セグメント)")
            
            merge_videos(sorted_segments, output_path, self.args.merge_encoder)
            step_times['マージ'] = time.time() - step_start
            
            # 結果表示
            total_elapsed = time.time() - total_start_time
            
            print("\n" + "=" * 70)
            print("処理完了！ 🎉")
            print("=" * 70)
            print(f"出力: {output_path}")
            print(f"総処理時間: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分)")
            
            # 処理時間の内訳
            if step_times:
                print(f"\n処理時間の内訳:")
                for step_name, step_time in step_times.items():
                    percentage = (step_time / total_elapsed * 100) if total_elapsed > 0 else 0
                    print(f"  {step_name}: {step_time:.1f}秒 ({percentage:.1f}%)")
            
            print(f"\n並列数: {self.args.parallel_workers}")
            print(f"セグメント総数: {self.stats['total_segments']}")
            print(f"新規処理: {self.stats['processed']}")
            print(f"スキップ: {self.stats['skipped']}")
            
            if self.stats['errors']:
                print(f"エラー: {len(self.stats['errors'])}")
                for err in self.stats['errors'][:5]:
                    print(f"  - {err}")
            
            # 速度向上の推定
            if self.args.parallel_workers > 1:
                theoretical_speedup = self.args.parallel_workers
                print(f"\n理論上の速度向上: {theoretical_speedup}倍")
            
            print("=" * 70)
            
            # 処理済みセグメント削除
            if self.args.delete_segments:
                print("\n処理済みセグメントを削除中...")
                for seg in processed_dir.glob("*.mp4"):
                    seg.unlink()
                print("削除完了")
            
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("⚠️ Ctrl+C により処理が中断されました")
            print("=" * 70)
            
            # 処理済みセグメントの確認
            if processed_dir.exists():
                processed_count = len(list(processed_dir.glob("*.mp4")))
                print(f"処理済みセグメント: {processed_count}個")
                print(f"場所: {processed_dir}")
                print("\n次回実行時に --keep-temp オプションで差分処理が可能です")
            
            # 不完全な出力ファイルを削除
            if output_path.exists():
                try:
                    output_path.unlink()
                    print(f"不完全な出力ファイルを削除しました: {output_path}")
                except:
                    pass
            
            print("=" * 70)
            raise  # KeyboardInterruptを再送出
            
        except Exception as e:
            print(f"\n致命的エラー: {e}")
            raise
        
        finally:
            # 一時ファイル削除（安全性チェック追加）
            if not self.args.keep_temp:
                print("\n一時ファイルを削除中...")
                import shutil
                
                # 出力ファイルが一時ディレクトリ内にないか確認
                try:
                    output_parent = output_path.resolve().parent
                    temp_parent = temp_dir.resolve()
                    
                    # 出力ファイルが一時ディレクトリの子孫の場合は警告
                    if output_parent == temp_parent or temp_parent in output_parent.parents:
                        print("⚠️ 警告: 出力ファイルが一時ディレクトリ内にあります")
                        print("⚠️ 一時ディレクトリの削除をスキップします")
                        print(f"⚠️ 手動で削除してください: {temp_dir}")
                    else:
                        # 安全：一時ディレクトリを削除
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print("削除完了")
                except Exception as e:
                    print(f"⚠️ 一時ファイル削除中にエラー: {e}")
                    print(f"⚠️ 手動で削除してください: {temp_dir}")
            
            # 最終メモリ解放（より積極的に）
            print("\n最終メモリ解放中...")
            self._final_memory_cleanup()
            print("メモリ解放完了")
    
    def _final_memory_cleanup(self):
        """
        最終メモリクリーンアップ
        処理完了後に徹底的にメモリを解放
        """
        try:
            import torch
            
            if self.args.device == 'mps' and torch.backends.mps.is_available():
                # MPS: 徹底的にクリア
                print("  MPS キャッシュクリア中...")
                for _ in range(5):
                    torch.mps.empty_cache()
                    time.sleep(0.1)
                torch.mps.synchronize()
                torch.mps.empty_cache()
                
            elif 'cuda' in self.args.device and torch.cuda.is_available():
                # CUDA: 徹底的にクリア
                print("  CUDA キャッシュクリア中...")
                for _ in range(5):
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except:
            pass
        
        # Python GCを徹底的に実行
        print("  Python GC実行中...")
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
        
        # グローバルクリーンアップ
        cleanup_resources()

# ===== バッチ処理 =====
def process_batch(input_dir, output_dir, temp_dir_base, args):
    """バッチ処理"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
    
    video_files = sorted(set(video_files))
    
    if not video_files:
        print(f"エラー: {input_dir} に動画ファイルが見つかりません")
        return
    
    print(f"\nバッチ処理: {len(video_files)}個の動画")
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(video_files)}] {video_file.name}")
        print('=' * 70)
        
        output_file = output_dir / f"{video_file.stem}-UC{video_file.suffix}"
        
        temp_dir = Path(temp_dir_base) / video_file.stem

        # 既に処理済みか確認。ただし分割済みセグメントに未処理がある場合は再開を優先する。
        if output_file.exists() and not args.overwrite:
            # 簡易チェック（ファイルサイズが100KB以上）
            if output_file.stat().st_size > 100 * 1024 and not has_pending_segment_work(temp_dir):
                print(f"スキップ: 既に存在します")
                skip_count += 1
                continue
            if has_pending_segment_work(temp_dir):
                print("再開: segments に未処理セグメントがあるため処理します")
        
        try:
            processor = ParallelVideoProcessor(args)
            processor.process(video_file, output_file, temp_dir)
            success_count += 1
            
            # 各動画処理後にメモリクリーンアップ
            cleanup_resources()
            time.sleep(0.5)  # 次の動画まで少し待機
            
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("⚠️ Ctrl+C により処理が中断されました")
            print("=" * 70)
            print(f"現在の進捗:")
            print(f"  成功: {success_count}個")
            print(f"  スキップ: {skip_count}個")
            print(f"  エラー: {error_count}個")
            print(f"  残り: {len(video_files) - idx}個")
            print("=" * 70)
            cleanup_resources()
            break
        except Exception as e:
            print(f"エラー: {e}")
            error_count += 1
            cleanup_resources()
            continue
    
    print(f"\n{'=' * 70}")
    print(f"バッチ処理完了")
    print(f"{'=' * 70}")
    print(f"成功: {success_count}個")
    print(f"スキップ: {skip_count}個")
    print(f"エラー: {error_count}個")

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='LADA動画処理スクリプト - 真の並列処理版 🚀',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（1並列）
  python %(prog)s --input video.mp4 --output output.mp4
  
  # 2並列（16GB推奨）
  python %(prog)s --input video.mp4 --output output.mp4 --parallel-workers 2
  
  # 4並列（32GB+推奨）
  python %(prog)s --input video.mp4 --output output.mp4 --parallel-workers 4
  
  # ビットレート倍率指定（2.5倍）
  python %(prog)s --input video.mp4 --output output.mp4 --bitrate-multiplier 2.5
  
  # 品質制御（-q:v）- FFmpeg 4.4+/Apple Silicon
  python %(prog)s --input video.mp4 --output output.mp4 --quality 70
  
  # 品質制御（qmin/qmax）- PyAV 16.0+
  python %(prog)s --input video.mp4 --output output.mp4 --qmin 10 --qmax 30
  
  # fps変換（処理後、デフォルト）
  python %(prog)s --input video.mp4 --output output.mp4 --fps 30
  
  # fps変換（処理前、高速化）
  python %(prog)s --input video.mp4 --output output.mp4 --fps 30 --pre-fps-conversion
  
  # カスタムエンコーダーオプション
  python %(prog)s --input video.mp4 --output output.mp4 --encoder-options "-b:v 20M -maxrate 25M"
  
  # バッチ処理（ディレクトリ）
  python %(prog)s --input /path/to/videos/ --output /path/to/output/ --parallel-workers 2

推奨設定:
  8GB:    --parallel-workers 1 (デフォルト)
  16GB:   --parallel-workers 2 (2倍速)
  32GB+:  --parallel-workers 3-4 (3-4倍速)
  
  ビットレート倍率:
    超高画質: --bitrate-multiplier 4.0〜5.0
    高品質:   --bitrate-multiplier 3.0 (デフォルト)
    バランス: --bitrate-multiplier 2.0〜2.5
    軽量:     --bitrate-multiplier 1.5
  
  品質制御（-q:v）- FFmpeg 4.4+/Apple Silicon限定:
    ※ --quality使用時はビットレート制御が無効化されます
    最高品質: --quality 85〜95 (大容量)
    高品質:   --quality 65〜75 (推奨)
    標準:     --quality 50〜60
    軽量:     --quality 30〜45
    
    注意: FFmpeg 4.4以降とApple Siliconが必須
  
  品質制御（qmin/qmax）- PyAV 16.0+ & FFmpeg 6.0+:
    ※ qmin/qmax使用時はビットレート制御が無効化されます
    最高品質: --qmin 10 --qmax 20 (大容量)
    高品質:   --qmin 10 --qmax 30 (推奨)
    標準:     指定なし (ビットレート制御を使用)
    
    品質モード vs ビットレートモード:
    - quality/qmin/qmax指定時: 品質優先、ファイルサイズ可変
    - 未指定時: ファイルサイズ予測可能
  
  fps変換タイミング:
    処理後（デフォルト）: --fps 30
      → 品質優先、全フレームをLADA処理
    処理前（高速化）: --fps 30 --pre-fps-conversion
      → 速度優先、約26%%高速化（60fps→30fpsの場合）
"""
    )
    
    # 基本設定
    parser.add_argument('--input', help='入力動画ファイルまたはディレクトリ。ディレクトリ指定時は対応動画をまとめて処理')
    parser.add_argument('--output', help='出力動画ファイルまたはディレクトリ。入力がディレクトリの場合は出力もディレクトリ指定')
    parser.add_argument('--temp-dir', default='/tmp', help='セグメント、処理済み中間動画、結合リストを置く一時ディレクトリ（デフォルト: /tmp）')
    parser.add_argument('--ffmpeg-temp-dir', help='FFmpegの分割/結合用一時ディレクトリ。未指定時は temp-dir 配下を自動使用')
    parser.add_argument('--lada-temp-dir', help='各セグメント内で lada-cli が使う一時ディレクトリ。未指定時はシステム既定')
    
    # 並列処理設定
    parser.add_argument('--parallel-workers', type=int, default=1,
                        help='同時に処理するセグメント数。MPS/統一メモリでは増やしすぎるとswapで遅くなる（デフォルト: 1、目安: 16GB=1〜2, 32GB=2〜4）')
    parser.add_argument('--executor', choices=['process', 'thread'], default='process',
                        help='セグメント処理の並列実行方式（process=プロセス分離、thread=旧ThreadPool方式。デフォルト: process）')
    
    # セグメント設定
    parser.add_argument('--segment-duration', type=int, default=60,
                        help='入力を分割する長さ（秒）。長いほど時間軸の整合性は保ちやすいが、失敗時のやり直しとメモリ負荷が増える（デフォルト: 60）')
    parser.add_argument('--segment-count', type=int, default=None,
                        help='入力を指定個数に均等分割する。指定時は --segment-duration より優先（例: --segment-count 8）')
    parser.add_argument('--merge-encoder', default='copy',
                        help='最終結合時のエンコーダー。copyは再エンコードせず高速・画質劣化なし（デフォルト: copy）')
    parser.add_argument('--delete-segments', action='store_true',
                        help='正常終了後に処理済みセグメントを削除してディスク使用量を減らす')
    parser.add_argument('--keep-temp', action='store_true',
                        help='一時ファイルを保持。処理済みセグメント確認、途中再開、品質比較に使う')
    parser.add_argument('--force-split', action='store_true',
                        help='既存セグメントを無視して強制的に再分割')
    
    # LADA設定
    try:
        import torch
        try:
            from lada.utils.os_utils import get_default_torch_device, gpu_has_fp16_acceleration
            default_device = get_default_torch_device()
            default_fp16 = gpu_has_fp16_acceleration(torch.device(default_device))
        except Exception:
            # LADAユーティリティが読めない場合のフォールバック
            default_device = 'cpu'
            if torch.cuda.is_available():
                default_device = 'cuda:0'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                default_device = 'mps'
            default_fp16 = default_device != 'cpu'
    except ImportError:
        default_device = 'cpu'
        default_fp16 = False
    
    parser.add_argument('--device', default=default_device,
                        help=f'lada-cli に渡す推論デバイス。例: cpu, cuda:0, mps（デフォルト: {default_device}）')
    parser.add_argument('--fp16', action='store_true', default=default_fp16,
                        help='半精度(fp16)を有効化。CUDAでは高速化しやすいが、MPSでは不安定な場合あり')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='半精度(fp16)を無効化。MPSでabortや画質/安定性問題がある場合はこちらを推奨')
    
    # エンコーディング設定
    parser.add_argument('--encoding-preset', help='lada-cli のエンコーディングプリセット名。未指定時は解像度/デバイスに応じて自動選択')
    parser.add_argument('--list-encoding-presets', action='store_true',
                        help='lada-cli のエンコーディングプリセット一覧を表示')
    parser.add_argument('--encoder', help='出力エンコーダー。例: h264_videotoolbox, hevc_videotoolbox, libx264。未指定時は自動選択')
    parser.add_argument('--list-encoders', action='store_true',
                        help='lada-cli のエンコーダー一覧を表示')
    parser.add_argument('--encoder-options', help='lada-cli/FFmpegへ渡すエンコーダー追加オプション文字列。自動設定を上書きしたい場合に使用')
    parser.add_argument('--list-encoder-options', metavar='ENCODER',
                        help='指定エンコーダーの lada-cli オプション一覧を表示')
    parser.add_argument('--bitrate-multiplier', type=float, default=3.0,
                        help='元動画ビットレートに掛ける倍率。大きいほど画質保持・ファイルサイズ増（デフォルト: 3.0、範囲: 0.1〜10.0）')
    parser.add_argument('--quality', type=int, default=None,
                        help='品質値 -q:v（0-100、VideoToolbox専用、FFmpeg 4.4+/Apple Silicon必須）。100=最高品質、0=最低品質')
    parser.add_argument('--qmin', type=int, default=None,
                        help='最小量子化値（0-51、小さいほど高品質）PyAV 16.0+で利用可能')
    parser.add_argument('--qmax', type=int, default=None,
                        help='最大量子化値（0-51、大きいほど低品質）PyAV 16.0+で利用可能')
    parser.add_argument('--fps', type=int, default=None,
                        help='出力フレームレート（例: 30）。指定しない場合は元のfpsを維持')
    parser.add_argument('--pre-fps-conversion', action='store_true',
                        help='LADA処理前にfps変換して処理フレーム数を減らす。速度優先だが、復元対象フレームも減る')
    parser.add_argument('--mp4-fast-start', action='store_true',
                        help='MP4のfaststart/fragment設定を有効化し、書き込み中/転送中の再生互換性を上げる')
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                        help='解像度に応じた自動最適化（デフォルト: True）')
    parser.add_argument('--no-auto-optimize', dest='auto_optimize', action='store_false',
                        help='自動最適化を無効化')
    
    # モザイク設定
    parser.add_argument('--list-mosaic-restoration-models', action='store_true',
                        help='lada-cli の復元モデル一覧を表示')
    default_restoration_model = get_default_mosaic_restoration_model()
    parser.add_argument('--mosaic-restoration-model', default=default_restoration_model,
                        help=f'復元モデル名または重みパス（デフォルト: {default_restoration_model}）')
    parser.add_argument('--max-clip-length', type=int, default=None,
                        help='復元モデルへ渡す最大フレーム数。未指定時はCore AI T18=98、T36=104、T90=178、その他=180')
    parser.add_argument('--restore-max-frames', type=int, default=None,
                        help='BasicVSR++復元チャンク数の上書き。未指定はMPS自動調整、-1は分割なし、正の値は固定チャンク')
    parser.add_argument('--restore-sharpen-strength', type=float, default=0.0,
                        help='復元ROIを合成前にunsharp maskでシャープ化する強度（0で無効、例: 0.3）')
    parser.add_argument('--restore-detail-boost', type=float, default=0.0,
                        help='復元ROIの局所ディテール/コントラストを合成前に強める強度（0で無効、例: 0.15）')
    parser.add_argument('--restore-blend-feather', type=float, default=1.0,
                        help='復元ROI境界ブレンドのぼかし倍率（1.0で標準、例: 1.0〜1.5）')
    parser.add_argument('--restore-texture-mix', type=float, default=0.0,
                        help='元ROIの中周波テクスチャを復元ROIへ薄く戻す強度（0で無効、例: 0.08）')
    parser.add_argument('--restore-smooth-strength', type=float, default=0.0,
                        help='texture/detail/sharpen後の復元ROIを合成前に滑らかにする強度（0で無効、例: 0.10〜0.25）')
    parser.add_argument('--restore-effect-upscale', type=int, default=1,
                        help='texture/detail/sharpenをOpenCVで拡大後に適用して戻す倍率（1で無効、例: 2）')
    parser.add_argument('--restore-roi-enhancer', choices=('none', 'realesrgan', 'mewzoom', 'swinir'), default='none',
                        help='復元ROIに追加の高画質化処理をかける。モデルは指定時のみロード（デフォルト: none）')
    parser.add_argument('--restore-roi-enhancer-model-path',
                        help='エンハンサーモデル。登録名（realesrgan-x2/x4, realesrgan-x4-coreml, mewzoom-x4-coreml）、'
                             'model_weights内のファイル名、またはパス。省略時は <enhancer>-x4-coreml に自動解決')
    parser.add_argument('--restore-roi-enhancer-scale', type=int, default=2,
                        help='エンハンサーの倍率。処理後はROIサイズへ戻して合成（デフォルト: 2）')
    parser.add_argument('--restore-roi-enhancer-strength', type=float, default=0.0,
                        help='エンハンサー結果を復元ROIへ混ぜる強度（0で無効、例: 0.25）')
    parser.add_argument('--restore-roi-enhancer-tile', type=int, default=0,
                        help='PyTorch版Real-ESRGANのtileサイズ。0で無効。Core ML版では未使用（デフォルト: 0）')
    default_detection_model = get_default_mosaic_detection_model()
    parser.add_argument('--mosaic-detection-model', default=default_detection_model,
                        help=f'検出モデル名または重みパス（デフォルト: {default_detection_model}）')
    parser.add_argument('--list-mosaic-detection-models', action='store_true',
                        help='lada-cli の検出モデル一覧を表示')
    parser.add_argument('--mosaic-detection-empty-lookahead', type=int, default=10,
                        help='先読み範囲の先頭/末尾がどちらも検出なしなら、その範囲を検出なしとしてスキップ（0で無効、デフォルト: 10）')
    parser.add_argument('--detect-face-mosaics', action='store_true',
                        help='顔モザイク検出を有効化')
    parser.add_argument('--no-detect-face-mosaics', dest='detect_face_mosaics', action='store_false',
                        help='顔モザイク検出を無効化（デフォルト）')
    parser.set_defaults(detect_face_mosaics=False)  # デフォルトはFalse
    
    # メモリ管理
    parser.add_argument('--memory-cleanup-interval', type=int, default=1,
                        help='何セグメントごとにメモリ掃除を試みるか。並列処理では各セグメント後にも掃除する（デフォルト: 1）')
    parser.add_argument('--cleanup-trigger-gb', type=float, default=4.0,
                        help='利用可能メモリがこの値(GB)未満の時のみ並列中クリーンアップを強化（デフォルト: 4.0）')
    parser.add_argument('--mps-memory-fraction', type=float, default=None,
                        help='torch.mps.set_per_process_memory_fraction に渡す値。未指定時は並列数に応じて自動設定')
    parser.add_argument('--log-mps-memory', action='store_true',
                        help='起動時に torch.mps のメモリ統計を表示')
    
    # その他
    parser.add_argument('--list-devices', action='store_true',
                        help='lada-cli の利用可能デバイス一覧を表示')
    parser.add_argument('--overwrite', action='store_true',
                        help='既存の出力/処理済みセグメントを再利用せず、上書きして再処理する')

    return parser


# ===== メイン関数 =====
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    list_return_code = run_lada_cli_list_command(args)
    if list_return_code is not None:
        if list_return_code != 0:
            sys.exit(list_return_code)
        return

    args.max_clip_length = get_effective_max_clip_length(
        args.mosaic_restoration_model,
        args.max_clip_length,
    )
    requested_workers = args.parallel_workers
    args.parallel_workers = get_memory_safe_parallel_workers(
        args.mosaic_restoration_model,
        requested_workers,
    )
    if args.parallel_workers != requested_workers:
        print(
            "Core AI T90は統一メモリ使用量が大きいため、"
            f"並列数を {requested_workers} から 1 に制限します"
        )

    if not args.input:
        parser.error("--input is required unless using --list-*")
    if not args.output:
        parser.error("--output is required unless using --list-*")
    
    # 並列数の検証
    if args.parallel_workers < 1:
        print("エラー: --parallel-workers は1以上である必要があります")
        return
    if args.max_clip_length < 1:
        print("エラー: --max-clip-length は1以上である必要があります")
        return
    if args.segment_count is not None and args.segment_count < 1:
        print("エラー: --segment-count は1以上である必要があります")
        return
    if args.memory_cleanup_interval < 1:
        print("エラー: --memory-cleanup-interval は1以上である必要があります")
        return
    if args.cleanup_trigger_gb <= 0:
        print("エラー: --cleanup-trigger-gb は0より大きい必要があります")
        return
    if args.mps_memory_fraction is not None and not (0.0 < args.mps_memory_fraction <= 1.0):
        print("エラー: --mps-memory-fraction は 0 より大きく 1 以下である必要があります")
        return
    if args.restore_sharpen_strength < 0:
        print("エラー: --restore-sharpen-strength は0以上である必要があります")
        return
    if args.restore_detail_boost < 0:
        print("エラー: --restore-detail-boost は0以上である必要があります")
        return
    if args.restore_blend_feather < 0:
        print("エラー: --restore-blend-feather は0以上である必要があります")
        return
    if args.restore_texture_mix < 0:
        print("エラー: --restore-texture-mix は0以上である必要があります")
        return
    if args.restore_smooth_strength < 0:
        print("エラー: --restore-smooth-strength は0以上である必要があります")
        return
    if args.restore_effect_upscale < 1:
        print("エラー: --restore-effect-upscale は1以上である必要があります")
        return
    # model-path省略時はlada-cli側で <enhancer>-x4-coreml に既定解決される
    if args.restore_roi_enhancer_scale < 1:
        print("エラー: --restore-roi-enhancer-scale は1以上である必要があります")
        return
    if args.restore_roi_enhancer_strength < 0:
        print("エラー: --restore-roi-enhancer-strength は0以上である必要があります")
        return
    if args.restore_roi_enhancer_tile < 0:
        print("エラー: --restore-roi-enhancer-tile は0以上である必要があります")
        return
    if args.mosaic_detection_empty_lookahead < 0:
        print("エラー: --mosaic-detection-empty-lookahead は0以上である必要があります")
        return
    
    if args.parallel_workers > 1:
        print("\n" + "=" * 70)
        print("⚠️  並列処理モード")
        print("=" * 70)
        print(f"並列数: {args.parallel_workers}")
        print(f"推奨メモリ: {args.parallel_workers * 10}GB以上")
        print("メモリ不足の場合はスワップが発生し、逆に遅くなる可能性があります")
        print("=" * 70 + "\n")
    
    # デバイス情報表示
    print("=" * 70)
    print("デバイス情報")
    print("=" * 70)
    print(f"選択デバイス: {args.device}")
    print(f"並列処理数: {args.parallel_workers}")
    print(f"復元Clip長: {args.max_clip_length}フレーム")
    if args.device == 'mps':
        effective_mps_fraction = get_effective_mps_memory_fraction(args)
        if effective_mps_fraction is not None:
            print(f"MPS メモリ上限: {effective_mps_fraction:.2f} / process")
            configure_mps_runtime(effective_mps_fraction, verbose=False)
        if args.log_mps_memory:
            stats_line = format_mps_memory_stats()
            if stats_line:
                print(stats_line)
    print("=" * 70 + "\n")
    
    # 処理実行
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"エラー: 入力パス '{input_path}' が見つかりません")
        return
    
    # 単一ファイル処理
    if input_path.is_file():
        temp_dir = Path(args.temp_dir) / input_path.stem
        output_path = resolve_single_output_path(input_path, output_path)
        processor = ParallelVideoProcessor(args)
        processor.process(input_path, output_path, temp_dir)
    
    # バッチ処理
    elif input_path.is_dir():
        temp_dir_base = Path(args.temp_dir)
        process_batch(input_path, output_path, temp_dir_base, args)
    
    else:
        print(f"エラー: '{input_path}' はファイルでもディレクトリでもありません")

if __name__ == "__main__":
    main()
