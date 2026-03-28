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
from pathlib import Path
import json
import gc
import time
import math
import signal
import sys
import atexit
from multiprocessing import resource_tracker
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

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
    
    # リソーストラッカーのクリーンアップ
    try:
        resource_tracker._resource_tracker._stop()
    except:
        pass

# ===== シグナルハンドリング（Ctrl+C即座停止対応） =====
# グローバル停止フラグ
_shutdown_requested = False
_force_shutdown = False

def signal_handler(signum, frame):
    """Ctrl+CまたはGUI停止時のハンドリング（改善版）"""
    global _shutdown_requested, _force_shutdown
    
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    
    if _force_shutdown:
        # 2回目のCtrl+C → 即座に終了
        print(f"\n\n⚠️  強制終了します...")
        sys.exit(1)
    
    if _shutdown_requested:
        # 既に停止処理中 → 次回は強制終了
        print(f"\n\n⏳ 停止処理中です。もう一度 Ctrl+C を押すと強制終了します...")
        _force_shutdown = True
        return
    
    # 初回のCtrl+C
    print(f"\n\n🛑 {signal_name}を受信しました。処理を停止中...")
    print(f"💡 完全に停止するまでお待ちください。すぐに終了したい場合はもう一度 Ctrl+C を押してください。")
    _shutdown_requested = True
    
    # KeyboardInterruptを発生させてfinallyブロックを実行
    raise KeyboardInterrupt(f"{signal_name}による中断")

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

def split_video(input_video, output_dir, segment_duration=60, force_split=False, pre_fps=None, encoder_options=None):
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
    
    # 既存セグメントをチェック
    existing_segments = sorted(output_dir.glob("segment_*.mp4"))
    
    if existing_segments and not force_split:
        # 既存セグメントがある場合
        # 期待されるセグメント数を計算
        duration = get_video_duration(input_video)
        if duration:
            expected_count = calculate_segment_count(duration, segment_duration)
            
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
    
    # 分割実行
    if pre_fps:
        # fps変換しながら分割（再エンコード）
        print(f"動画を{segment_duration}秒ごとに分割中（fps変換: {pre_fps}fps）...")
        
        # エンコーダーオプションをパース
        encoder_opts = []
        if encoder_options:
            # 文字列を安全に分割してリストに変換
            import shlex
            try:
                encoder_opts = shlex.split(encoder_options)
            except ValueError:
                # shlex.splitが失敗した場合は単純にスペースで分割
                encoder_opts = encoder_options.split()
        
        cmd = [
            'ffmpeg', '-i', str(input_video),
            '-c:v', 'h264_videotoolbox',  # VideoToolboxエンコーダー
            '-r', str(pre_fps),            # fps変換
        ]
        
        # エンコーダーオプションを追加
        if encoder_opts:
            cmd.extend(encoder_opts)
        
        cmd.extend([
            '-c:a', 'copy',                # 音声はコピー
            '-map', '0',
            '-segment_time', str(segment_duration),
            '-f', 'segment',
            '-reset_timestamps', '1',
            str(output_pattern)
        ])
    else:
        # コピーモード（従来の動作）
        print(f"動画を{segment_duration}秒ごとに分割中...")
        cmd = [
            'ffmpeg', '-i', str(input_video),
            '-map', '0:v',              # ビデオストリーム
            '-map', '0:a?',             # 音声ストリーム（あれば）
            '-c:v', 'copy',             # ビデオをコピー
            '-c:a', 'copy',             # 音声をコピー
            '-segment_time', str(segment_duration),
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

def merge_videos(segment_paths, output_path, encoder='copy'):
    """複数の動画を結合（音声も含む）"""
    print(f"\n動画を結合中... ({len(segment_paths)}個のセグメント)")
    
    concat_file = output_path.parent / "concat_list.txt"
    
    with open(concat_file, 'w') as f:
        for segment in segment_paths:
            f.write(f"file '{segment.resolve()}'\n")
    
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
        """lada-cliコマンドが利用可能か確認"""
        import shutil
        
        lada_cli_path = shutil.which('lada-cli')
        
        if lada_cli_path is None:
            print("\n" + "=" * 70)
            print("⚠️  エラー: lada-cliコマンドが見つかりません")
            print("=" * 70)
            print("\n以下のいずれかの方法で解決してください:\n")
            print("方法1: LADAをインストール")
            print("  cd /path/to/lada-integrated")
            print("  pip install -e .\n")
            print("方法2: python -m lada.cliを使用")
            print("  スクリプトを編集して呼び出し方法を変更してください\n")
            print("=" * 70)
            raise FileNotFoundError("lada-cli command not found")
        else:
            print(f"✓ lada-cli検出: {lada_cli_path}")
    
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
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            env['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
        
        env['OMP_NUM_THREADS'] = '2'
        env['MKL_NUM_THREADS'] = '2'
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['NUMEXPR_NUM_THREADS'] = '2'
        env['PYTHONMALLOC'] = 'malloc'
        env['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        
        # コマンド構築
        cmd = [
            'lada-cli',
            '--input', str(input_video),
            '--output', str(output_video),
            '--device', self.args.device,
        ]
        
        if self.args.fp16:
            cmd.append('--fp16')
        else:
            cmd.append('--no-fp16')
        
        # エンコーディング設定（解像度最適化）
        if self.args.encoding_preset:
            # プリセット指定時はそのまま使用
            cmd.extend(['--encoding-preset', self.args.encoding_preset])
        elif self.args.encoder:
            # エンコーダー指定時
            cmd.extend(['--encoder', self.args.encoder])
            # 最適化されたオプションを使用（ユーザー指定がある場合はそちらを優先）
            if self.optimal_encoder_options:
                # encoder-optionsは文字列として1つの引数で渡す
                cmd.extend(['--encoder-options', str(self.optimal_encoder_options)])
        elif self.optimal_encoder_options:
            # エンコーダー未指定で最適化オプションがある場合
            # macOSではVideoToolbox（ハードウェアエンコーダ）を使用
            cmd.extend(['--encoder', 'h264_videotoolbox'])
            # encoder-optionsは文字列として1つの引数で渡す
            cmd.extend(['--encoder-options', str(self.optimal_encoder_options)])
        
        # ユーザーが直接encoder-optionsを指定している場合も処理
        elif self.args.encoder_options:
            # デフォルトエンコーダーを設定
            if not self.args.encoder:
                cmd.extend(['--encoder', 'h264_videotoolbox'])
            cmd.extend(['--encoder-options', str(self.args.encoder_options)])
        
        if self.args.mp4_fast_start:
            cmd.append('--mp4-fast-start')
        
        cmd.extend(['--mosaic-restoration-model', self.args.mosaic_restoration_model])
        cmd.extend(['--max-clip-length', str(self.args.max_clip_length)])
        cmd.extend(['--mosaic-detection-model', self.args.mosaic_detection_model])
        
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
                # エラーメッセージは表示
                elif 'error' in line.lower() or 'warning' in line.lower():
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
        print(f"セグメント長: {self.args.segment_duration}秒")
        print("=" * 70 + "\n")
        
        # 一時ディレクトリ作成
        temp_dir.mkdir(parents=True, exist_ok=True)
        segments_dir = temp_dir / "segments"
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpegの一時ディレクトリを設定（グローバル）
        ffmpeg_temp = temp_dir / 'ffmpeg_temp'
        ffmpeg_temp.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(ffmpeg_temp)  # macOS/Linux
        os.environ['TEMP'] = str(ffmpeg_temp)    # Windows互換
        os.environ['TMP'] = str(ffmpeg_temp)     # Windows互換
        print(f"📁 FFmpeg一時ディレクトリ: {ffmpeg_temp}\n")
        
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
                    encoder_options=self.optimal_encoder_options
                )
            else:
                # 従来の動作（コピーモード）
                segments = split_video(
                    input_video, 
                    segments_dir, 
                    self.args.segment_duration, 
                    force_split
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
                    if output_path_seg.stat().st_size > 100 * 1024:
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
                executor = ThreadPoolExecutor(max_workers=self.args.parallel_workers)
                try:
                    # 全タスクを投入
                    future_to_task = {
                        executor.submit(self.process_segment, task): task 
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
                            idx, output_path_seg, status = future.result()
                            results[idx] = output_path_seg
                            completed += 1
                            
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
                                
                                # 常に改行表示（並列処理では上書きは混乱する）
                                print(progress_msg, flush=True)
                            
                        except Exception as e:
                            self.safe_print(f"\n[エラー] タスク {task[0]} 失敗: {e}")
                    
                    print()  # 改行
                
                except KeyboardInterrupt:
                    print("\n\n🛑 KeyboardInterruptを受信。並列処理を即座に停止します...")
                    # 即座にシャットダウン（実行中タスクをキャンセル）
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                
                finally:
                    # 必ずexecutorをクリーンアップ
                    if not executor._shutdown:
                        executor.shutdown(wait=True)
                
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
        
        # 既に処理済みか確認
        if output_file.exists() and not args.overwrite:
            # 簡易チェック（ファイルサイズが100KB以上）
            if output_file.stat().st_size > 100 * 1024:
                print(f"スキップ: 既に存在します")
                skip_count += 1
                continue
        
        temp_dir = Path(temp_dir_base) / video_file.stem
        
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

# ===== メイン関数 =====
def main():
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
      → 速度優先、約26%高速化（60fps→30fpsの場合）
"""
    )
    
    # 基本設定
    parser.add_argument('--input', required=True, help='入力動画ファイルまたはディレクトリ')
    parser.add_argument('--output', required=True, help='出力動画ファイルまたはディレクトリ')
    parser.add_argument('--temp-dir', default='/tmp', help='一時ディレクトリ')
    parser.add_argument('--lada-temp-dir', help='lada-cli内部用一時ディレクトリ')
    
    # 並列処理設定
    parser.add_argument('--parallel-workers', type=int, default=1,
                        help='並列処理数（デフォルト: 1、推奨: 8GB=1, 16GB=2, 32GB=4）')
    
    # セグメント設定
    parser.add_argument('--segment-duration', type=int, default=60, 
                        help='セグメント長（秒）（デフォルト: 60）')
    parser.add_argument('--merge-encoder', default='copy', 
                        help='マージ時のエンコーダー（デフォルト: copy）')
    parser.add_argument('--delete-segments', action='store_true', 
                        help='処理済みセグメントを削除')
    parser.add_argument('--keep-temp', action='store_true', 
                        help='一時ファイルを保持（デバッグ用）')
    parser.add_argument('--force-split', action='store_true',
                        help='既存セグメントを無視して強制的に再分割')
    
    # LADA設定
    try:
        import torch
        default_device = 'cpu'
        if torch.backends.mps.is_available():
            default_device = 'mps'
        elif torch.cuda.is_available():
            default_device = 'cuda:0'
    except ImportError:
        default_device = 'cpu'
    
    parser.add_argument('--device', default=default_device, 
                        help=f'デバイス（デフォルト: {default_device}）')
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    
    # エンコーディング設定
    parser.add_argument('--encoding-preset', help='エンコーディングプリセット')
    parser.add_argument('--encoder', help='エンコーダー')
    parser.add_argument('--encoder-options', help='エンコーダーオプション')
    parser.add_argument('--bitrate-multiplier', type=float, default=3.0,
                        help='ビットレート倍率（デフォルト: 3.0、範囲: 0.1〜10.0）')
    parser.add_argument('--quality', type=int, default=None,
                        help='品質値 -q:v（0-100、VideoToolbox専用、FFmpeg 4.4+/Apple Silicon必須）。100=最高品質、0=最低品質')
    parser.add_argument('--qmin', type=int, default=None,
                        help='最小量子化値（0-51、小さいほど高品質）PyAV 16.0+で利用可能')
    parser.add_argument('--qmax', type=int, default=None,
                        help='最大量子化値（0-51、大きいほど低品質）PyAV 16.0+で利用可能')
    parser.add_argument('--fps', type=int, default=None,
                        help='出力フレームレート（例: 30）。指定しない場合は元のfpsを維持')
    parser.add_argument('--pre-fps-conversion', action='store_true',
                        help='セグメント分割時にfps変換を実行（LADA処理前、処理高速化）')
    parser.add_argument('--mp4-fast-start', action='store_true')
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                        help='解像度に応じた自動最適化（デフォルト: True）')
    parser.add_argument('--no-auto-optimize', dest='auto_optimize', action='store_false',
                        help='自動最適化を無効化')
    
    # モザイク設定
    parser.add_argument('--mosaic-restoration-model', default='basicvsrpp-v1.2')
    parser.add_argument('--max-clip-length', type=int, default=180)
    parser.add_argument('--mosaic-detection-model', default='v4-fast')
    parser.add_argument('--detect-face-mosaics', action='store_true',
                        help='顔モザイク検出を有効化')
    parser.add_argument('--no-detect-face-mosaics', dest='detect_face_mosaics', action='store_false',
                        help='顔モザイク検出を無効化（デフォルト）')
    parser.set_defaults(detect_face_mosaics=False)  # デフォルトはFalse
    
    # メモリ管理
    parser.add_argument('--memory-cleanup-interval', type=int, default=1,
                        help='（並列処理では常に各セグメント後にクリーンアップ実行）')
    
    # その他
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()
    
    # 並列数の検証
    if args.parallel_workers < 1:
        print("エラー: --parallel-workers は1以上である必要があります")
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
