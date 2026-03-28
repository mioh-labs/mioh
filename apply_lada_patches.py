#!/usr/bin/env python3
"""
Apply patches to LADA environment site-packages

This script applies various patches to fix compatibility issues and
optimize performance in the LADA environment.

Supports:
- pyenv virtual environments
- .venv virtual environments (Linux/macOS standard)
- Python 3.12 and 3.13
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import site


def find_site_packages() -> Path:
    """
    Automatically detect site-packages directory.
    
    Priority:
    1. Current virtual environment (if active)
    2. .venv directory in current/parent directories (Linux/macOS/Windows)
    3. Manual specification via --site-packages argument
    
    Supports:
    - Linux/macOS: .venv/lib/python3.1X/site-packages
    - Windows: .venv/Lib/site-packages
    - pyenv: ~/.pyenv/versions/X.Y.Z/envs/NAME/lib/pythonX.Y/site-packages
    """
    # Check if we're in a virtual environment
    if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        # We're in a virtual environment
        site_packages_dirs = site.getsitepackages()
        if site_packages_dirs:
            sp_path = Path(site_packages_dirs[0])
            print(f"Detected virtual environment site-packages: {sp_path}")
            return sp_path
    
    # Try to find .venv in current or parent directories
    current = Path.cwd()
    for _ in range(3):  # Search up to 3 levels up
        venv_path = current / ".venv"
        if venv_path.exists():
            # Try different Python version paths
            # Windows: .venv/Lib/site-packages
            windows_path = venv_path / "Lib" / "site-packages"
            if windows_path.exists():
                print(f"Found .venv site-packages (Windows): {windows_path}")
                return windows_path
            
            # Linux/macOS: .venv/lib/python3.1X/site-packages
            for py_version in ["3.13", "3.12"]:
                unix_path = venv_path / "lib" / f"python{py_version}" / "site-packages"
                if unix_path.exists():
                    print(f"Found .venv site-packages (Unix): {unix_path}")
                    return unix_path
        current = current.parent
    
    # Could not auto-detect
    return None


# Try to auto-detect site-packages
SITE_PACKAGES = find_site_packages()

# Model weights directory - current directory
MODEL_WEIGHTS_DIR = Path.cwd() / "model_weights"


def create_backup(file_path: Path) -> Path:
    """Create a backup of the file before patching."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".backup_{timestamp}")
    shutil.copy2(file_path, backup_path)
    print(f"  ✓ Backup created: {backup_path.name}")
    return backup_path


def apply_patch_mmengine_resume_dataloader():
    """
    Patch: adjust_mmengine_resume_dataloader.patch
    Remove dataloader skip logic when resuming training
    """
    print("\n[1/7] Applying mmengine resume dataloader patch...")
    
    file_path = SITE_PACKAGES / "mmengine" / "runner" / "loops.py"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the dataloader skip block
    old_block = """        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
"""
    
    if old_block in content:
        content = content.replace(old_block, "")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    else:
        print("  ⚠ Code block not found (may already be patched)")
        return False


def apply_patch_mmengine_torch26_weights():
    """
    Patch: fix_loading_mmengine_weights_on_torch26_and_higher.diff
    Add weights_only=False to torch.load for PyTorch 2.6+ compatibility
    """
    print("\n[2/7] Applying mmengine torch 2.6+ weights loading patch...")
    
    file_path = SITE_PACKAGES / "mmengine" / "runner" / "checkpoint.py"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace torch.load line
    old_line = "    checkpoint = torch.load(filename, map_location=map_location)"
    new_line = "    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    elif new_line in content:
        print("  ⚠ Already patched")
        return True
    else:
        print("  ✗ Code line not found")
        return False


def apply_patch_ultralytics_mms_time_limit():
    """
    Patch: increase_mms_time_limit.patch
    Increase max_time_img from 0.05 to 0.3
    """
    print("\n[3/7] Applying ultralytics MMS time limit patch...")
    
    file_path = SITE_PACKAGES / "ultralytics" / "utils" / "nms.py"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace max_time_img default value
    old_line = "    max_time_img: float = 0.05,"
    new_line = "    max_time_img: float = 0.3,"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    elif new_line in content:
        print("  ⚠ Already patched")
        return True
    else:
        print("  ✗ Code line not found")
        return False


def apply_patch_mmengine_torch_dist_compatibility():
    """
    Patch: remove_use_of_torch_dist_in_mmengine.patch
    Add torch.distributed compatibility for MPS/non-distributed environments
    """
    print("\n[4/7] Applying MMEngine torch.distributed compatibility patch...")
    
    results = []
    
    # Patch 1: mmengine/dist/dist.py
    file_path1 = SITE_PACKAGES / "mmengine" / "dist" / "dist.py"
    if file_path1.exists():
        create_backup(file_path1)
        
        with open(file_path1, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add DummyReduceOp after imports
        import_section = "from mmengine.device import is_npu_available"
        patch_code = """from mmengine.device import is_npu_available


if not hasattr(torch.distributed, "ReduceOp"):
    class DummyReduceOp:
        SUM = None
        MEAN = None
    torch.distributed.ReduceOp = DummyReduceOp
"""
        
        if import_section in content and "DummyReduceOp" not in content:
            content = content.replace(import_section, patch_code)
            with open(file_path1, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ Patched mmengine/dist/dist.py")
            results.append(True)
        elif "DummyReduceOp" in content:
            print("  ⚠ mmengine/dist/dist.py already patched")
            results.append(True)
        else:
            print("  ✗ Could not patch mmengine/dist/dist.py")
            results.append(False)
    else:
        print(f"  ✗ File not found: {file_path1}")
        results.append(False)
    
    # Patch 2: mmengine/model/wrappers/__init__.py
    file_path2 = SITE_PACKAGES / "mmengine" / "model" / "wrappers" / "__init__.py"
    if file_path2.exists():
        create_backup(file_path2)
        
        with open(file_path2, 'r', encoding='utf-8') as f:
            content = f.read()
        
        changes_made = False
        
        # Add torch.distributed namespace setup at the top
        first_line = "# Copyright (c) OpenMMLab. All rights reserved."
        namespace_code = """# Copyright (c) OpenMMLab. All rights reserved.
import torch, types
if not hasattr(torch, "distributed") or not hasattr(torch.distributed, "fsdp"):
    torch.distributed = types.SimpleNamespace()
    torch.distributed.fsdp = types.SimpleNamespace()
    torch.distributed.fsdp.fully_sharded_data_parallel = types.SimpleNamespace()


"""
        
        if first_line in content and "types.SimpleNamespace()" not in content:
            content = content.replace(first_line + "\n", namespace_code)
            changes_made = True
        elif "types.SimpleNamespace()" in content:
            print("  ⚠ mmengine/model/wrappers/__init__.py namespace already added")
        
        # Wrap FSDP import in try-except
        old_fsdp_import = """if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):
    from .fully_sharded_distributed import \
        MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')"""
        
        new_fsdp_import = """if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):
    try:
        from .fully_sharded_distributed import MMFullyShardedDataParallel  # noqa:F401
    except Exception as e:
        import warnings
        warnings.warn(f"FSDP disabled: {e}")
        MMFullyShardedDataParallel = None
        
        
    __all__.append('MMFullyShardedDataParallel')"""
        
        if old_fsdp_import in content:
            content = content.replace(old_fsdp_import, new_fsdp_import)
            changes_made = True
        elif "FSDP disabled" in content:
            print("  ⚠ mmengine/model/wrappers/__init__.py FSDP already wrapped")
        
        if changes_made:
            with open(file_path2, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ Patched mmengine/model/wrappers/__init__.py")
            results.append(True)
        else:
            print("  ⚠ mmengine/model/wrappers/__init__.py already patched")
            results.append(True)
    else:
        print(f"  ✗ File not found: {file_path2}")
        results.append(False)
    
    return all(results)


def apply_patch_gvsbuild_ffmpeg():
    """
    Patch: gvsbuild_ffmpeg.patch
    Enable mp2float decoder in FFmpeg build
    """
    print("\n[5/7] Applying gvsbuild FFmpeg mp2float decoder patch...")
    
    file_path = SITE_PACKAGES / "gvsbuild" / "patches" / "ffmpeg" / "build" / "build.sh"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        print("  ℹ This patch is only needed for gvsbuild installations")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add mp2float decoder configuration
    old_line = 'configure_cmd[idx++]="--disable-swresample"'
    new_lines = '''configure_cmd[idx++]="--disable-swresample"
configure_cmd[idx++]="--enable-decoder=mp2float"'''
    
    if old_line in content and '--enable-decoder=mp2float' not in content:
        content = content.replace(old_line, new_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    elif '--enable-decoder=mp2float' in content:
        print("  ⚠ Already patched")
        return True
    else:
        print("  ✗ Code line not found")
        return False


def apply_patch_lada_adw_to_gtk_spinner():
    """
    Patch: adw_spinner_to_gtk_spinner.patch
    Replace AdwSpinner with GtkSpinner in preview UI
    """
    print("\n[6/7] Applying LADA AdwSpinner to GtkSpinner patch...")
    
    file_path = SITE_PACKAGES / "lada" / "gui" / "preview" / "preview_view.ui"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = False
    
    # Replace first AdwSpinner with GtkSpinner
    old_spinner1 = '''                                                                    <object class="AdwSpinner" id="spinner_overlay">
                                                                        <property name="width-request">64</property>'''
    new_spinner1 = '''                                                                    <object class="GtkSpinner" id="spinner_overlay">
                                                                        <property name="spinning">True</property>
                                                                        <property name="width-request">64</property>'''
    
    if old_spinner1 in content:
        content = content.replace(old_spinner1, new_spinner1)
        changes_made = True
        print("  ✓ Replaced first AdwSpinner (spinner_overlay)")
    
    # Replace second AdwSpinner with GtkSpinner
    old_spinner2 = '''                                                    <object class="AdwSpinner">
                                                        <property name="width-request">64</property>'''
    new_spinner2 = '''                                                    <object class="GtkSpinner">
                                                        <property name="spinning">True</property>
                                                        <property name="width-request">64</property>'''
    
    if old_spinner2 in content:
        content = content.replace(old_spinner2, new_spinner2)
        changes_made = True
        print("  ✓ Replaced second AdwSpinner")
    
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    else:
        print("  ⚠ No changes needed (may already be patched)")
        return False


def apply_patch_ultralytics_remove_telemetry():
    """
    Patch: remove_ultralytics_telemetry.patch
    Remove Sentry telemetry and disable sync by default
    """
    print("\n[7/7] Applying ultralytics telemetry removal patch...")
    
    file_path = SITE_PACKAGES / "ultralytics" / "utils" / "__init__.py"
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    create_backup(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = False
    
    # 1. Simplify is_online function
    old_is_online = '''    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False'''
    
    new_is_online = "    return True"
    
    if old_is_online in content:
        content = content.replace(old_is_online, new_is_online)
        changes_made = True
        print("  ✓ Simplified is_online function")
    
    # 2. Remove set_sentry function
    import re
    pattern = r'def set_sentry\(\):.*?(?=\n(?:def |class |[A-Z_]+ =))'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        changes_made = True
        print("  ✓ Removed set_sentry function")
    
    # 3. Change sync default to False
    old_sync = '            "sync": True,  # Enable synchronization'
    new_sync = '            "sync": False,  # Disable synchronization'
    
    if old_sync in content:
        content = content.replace(old_sync, new_sync)
        changes_made = True
        print("  ✓ Changed sync default to False")
    
    # 4. Remove set_sentry() call
    old_sentry_call = "set_sentry()"
    if old_sentry_call in content:
        content = re.sub(r'^set_sentry\(\)\s*$', '', content, flags=re.MULTILINE)
        changes_made = True
        print("  ✓ Removed set_sentry() call")
    
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Patch applied successfully")
        return True
    else:
        print("  ⚠ No changes needed (may already be patched)")
        return False


def download_model_weights():
    """Download LADA model weights from Hugging Face."""
    print("\n" + "=" * 70)
    print("Downloading Model Weights")
    print("=" * 70)
    
    print(f"\nModel weights directory:")
    print(f"  {MODEL_WEIGHTS_DIR}")
    
    # Create model_weights directory if it doesn't exist
    if not MODEL_WEIGHTS_DIR.exists():
        MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created directory")
    else:
        print(f"\n✓ Directory exists")
    
    # Model definitions: (URL, output_filename)
    models = [
        (
            "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true",
            "lada_mosaic_detection_model_v4_accurate.pt"
        ),
        (
            "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true",
            "lada_mosaic_detection_model_v4_fast.pt"
        ),
        (
            "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true",
            "lada_mosaic_detection_model_v3.1_accurate.pt"
        ),
        (
            "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true",
            "lada_mosaic_detection_model_v3.1_fast.pt"
        ),
        (
            "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true",
            "lada_mosaic_restoration_model_generic_v1.2.pth"
        ),
    ]
    
    print("\nDownloading models...\n")
    
    results = []
    for i, (url, filename) in enumerate(models, 1):
        output_path = MODEL_WEIGHTS_DIR / filename
        
        # Check if file already exists
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[{i}/{len(models)}] {filename}")
            print(f"  ⚠ Already exists ({file_size:.1f} MB) - skipping")
            results.append((filename, True))
            continue
        
        print(f"[{i}/{len(models)}] Downloading {filename}...")
        
        # Try wget first, then curl
        try:
            result = subprocess.run(["wget", "--version"], capture_output=True, text=True)
            has_wget = result.returncode == 0
        except FileNotFoundError:
            has_wget = False
        
        if not has_wget:
            try:
                result = subprocess.run(["curl", "--version"], capture_output=True, text=True)
                has_curl = result.returncode == 0
            except FileNotFoundError:
                has_curl = False
        else:
            has_curl = False
        
        try:
            if has_wget:
                subprocess.run(
                    ["wget", "--progress=bar:force:noscroll", url, "-O", str(output_path)],
                    check=True
                )
                download_success = True
            elif has_curl:
                subprocess.run(
                    ["curl", "-L", "-#", url, "-o", str(output_path)],
                    check=True
                )
                download_success = True
            else:
                print(f"  ✗ Neither wget nor curl found. Please install one:")
                print(f"     sudo apt install wget  # Ubuntu/Debian")
                results.append((filename, False))
                continue
            
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Downloaded successfully ({file_size:.1f} MB)")
                results.append((filename, True))
            else:
                print(f"  ✗ Download failed - file not found")
                results.append((filename, False))
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Download failed: {e}")
            results.append((filename, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for filename, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {filename}")
    
    print(f"\nTotal: {success_count}/{total_count} models downloaded/verified")
    print(f"\nModels are located at: {MODEL_WEIGHTS_DIR}")
    print("=" * 70)
    
    return success_count == total_count


def main():
    """Apply all patches."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply patches to LADA environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect site-packages (recommended)
  python %(prog)s
  
  # Manually specify site-packages directory
  python %(prog)s --site-packages /path/to/site-packages
  
  # Skip model downloads
  python %(prog)s --skip-downloads
        """
    )
    
    parser.add_argument(
        '--site-packages',
        type=Path,
        help='Path to site-packages directory (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--skip-downloads',
        action='store_true',
        help='Skip downloading model weights'
    )
    
    args = parser.parse_args()
    
    # Determine site-packages directory
    global SITE_PACKAGES
    if args.site_packages:
        SITE_PACKAGES = args.site_packages
        print(f"Using manually specified site-packages: {SITE_PACKAGES}")
    elif SITE_PACKAGES is None:
        print("\n" + "=" * 70)
        print("ERROR: Could not auto-detect site-packages directory")
        print("=" * 70)
        print("\nPlease either:")
        print("  1. Activate your virtual environment before running this script:")
        print("     source .venv/bin/activate  # Linux/macOS")
        print("     .venv\\Scripts\\activate     # Windows")
        print()
        print("  2. Or specify the path manually:")
        print("     python apply_lada_patches.py --site-packages /path/to/site-packages")
        print()
        print("  Examples:")
        print("     # pyenv")
        print("     --site-packages ~/.pyenv/versions/3.13.9/envs/lada/lib/python3.13/site-packages")
        print()
        print("     # .venv (Linux/macOS)")
        print("     --site-packages .venv/lib/python3.13/site-packages")
        print()
        print("     # .venv (Windows)")
        print("     --site-packages .venv\\Lib\\site-packages")
        print("=" * 70)
        sys.exit(1)
    
    print("=" * 70)
    print("LADA Environment Patch Application Script")
    print("=" * 70)
    print(f"\nTarget directory: {SITE_PACKAGES}")
    
    if not SITE_PACKAGES.exists():
        print(f"\n✗ Error: Site-packages directory not found: {SITE_PACKAGES}")
        sys.exit(1)
    
    print("\nApplying patches...\n")
    
    # Apply all patches
    results = []
    results.append(("MMEngine Resume Dataloader", apply_patch_mmengine_resume_dataloader()))
    results.append(("MMEngine Torch 2.6+ Weights", apply_patch_mmengine_torch26_weights()))
    results.append(("Ultralytics MMS Time Limit", apply_patch_ultralytics_mms_time_limit()))
    results.append(("MMEngine Torch.distributed Compat", apply_patch_mmengine_torch_dist_compatibility()))
    results.append(("gvsbuild FFmpeg mp2float", apply_patch_gvsbuild_ffmpeg()))
    results.append(("LADA AdwSpinner to GtkSpinner", apply_patch_lada_adw_to_gtk_spinner()))
    results.append(("Ultralytics Telemetry Removal", apply_patch_ultralytics_remove_telemetry()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print(f"\nTotal: {success_count}/{total_count} patches applied successfully")
    
    if success_count == total_count:
        print("\n✓ All patches applied successfully!")
    elif success_count > 0:
        print("\n⚠ Some patches applied, some failed or were already applied")
    else:
        print("\n✗ No patches were applied")
    
    print("\nBackup files (.backup_*) have been created for all modified files.")
    print("=" * 70)
    
    # Download model weights
    download_success = True
    if not args.skip_downloads:
        print("\n")
        download_success = download_model_weights()
    else:
        print("\n" + "=" * 70)
        print("Skipping model downloads (--skip-downloads specified)")
        print("=" * 70)
    
    # Final status
    print("\n" + "=" * 70)
    print("Overall Status")
    print("=" * 70)
    print(f"Patches: {success_count}/{total_count} successful")
    if not args.skip_downloads:
        print(f"Models: {'All downloaded/verified' if download_success else 'Some downloads failed'}")
    else:
        print(f"Models: Skipped")
    print("=" * 70)


if __name__ == "__main__":
    main()
