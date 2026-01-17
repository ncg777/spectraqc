#!/usr/bin/env python3
"""
Build script for creating SpectraQC standalone executable.

This script automates the process of building a standalone executable
using PyInstaller. It can be run directly or invoked via the CLI.

Usage:
    python scripts/build_exe.py [--onedir] [--clean] [--debug]

Options:
    --onedir    Create a one-folder bundle instead of a single executable
    --clean     Clean build artifacts before building
    --debug     Enable debug mode for troubleshooting
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def check_pyinstaller() -> bool:
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def install_pyinstaller() -> None:
    """Install PyInstaller."""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def clean_build_artifacts(project_root: Path) -> None:
    """Clean previous build artifacts."""
    dirs_to_clean = ["build", "dist"]
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)


def build_executable(
    project_root: Path,
    onedir: bool = False,
    debug: bool = False,
) -> Path:
    """
    Build the standalone executable.
    
    Args:
        project_root: Path to the project root directory
        onedir: If True, create a one-folder bundle; otherwise single file
        debug: If True, enable debug mode
        
    Returns:
        Path to the built executable
    """
    spec_file = project_root / "spectraqc.spec"
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
    ]
    
    if debug:
        cmd.append("--log-level=DEBUG")
    
    if onedir:
        # Override the spec file for onedir build
        cmd.extend([
            "--name=spectraqc",
            "--onedir",
            "--console",
            "--collect-submodules=spectraqc",
            "--collect-data=soundfile",
            "--hidden-import=numpy",
            "--hidden-import=yaml",
            "--hidden-import=soundfile",
            str(project_root / "spectraqc" / "cli" / "main.py"),
        ])
    else:
        # Build single-file executable (ignore existing spec, use explicit options)
        cmd.extend([
            "--name=spectraqc",
            "--onefile",
            "--console",
            "--collect-submodules=spectraqc",
            "--collect-data=soundfile",
            "--hidden-import=numpy",
            "--hidden-import=yaml",
            "--hidden-import=soundfile",
            str(project_root / "spectraqc" / "cli" / "main.py"),
        ])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=project_root)
    
    # Determine output path
    dist_dir = project_root / "dist"
    exe_name = "spectraqc.exe" if sys.platform == "win32" else "spectraqc"
    
    if onedir:
        exe_path = dist_dir / "spectraqc" / exe_name
    else:
        exe_path = dist_dir / exe_name
    
    # Fallback: check if it ended up in a folder anyway
    if not exe_path.exists():
        alt_path = dist_dir / "spectraqc" / exe_name
        if alt_path.exists():
            exe_path = alt_path
    
    return exe_path


def verify_executable(exe_path: Path) -> bool:
    """Verify the built executable works."""
    if not exe_path.exists():
        print(f"ERROR: Executable not found at {exe_path}")
        return False
    
    print(f"\nVerifying executable at {exe_path}...")
    try:
        result = subprocess.run(
            [str(exe_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"Output: {result.stdout.strip()}")
        if result.returncode == 0:
            print("✓ Executable verified successfully!")
            return True
        else:
            print(f"ERROR: Executable returned non-zero exit code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: Executable timed out")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run executable: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build SpectraQC standalone executable"
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create a one-folder bundle instead of a single executable",
    )
    parser.add_argument(
        "--clean",
        action="store_true", 
        help="Clean build artifacts before building",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for troubleshooting",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip executable verification after build",
    )
    
    args = parser.parse_args()
    project_root = get_project_root()
    
    print(f"SpectraQC Executable Builder")
    print(f"Project root: {project_root}")
    print("-" * 50)
    
    # Check/install PyInstaller
    if not check_pyinstaller():
        install_pyinstaller()
    
    # Clean if requested
    if args.clean:
        clean_build_artifacts(project_root)
    
    # Build
    try:
        exe_path = build_executable(
            project_root,
            onedir=args.onedir,
            debug=args.debug,
        )
        print(f"\n✓ Build complete!")
        print(f"  Executable: {exe_path}")
        
        # Get file size
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        
        # Verify
        if not args.skip_verify:
            if not verify_executable(exe_path):
                return 1
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Build failed with exit code {e.returncode}")
        return 1
    except Exception as e:
        print(f"\nERROR: Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
