#!/usr/bin/env python3
"""
Build and package SpectraQC for GitHub release.

This script:
1. Builds the standalone executable using PyInstaller
2. Packages it with documentation into a release zip file
3. Creates a release-ready archive for GitHub

Usage:
    python scripts/build_release.py [--skip-build] [--platform PLATFORM]

Options:
    --skip-build    Skip the executable build step (use existing dist/)
    --platform      Override platform name (default: auto-detect)
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def get_version() -> str:
    """Extract version from spectraqc/version.py."""
    version_file = get_project_root() / "spectraqc" / "version.py"
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                # Parse: __version__ = '1.0.0'
                return line.split("=")[1].strip().strip("'\"")
    raise RuntimeError("Could not find __version__ in version.py")


def get_platform_tag() -> str:
    """Get platform tag for release filename."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "windows-x64"
        return f"windows-{machine}"
    elif system == "darwin":
        if machine == "arm64":
            return "macos-arm64"
        return "macos-x64"
    elif system == "linux":
        if machine in ("amd64", "x86_64"):
            return "linux-x64"
        return f"linux-{machine}"
    return f"{system}-{machine}"


def build_executable(project_root: Path) -> Path:
    """Build the executable using build_exe.py."""
    print("Building executable...")
    build_script = project_root / "scripts" / "build_exe.py"
    
    subprocess.check_call(
        [sys.executable, str(build_script), "--clean"],
        cwd=project_root,
    )
    
    # Find the executable
    dist_dir = project_root / "dist"
    if sys.platform == "win32":
        exe_name = "spectraqc.exe"
    else:
        exe_name = "spectraqc"
    
    # Check single-file location first
    exe_path = dist_dir / exe_name
    if exe_path.exists():
        return exe_path
    
    # Check one-folder location
    exe_path = dist_dir / "spectraqc" / exe_name
    if exe_path.exists():
        return exe_path
    
    raise RuntimeError(f"Executable not found in {dist_dir}")


def copy_docs(project_root: Path, release_dir: Path) -> None:
    """Copy documentation files, excluding marketing subfolder."""
    docs_src = project_root / "docs"
    docs_dst = release_dir / "docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    
    for item in docs_src.iterdir():
        if item.name == "marketing":
            # Skip marketing subfolder
            continue
        if item.is_file():
            shutil.copy2(item, docs_dst / item.name)
        elif item.is_dir():
            shutil.copytree(item, docs_dst / item.name)
    
    # Copy RELEASE_README.md as README.md for the distribution
    release_readme = project_root / "RELEASE_README.md"
    if release_readme.exists():
        shutil.copy2(release_readme, release_dir / "README.md")
    else:
        # Fall back to main README if release version doesn't exist
        readme_src = project_root / "README.md"
        if readme_src.exists():
            shutil.copy2(readme_src, release_dir / "README.md")


def create_release_zip(
    project_root: Path,
    exe_path: Path,
    version: str,
    platform_tag: str,
) -> Path:
    """Create the release zip file."""
    release_name = f"spectraqc-{version}-{platform_tag}"
    release_dir = project_root / "release" / release_name
    
    # Clean and create release directory
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)
    
    # Copy executable
    if exe_path.parent.name == "spectraqc":
        # One-folder bundle - copy entire folder contents
        print("Packaging one-folder bundle...")
        for item in exe_path.parent.iterdir():
            if item.is_file():
                shutil.copy2(item, release_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, release_dir / item.name)
    else:
        # Single-file executable
        print("Packaging single-file executable...")
        shutil.copy2(exe_path, release_dir / exe_path.name)
    
    # Copy documentation
    print("Copying documentation...")
    copy_docs(project_root, release_dir)
    
    # Create zip file
    zip_path = project_root / "release" / f"{release_name}.zip"
    print(f"Creating {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in release_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(release_dir.parent)
                zf.write(file_path, arcname)
    
    return zip_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and package SpectraQC for GitHub release"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip executable build step (use existing dist/)",
    )
    parser.add_argument(
        "--platform",
        help="Override platform tag (default: auto-detect)",
    )
    
    args = parser.parse_args()
    project_root = get_project_root()
    
    print("=" * 60)
    print("SpectraQC Release Builder")
    print("=" * 60)
    
    # Get version and platform
    version = get_version()
    platform_tag = args.platform or get_platform_tag()
    
    print(f"Version: {version}")
    print(f"Platform: {platform_tag}")
    print(f"Project root: {project_root}")
    print("-" * 60)
    
    # Build executable
    if args.skip_build:
        print("Skipping build (--skip-build specified)")
        dist_dir = project_root / "dist"
        exe_name = "spectraqc.exe" if sys.platform == "win32" else "spectraqc"
        exe_path = dist_dir / exe_name
        if not exe_path.exists():
            exe_path = dist_dir / "spectraqc" / exe_name
        if not exe_path.exists():
            print(f"ERROR: No executable found in {dist_dir}")
            return 1
    else:
        try:
            exe_path = build_executable(project_root)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Build failed with exit code {e.returncode}")
            return 1
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
    
    print(f"Executable: {exe_path}")
    print(f"Size: {exe_path.stat().st_size / (1024 * 1024):.1f} MB")
    print("-" * 60)
    
    # Create release package
    try:
        zip_path = create_release_zip(project_root, exe_path, version, platform_tag)
    except Exception as e:
        print(f"ERROR: Failed to create release package: {e}")
        return 1
    
    print("-" * 60)
    print("✓ Release package created successfully!")
    print(f"  Archive: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / (1024 * 1024):.1f} MB")
    print()
    print("Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist())[:20]:
            print(f"  {name}")
        if len(zf.namelist()) > 20:
            print(f"  ... and {len(zf.namelist()) - 20} more files")
    
    print()
    print("To upload to GitHub Releases:")
    print(f"  1. Go to https://github.com/<owner>/spectraqc/releases/new")
    print(f"  2. Create tag: v{version}")
    print(f"  3. Upload: {zip_path.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
