#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import os

def remove_files_in_proj_temp(parent_dir: Path):
    """
    Recursively delete all files under parent/proj_temp, including files in all subdirectories.
    Directories are kept intact. Symbolic links to files are removed as files. Symbolic links to
    directories are not followed.
    """
    target = parent_dir / "proj_temp"
    if not target.exists():
        return
    if not target.is_dir():
        raise NotADirectoryError(f"{target} is not a directory")

    for root, dirs, files in os.walk(target, topdown=True, followlinks=False):
        for name in files:
            p = Path(root) / name
            try:
                p.unlink()  # remove regular files and file symlinks
            except FileNotFoundError:
                # Concurrent deletion can lead to missing files, ignore safely
                pass
            except PermissionError as e:
                # Continue on permission issues but report for visibility
                print(f"Warning: failed to delete file: {p} ({e})", file=sys.stderr)

def main():
    if len(sys.argv) != 2:
        print("Usage: python cleanluw.py <text file path>", file=sys.stderr)
        sys.exit(1)

    text_file = Path(sys.argv[1]).resolve()
    if not text_file.is_file():
        print(f"Error: {text_file} is not a valid file", file=sys.stderr)
        sys.exit(1)

    # cleanluw only removes temporary artefacts; it no longer mutates the deck itself.
    try:
        remove_files_in_proj_temp(text_file.parent)
    except Exception as e:
        print(f"Failed to clean proj_temp: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
