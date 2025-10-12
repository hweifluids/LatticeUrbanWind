#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import os

def _split_eol(line: str):
    """Split and return the line content and its original EOL so that newline style is preserved."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""

def truncate_after_equal(file_path: Path):
    """
    Truncate everything after the first '=' on each line, but only within the editable window.
    """
    encodings = ["utf-8", "utf-8-sig", "gb18030"]
    last_err = None
    chosen_enc = None
    for enc in encodings:
        try:
            with file_path.open("r", encoding=enc, newline="") as f:
                lines = f.readlines()
            chosen_enc = enc
            break
        except UnicodeDecodeError as e:
            last_err = e
    if chosen_enc is None:
        raise last_err or UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode")

    # Locate the marker line index. Match by stripped content equality.
    marker_idx = None
    stripped_cache = []
    for i, raw in enumerate(lines):
        body, _eol = _split_eol(raw)
        s = body.strip()
        stripped_cache.append(s)
        if marker_idx is None and s == "// CFD control":
            marker_idx = i

    start_edit = 10  # zero-based index, so this is the 11th line
    end_edit = marker_idx if marker_idx is not None else len(lines)

    new_lines = []
    for i, raw in enumerate(lines):
        body, eol = _split_eol(raw)
        if start_edit <= i < end_edit:
            idx = body.find("=")
            if idx != -1:
                body = body[: idx + 1]
        new_lines.append(body + eol)

    with file_path.open("w", encoding=chosen_enc, newline="") as f:
        f.writelines(new_lines)

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

    # Step 1 edit file content within the allowed window
    try:
        truncate_after_equal(text_file)
    except Exception as e:
        print(f"Failed to modify file: {e}", file=sys.stderr)
        # Continue to cleanup

    # Step 2 cleanup all files under proj_temp recursively
    try:
        remove_files_in_proj_temp(text_file.parent)
    except Exception as e:
        print(f"Failed to clean proj_temp: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
