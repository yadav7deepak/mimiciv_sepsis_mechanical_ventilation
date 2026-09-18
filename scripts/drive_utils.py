"""
Google Drive mounting and verification utilities for MIMIC-IV pipeline.
Supports automated mounting in Google Colab, mount-path checks on local machines,
and credential validation.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from .config import (
        GDRIVE_EMAIL,
        GDRIVE_PASSWORD,
        GDRIVE_MOUNT_PATH,
        MIMIC_PATH,
        print_config_summary,
    )
except ImportError:
    from config import (
        GDRIVE_EMAIL,
        GDRIVE_PASSWORD,
        GDRIVE_MOUNT_PATH,
        MIMIC_PATH,
        print_config_summary,
    )


def is_colab() -> bool:
    """Checks if current execution environment is Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def mount_drive(mount_point: Optional[str] = None) -> bool:
    """
    Mounts Google Drive.
    - If in Google Colab, mounts Google Drive to mount_point (default /content/drive).
    - If on local machine, checks if the mount directory exists.
    """
    target_mount = mount_point or GDRIVE_MOUNT_PATH or "/content/drive"

    if is_colab():
        try:
            print(f"[Drive] Detected Google Colab environment. Mounting drive at '{target_mount}'...")
            from google.colab import drive
            drive.mount(target_mount)
            print("[Drive] Successfully mounted Google Drive.")
            return True
        except Exception as e:
            print(f"[Drive] Error mounting drive in Colab: {e}")
            return False

    # Non-Colab environment (e.g. macOS / Linux / Windows)
    if os.path.exists(target_mount):
        print(f"[Drive] Google Drive mount path exists: '{target_mount}'")
        return True

    print(f"[Drive] Notice: Mount directory '{target_mount}' is not found locally.")
    if GDRIVE_EMAIL:
        print(f"[Drive] Configured account: {GDRIVE_EMAIL}")
    print("[Drive] If running locally, please ensure Google Drive is mounted or sync folder is active.")
    print("        On macOS: Install Google Drive for Desktop (/Volumes/GoogleDrive/My Drive)")
    print("        On Linux: Use rclone or google-drive-ocamlfuse to mount.")
    return False


def verify_mimic_path(path: Optional[str] = None, raise_error: bool = False) -> bool:
    """
    Checks if the MIMIC-IV directory on Google Drive is accessible.
    """
    target_path = path or MIMIC_PATH
    if os.path.isdir(target_path):
        print(f"[Drive] MIMIC-IV dataset path verified: '{target_path}'")
        return True

    msg = (
        f"[Drive Error] MIMIC-IV path not found on Drive: '{target_path}'\n"
        f"Please check your .env configuration:\n"
        f"  1. Ensure Google Drive is mounted at GDRIVE_MOUNT_PATH ({GDRIVE_MOUNT_PATH})\n"
        f"  2. Verify MIMIC_PATH in .env points to the correct folder on Google Drive.\n"
        f"  3. Credentials email configured: {GDRIVE_EMAIL or '(none)'}\n"
    )
    if raise_error:
        raise FileNotFoundError(msg)
    else:
        print(msg)
        return False


def verify_tables_exist(table_names: List[str], subfolder: str = "icu", raise_error: bool = False) -> bool:
    """
    Verifies that specific MIMIC-IV raw table files exist in the Google Drive path.
    """
    missing = []
    for tbl in table_names:
        full_path = os.path.join(MIMIC_PATH, subfolder, tbl)
        if not os.path.exists(full_path):
            # Check without .gz or with .gz
            alt_path = full_path[:-3] if full_path.endswith(".gz") else f"{full_path}.gz"
            if not os.path.exists(alt_path):
                missing.append(full_path)

    if missing:
        msg = f"[Drive Warning] Missing {len(missing)} required raw table(s) on Google Drive:\n" + "\n".join(f"  - {m}" for m in missing)
        if raise_error:
            raise FileNotFoundError(msg)
        else:
            print(msg)
            return False
    return True


if __name__ == "__main__":
    print_config_summary()
    mount_drive()
    verify_mimic_path()
