"""
Echo Notes Feature Extraction via Regular Expressions (converted from data_controls.ipynb).
Extracts clinical measurements from echocardiogram reports in MIMIC-IV note events:
- Indication, Height, Weight, Body Surface Area (BSA)
- Blood Pressure (BPSys, BPDias), Heart Rate (HR)
- Status, Test type, Doppler findings, Contrast, Technical Quality

Reads raw note files from Google Drive (MIMIC_PATH/mimic-iv-note/2.2/note).
Saves output locally to LOCAL_DATA_DIR/echo_features_regex.csv.
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np

# Import configuration and drive utilities
try:
    from .config import (
        MIMIC_PATH,
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from .drive_utils import mount_drive, verify_mimic_path
except ImportError:
    from config import (
        MIMIC_PATH,
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from drive_utils import mount_drive, verify_mimic_path


def extract_echo_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive regex patterns to parse structured clinical fields from echo text.
    """
    df = df.copy()

    # Indication
    df['Indication'] = df['text'].str.extract(r'Indication:\s*(.*?)\n', flags=re.IGNORECASE)[0]

    # Height
    height1 = df["text"].str.extract(r"Height:\s*\(in\)\s*([^\n\*]*)\n", flags=re.IGNORECASE)[0]
    height2 = df["text"].str.extract(r"height\s*[:=]?\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
    df["Height"] = pd.to_numeric(height1, errors='coerce')
    missing_h = df["Height"].isnull()
    df.loc[missing_h, "Height"] = pd.to_numeric(height2[missing_h], errors='coerce')
    df.loc[df["Height"].astype(str).str.contains(r'\*', na=False), "Height"] = None

    # Weight
    weight1 = df["text"].str.extract(r"Weight\s*\(lb\):\s*([^\n\*]*)\n", flags=re.IGNORECASE)[0]
    weight2 = df["text"].str.extract(r"weight\s*[:=]\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
    df["Weight"] = pd.to_numeric(weight1, errors='coerce')
    missing_w = df["Weight"].isnull()
    df.loc[missing_w, "Weight"] = pd.to_numeric(weight2[missing_w], errors='coerce')
    df.loc[df["Weight"].astype(str).str.contains(r'\*', na=False), "Weight"] = None

    # BSA
    bsa1 = df["text"].str.extract(r"BSA\s*\(m2\):\s*([^\s\*]+)", flags=re.IGNORECASE)[0]
    bsa2 = df["text"].str.extract(r"BSA\s*\(m2\)\s*:?=?\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
    df["BSA"] = pd.to_numeric(bsa1, errors='coerce')
    missing_bsa = df["BSA"].isnull()
    df.loc[missing_bsa, "BSA"] = pd.to_numeric(bsa2[missing_bsa], errors='coerce')
    df.loc[df["BSA"].astype(str).str.contains(r'\*', na=False), "BSA"] = None

    # Blood Pressure (Systolic and Diastolic)
    df["BP"] = df["text"].str.extract(r'BP\s*\(mm Hg\):\s*([^\n]*)', flags=re.IGNORECASE)[0]
    systolic = df["text"].str.extract(r'BP\s*\(mm Hg\):\s*([0-9]+)\s*/\s*[0-9]+', flags=re.IGNORECASE)[0]
    diastolic = df["text"].str.extract(r'BP\s*\(mm Hg\):\s*[0-9]+\s*/\s*([0-9]+)', flags=re.IGNORECASE)[0]
    df["BPSys"] = pd.to_numeric(systolic, errors='coerce')
    df["BPDias"] = pd.to_numeric(diastolic, errors='coerce')

    # Heart Rate
    hr1 = df["text"].str.extract(r'HR\s*\(bpm\):\s*([^\n\*]*)\n', flags=re.IGNORECASE)[0]
    df["HR"] = pd.to_numeric(hr1, errors='coerce')
    hr2 = df["text"].str.extract(r'HR\s*\(bpm\)?\s*[:=]?\s*([0-9]+\.?\d*)', flags=re.IGNORECASE)[0]
    missing_hr = df["HR"].isnull()
    df.loc[missing_hr, "HR"] = pd.to_numeric(hr2[missing_hr], errors='coerce')
    df.loc[df["HR"].astype(str).str.contains(r'\*', na=False), "HR"] = None

    # Categorical / descriptive fields
    for field, key in [
        ("Status", "Status"), ("Test", "Test"), ("Doppler", "Doppler"),
        ("Contrast", "Contrast"), ("TechnicalQuality", "Technical Quality")
    ]:
        df[field] = df["text"].str.extract(rf"{key}:\s*(.*?)\n", flags=re.IGNORECASE)[0]

    return df


def load_echo_notes(mimic_path: str) -> pd.DataFrame:
    """Loads discharge and radiology notes from Drive and filters for Echo records."""
    note_dir = os.path.join(mimic_path, "mimic-iv-note/2.2/note")
    print(f"\n[data_controls] Loading notes from Google Drive: {note_dir}")

    dfs = []
    for note_file in ["discharge.csv.gz", "discharge_detail.csv.gz", "radiology.csv.gz", "radiology_detail.csv.gz"]:
        fpath = os.path.join(note_dir, note_file)
        if os.path.exists(fpath):
            print(f"  Reading {note_file}...")
            dfs.append(pd.read_csv(fpath))

    if not dfs:
        raise FileNotFoundError(f"No note files found in {note_dir}")

    noteevents = pd.concat(dfs, ignore_index=True)
    print(f"Total notes loaded: {len(noteevents):,}")

    echo_notes = noteevents[noteevents['text'].str.contains('Echo', case=False, na=False)].copy()
    print(f"Echo notes identified: {len(echo_notes):,}")
    return echo_notes


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR)):
    """Runs regex-based echo feature extraction."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    echo_notes = load_echo_notes(mimic_path)
    df_extracted = extract_echo_fields(echo_notes)

    out_file = os.path.join(output_path, "echo_features_regex.csv")
    df_extracted.to_csv(out_file, index=False)
    print(f"[data_controls] Extracted echo features saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Echo Report Parameters via Regex")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir)
