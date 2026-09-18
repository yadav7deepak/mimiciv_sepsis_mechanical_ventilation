"""
Vasopressors Data Preparation Script (converted from vasopressors_6files.ipynb).
Extracts weight durations and dose/rate information for:
- Dopamine
- Epinephrine
- Norepinephrine
- Phenylephrine
- Vasopressin

Reads raw inputevents, chartevents, and icustays from Google Drive (MIMIC_PATH).
Saves all outputs locally to LOCAL_DATA_DIR (./data).
"""

import os
import sys
import argparse
from datetime import timedelta
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


def create_weight_durations(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Creates a dataframe with weights for ICU patients with start and end times.
    """
    out_file = os.path.join(output_path, "vasopressor_weights.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file, parse_dates=['starttime', 'endtime'])

    drive_file = resolve_intermediate_path("vasopressor_weights.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[vasopressors] Loading pre-existing vasopressor weights from Drive: {drive_file}")
        df = pd.read_csv(drive_file, parse_dates=['starttime', 'endtime'])
        df.to_csv(out_file, index=False)
        return df

    local_weight = resolve_intermediate_path("patient_weight.csv")
    if not force and os.path.exists(local_weight):
        print("\n[vasopressors] Building weight durations from existing weight data and icustays...")
        weights = pd.read_csv(local_weight)
        icustays = pd.read_csv(
            os.path.join(mimic_path, "icu/icustays.csv.gz"),
            usecols=['stay_id', 'intime', 'outtime'],
            parse_dates=['intime', 'outtime']
        )
        wt = weights.merge(icustays, on='stay_id', how='inner')
        wt['starttime'] = wt['intime'] - timedelta(hours=2)
        wt['endtime'] = wt['outtime'] + timedelta(hours=2)
        wt_dur = wt[['stay_id', 'starttime', 'endtime', 'weight']].rename(columns={'stay_id': 'icustay_id'})
        wt_dur.to_csv(out_file, index=False)
        print(f"[vasopressors] Vasopressor weights saved to: {out_file}")
        return wt_dur

    print("\n[vasopressors] Creating weight durations dataframe...")
    weight_itemids = [762, 226512, 763, 226531]
    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[
            (chunk['itemid'].isin(weight_itemids)) &
            (chunk['valuenum'].notna()) &
            (chunk['valuenum'] != 0)
        ]
        if not filtered.empty:
            chunks.append(filtered)

    wt_stg = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['stay_id', 'charttime', 'itemid', 'valuenum']
    )
    wt_stg['weight_type'] = wt_stg['itemid'].apply(lambda x: 'admit' if x in [762, 226512] else 'daily')
    wt_stg = wt_stg[['stay_id', 'charttime', 'weight_type', 'valuenum']]
    wt_stg.rename(columns={'valuenum': 'weight', 'stay_id': 'icustay_id'}, inplace=True)

    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['stay_id', 'intime', 'outtime'],
        parse_dates=['intime', 'outtime']
    )
    icustays.rename(columns={'stay_id': 'icustay_id'}, inplace=True)

    wt_stg['rn'] = wt_stg.groupby(['icustay_id', 'weight_type']).cumcount() + 1
    wt_stg2 = pd.merge(wt_stg, icustays, on='icustay_id', how='inner')

    wt_stg2['starttime'] = wt_stg2.apply(
        lambda row: row['intime'] + timedelta(hours=2) if row['weight_type'] == 'admit' and row['rn'] == 1 else row['charttime'],
        axis=1
    )
    wt_stg2 = wt_stg2[~((wt_stg2['weight_type'] == 'admit') & (wt_stg2['rn'] == 1))]

    wt_stg2 = wt_stg2.sort_values(['icustay_id', 'starttime'])
    wt_stg2['endtime'] = wt_stg2.groupby('icustay_id')['starttime'].shift(-1)
    wt_stg2['endtime'] = wt_stg2['endtime'].fillna(wt_stg2['outtime'] + timedelta(hours=2))

    wt_stg2 = wt_stg2[['icustay_id', 'starttime', 'endtime', 'weight']]

    wt1 = pd.merge(icustays[['icustay_id', 'intime', 'outtime']], wt_stg2, on='icustay_id', how='left')
    wt1_grouped = wt1.sort_values(['icustay_id', 'starttime'])
    wt1_grouped['next_starttime'] = wt1_grouped.groupby('icustay_id')['starttime'].shift(-1)
    wt1_grouped['endtime'] = wt1_grouped.apply(
        lambda row: row['next_starttime'] if pd.notna(row['next_starttime']) else row['outtime'] + timedelta(hours=2),
        axis=1
    )

    first_weights = wt1_grouped.sort_values(['icustay_id', 'starttime']).groupby('icustay_id').first().reset_index()
    first_weights = first_weights[['icustay_id', 'starttime', 'weight']]

    wt_fix = pd.merge(icustays[['icustay_id', 'intime']], first_weights, on='icustay_id', how='inner')
    wt_fix = wt_fix[wt_fix['intime'] < wt_fix['starttime']].copy()

    if not wt_fix.empty:
        wt_fix['endtime'] = wt_fix['starttime']
        wt_fix['starttime'] = wt_fix['intime'] - timedelta(hours=2)
        wt_fix = wt_fix[['icustay_id', 'starttime', 'endtime', 'weight']]
        wt2 = pd.concat([wt1_grouped[['icustay_id', 'starttime', 'endtime', 'weight']], wt_fix], ignore_index=True)
    else:
        wt2 = wt1_grouped[['icustay_id', 'starttime', 'endtime', 'weight']]

    weightdurations = wt2.sort_values(['icustay_id', 'starttime', 'endtime'])
    out_file = os.path.join(output_path, "vasopressor_weights.csv")
    weightdurations.to_csv(out_file, index=False)
    print(f"[vasopressors] Vasopressor weights saved to: {out_file}")
    return weightdurations


def create_dopamine_dose(mimic_path: str, output_path: str, weightdurations=None, force: bool = False) -> pd.DataFrame:
    """Creates dataframe with dopamine dose and duration into vasopressor_dopamine.csv."""
    out_file = os.path.join(output_path, "vasopressor_dopamine.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    print("\n[vasopressors] Creating dopamine dose dataframe...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'rate', 'amount', 'orderid', 'statusdescription'],
        parse_dates=['starttime', 'endtime']
    )
    inputevents.rename(columns={'stay_id': 'icustay_id', 'orderid': 'linkorderid'}, inplace=True)

    dopamine_itemid = 221662
    dopamine_mv = inputevents[
        (inputevents['itemid'] == dopamine_itemid) &
        (inputevents['statusdescription'] != 'Rewritten')
    ].copy()

    dopamine_mv_grouped = dopamine_mv.groupby(['icustay_id', 'linkorderid']).agg({
        'rate': 'max',
        'amount': 'sum',
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()

    dopamine_mv_grouped.rename(columns={'rate': 'vaso_rate', 'amount': 'vaso_amount'}, inplace=True)
    dopamine_dose = dopamine_mv_grouped.sort_values(['icustay_id', 'starttime'])

    dopamine_dose.to_csv(out_file, index=False)
    print(f"[vasopressors] Dopamine dose saved to: {out_file}")
    return dopamine_dose


def create_epinephrine_dose(mimic_path: str, output_path: str, weightdurations=None, force: bool = False) -> pd.DataFrame:
    """Creates dataframe with epinephrine dose and duration into vasopressor_epinephrine.csv."""
    out_file = os.path.join(output_path, "vasopressor_epinephrine.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    print("\n[vasopressors] Creating epinephrine dose dataframe...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'rate', 'amount', 'orderid', 'statusdescription'],
        parse_dates=['starttime', 'endtime']
    )
    inputevents.rename(columns={'stay_id': 'icustay_id', 'orderid': 'linkorderid'}, inplace=True)

    if weightdurations is None:
        wt_file = resolve_intermediate_path("vasopressor_weights.csv")
        if os.path.exists(wt_file):
            weightdurations = pd.read_csv(wt_file, parse_dates=['starttime', 'endtime'])

    epinephrine_itemid = 221289
    epinephrine_mv = inputevents[
        (inputevents['itemid'] == epinephrine_itemid) &
        (inputevents['statusdescription'] != 'Rewritten')
    ].copy()

    if weightdurations is not None and 'weight' in weightdurations.columns:
        wt_map = weightdurations.drop_duplicates('icustay_id').set_index('icustay_id')['weight'].to_dict()
        epinephrine_mv['weight'] = epinephrine_mv['icustay_id'].map(wt_map).fillna(80.0)
    else:
        epinephrine_mv['weight'] = 80.0

    epinephrine_mv['vaso_rate'] = epinephrine_mv['rate']

    epinephrine_mv_grouped = epinephrine_mv.groupby(['icustay_id', 'linkorderid']).agg({
        'vaso_rate': 'max',
        'amount': 'sum',
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()

    epinephrine_mv_grouped.rename(columns={'amount': 'vaso_amount'}, inplace=True)
    epinephrine_dose = epinephrine_mv_grouped.sort_values(['icustay_id', 'starttime'])

    epinephrine_dose.to_csv(out_file, index=False)
    print(f"[vasopressors] Epinephrine dose saved to: {out_file}")
    return epinephrine_dose


def create_norepinephrine_dose(mimic_path: str, output_path: str, weightdurations=None, force: bool = False) -> pd.DataFrame:
    """Creates dataframe with norepinephrine dose and duration into vasopressor_norepinephrine.csv."""
    out_file = os.path.join(output_path, "vasopressor_norepinephrine.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    print("\n[vasopressors] Creating norepinephrine dose dataframe...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'rate', 'amount', 'orderid', 'statusdescription'],
        parse_dates=['starttime', 'endtime']
    )
    inputevents.rename(columns={'stay_id': 'icustay_id', 'orderid': 'linkorderid'}, inplace=True)

    if weightdurations is None:
        wt_file = resolve_intermediate_path("vasopressor_weights.csv")
        if os.path.exists(wt_file):
            weightdurations = pd.read_csv(wt_file, parse_dates=['starttime', 'endtime'])

    norepinephrine_itemid = 221906
    norepinephrine_mv = inputevents[
        (inputevents['itemid'] == norepinephrine_itemid) &
        (inputevents['statusdescription'] != 'Rewritten')
    ].copy()

    if weightdurations is not None and 'weight' in weightdurations.columns:
        wt_map = weightdurations.drop_duplicates('icustay_id').set_index('icustay_id')['weight'].to_dict()
        norepinephrine_mv['weight'] = norepinephrine_mv['icustay_id'].map(wt_map).fillna(80.0)
    else:
        norepinephrine_mv['weight'] = 80.0

    norepinephrine_mv['vaso_rate'] = norepinephrine_mv['rate']

    norepinephrine_mv_grouped = norepinephrine_mv.groupby(['icustay_id', 'linkorderid']).agg({
        'vaso_rate': 'max',
        'amount': 'sum',
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()

    norepinephrine_mv_grouped.rename(columns={'amount': 'vaso_amount'}, inplace=True)
    norepinephrine_dose = norepinephrine_mv_grouped.sort_values(['icustay_id', 'starttime'])

    norepinephrine_dose.to_csv(out_file, index=False)
    print(f"[vasopressors] Norepinephrine dose saved to: {out_file}")
    return norepinephrine_dose


def create_phenylephrine_dose(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """Creates dataframe with phenylephrine dose and duration into vasopressor_phenylephrine.csv."""
    out_file = os.path.join(output_path, "vasopressor_phenylephrine.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    print("\n[vasopressors] Creating phenylephrine dose dataframe...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'rate', 'amount', 'orderid', 'statusdescription'],
        parse_dates=['starttime', 'endtime']
    )
    inputevents.rename(columns={'stay_id': 'icustay_id', 'orderid': 'linkorderid'}, inplace=True)

    phenylephrine_itemid = 221749
    phenylephrine_mv = inputevents[
        (inputevents['itemid'] == phenylephrine_itemid) &
        (inputevents['statusdescription'] != 'Rewritten')
    ].copy()

    phenylephrine_mv_grouped = phenylephrine_mv.groupby(['icustay_id', 'linkorderid']).agg({
        'rate': 'max',
        'amount': 'sum',
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()

    phenylephrine_mv_grouped.rename(columns={'rate': 'vaso_rate', 'amount': 'vaso_amount'}, inplace=True)
    phenylephrine_dose = phenylephrine_mv_grouped.sort_values(['icustay_id', 'starttime'])

    phenylephrine_dose.to_csv(out_file, index=False)
    print(f"[vasopressors] Phenylephrine dose saved to: {out_file}")
    return phenylephrine_dose


def create_vasopressin_dose(mimic_path: str, output_path: str, weightdurations=None, force: bool = False) -> pd.DataFrame:
    """Creates dataframe with vasopressin dose and duration into vasopressor_vasopressin.csv."""
    out_file = os.path.join(output_path, "vasopressor_vasopressin.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[vasopressors] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    print("\n[vasopressors] Creating vasopressin dose dataframe...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'rate', 'amount', 'orderid', 'statusdescription'],
        parse_dates=['starttime', 'endtime']
    )
    inputevents.rename(columns={'stay_id': 'icustay_id', 'orderid': 'linkorderid'}, inplace=True)

    if weightdurations is None:
        wt_file = resolve_intermediate_path("vasopressor_weights.csv")
        if os.path.exists(wt_file):
            weightdurations = pd.read_csv(wt_file, parse_dates=['starttime', 'endtime'])

    vasopressin_itemid = 222315
    vasopressin_mv = inputevents[
        (inputevents['itemid'] == vasopressin_itemid) &
        (inputevents['statusdescription'] != 'Rewritten')
    ].copy()

    if weightdurations is not None and 'weight' in weightdurations.columns:
        wt_map = weightdurations.drop_duplicates('icustay_id').set_index('icustay_id')['weight'].to_dict()
        vasopressin_mv['weight'] = vasopressin_mv['icustay_id'].map(wt_map).fillna(80.0)
    else:
        vasopressin_mv['weight'] = 80.0

    vasopressin_mv['vaso_rate'] = vasopressin_mv['rate']

    vasopressin_mv_grouped = vasopressin_mv.groupby(['icustay_id', 'linkorderid']).agg({
        'vaso_rate': 'max',
        'amount': 'sum',
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()

    vasopressin_mv_grouped.rename(columns={'amount': 'vaso_amount'}, inplace=True)
    vasopressin_dose = vasopressin_mv_grouped.sort_values(['icustay_id', 'starttime'])

    vasopressin_dose.to_csv(out_file, index=False)
    print(f"[vasopressors] Vasopressin dose saved to: {out_file}")
    return vasopressin_dose


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR), force: bool = False):
    """Runs all vasopressor calculations in dependency order."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting vasopressors extraction:\n  Drive Source: {mimic_path}\n  Local Output: {output_path}\n  Force Recompute: {force}")
    wt = create_weight_durations(mimic_path, output_path, force=force)
    create_dopamine_dose(mimic_path, output_path, weightdurations=wt, force=force)
    create_epinephrine_dose(mimic_path, output_path, weightdurations=wt, force=force)
    create_norepinephrine_dose(mimic_path, output_path, weightdurations=wt, force=force)
    create_phenylephrine_dose(mimic_path, output_path, force=force)
    create_vasopressin_dose(mimic_path, output_path, weightdurations=wt, force=force)
    print("\n[vasopressors] All vasopressor processing finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Vasopressors Data")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all tables")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, force=args.force)
