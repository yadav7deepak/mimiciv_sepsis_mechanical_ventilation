"""
Lab and Fluid Data Preparation Script (converted from lab_and_fluid).
Extracts lab values, cumulative fluid balance, intravenous fluids,
and urine output for ICU patients from MIMIC-IV.

Reads raw tables from Google Drive (MIMIC_PATH).
Saves all processed outputs locally to LOCAL_DATA_DIR (./data).
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


def create_lab_values(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Retrieves lab values from LABEVENTS table and filters to ICU encounter windows.
    """
    out_file = os.path.join(output_path, "lab_values.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000000:
        print(f"\n[lab_and_fluid] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("lab_values.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[lab_and_fluid] Loading pre-existing lab_values from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[lab_and_fluid] Extracting lab values from labevents table...")
    lab_items = {
        'ALBUMIN': {'itemid': [50862], 'min': 0, 'max': 10},
        'ANIONGAP': {'itemid': [50868], 'min': 0, 'max': 10000},
        'BANDS': {'itemid': [51144], 'min': 0, 'max': 100},
        'BICARBONATE': {'itemid': [50882], 'min': 0, 'max': 10000},
        'BILIRUBIN': {'itemid': [50885], 'min': 0, 'max': 150},
        'CHLORIDE': {'itemid': [50806, 50902], 'min': 0, 'max': 10000},
        'CREATININE': {'itemid': [50912], 'min': 0, 'max': 150},
        'GLUCOSE': {'itemid': [50809, 50931], 'min': 0, 'max': 10000},
        'HEMATOCRIT': {'itemid': [50810, 51221], 'min': 0, 'max': 100},
        'HEMOGLOBIN': {'itemid': [50811, 51222], 'min': 0, 'max': 50},
        'LACTATE': {'itemid': [50813], 'min': 0, 'max': 50},
        'PLATELET': {'itemid': [51265], 'min': 0, 'max': 10000},
        'POTASSIUM': {'itemid': [50822, 50971], 'min': 0, 'max': 30},
        'PTT': {'itemid': [51275], 'min': 0, 'max': 150},
        'INR': {'itemid': [51237], 'min': 0, 'max': 50},
        'PT': {'itemid': [51274], 'min': 0, 'max': 150},
        'SODIUM': {'itemid': [50824, 50983], 'min': 0, 'max': 200},
        'BUN': {'itemid': [51006], 'min': 0, 'max': 300},
        'WBC': {'itemid': [51300, 51301], 'min': 0, 'max': 1000},
        'MAGNESIUM': {'itemid': [50960], 'min': 0, 'max': float('inf')},
        'CARBONDIOXIDE': {'itemid': [50804], 'min': 0, 'max': float('inf')},
        'BASE_EXCESS': {'itemid': [50802], 'min': -10, 'max': 10},
        'CALCIUM': {'itemid': [50893], 'min': 0, 'max': float('inf')},
        'PH': {'itemid': [50820], 'min': 7, 'max': 8},
        'PAO2': {'itemid': [50821], 'min': 70, 'max': 110},
        'PACO2': {'itemid': [50818], 'min': 22, 'max': 58}
    }

    all_itemids = [item for lab in lab_items.values() for item in lab['itemid']]

    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "hosp/labevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[chunk['itemid'].isin(all_itemids)]
        if not filtered.empty:
            chunks.append(filtered)

    labevents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
    )

    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'],
        parse_dates=['intime', 'outtime']
    )

    labevents_with_icustay = pd.merge(labevents, icustays, on=['subject_id', 'hadm_id'], how='inner')
    labevents_with_icustay = labevents_with_icustay[
        (labevents_with_icustay['charttime'] >= labevents_with_icustay['intime'] - timedelta(hours=6)) &
        (labevents_with_icustay['charttime'] <= labevents_with_icustay['intime'] + timedelta(days=1))
    ]

    lab_values = pd.DataFrame()
    lab_values['subject_id'] = labevents_with_icustay['subject_id']
    lab_values['hadm_id'] = labevents_with_icustay['hadm_id']
    lab_values['stay_id'] = labevents_with_icustay['stay_id']
    lab_values['charttime'] = labevents_with_icustay['charttime']

    for lab_name, lab_info in lab_items.items():
        mask = labevents_with_icustay['itemid'].isin(lab_info['itemid'])
        valid = (labevents_with_icustay['valuenum'] >= lab_info['min']) & (labevents_with_icustay['valuenum'] <= lab_info['max'])
        combined = mask & valid
        temp = pd.Series(index=labevents_with_icustay.index, dtype='float64')
        temp.loc[combined] = labevents_with_icustay.loc[combined, 'valuenum']
        lab_values[lab_name] = temp

    lab_values_grouped = lab_values.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime']).agg(
        {col: 'mean' for col in lab_values.columns if col not in ['subject_id', 'hadm_id', 'stay_id', 'charttime']}
    ).reset_index()

    out_file = os.path.join(output_path, "lab_values.csv")
    lab_values_grouped.to_csv(out_file, index=False)
    print(f"[lab_and_fluid] Lab values saved to: {out_file}")
    return lab_values_grouped


def create_cumulative_fluid_balance(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Calculates cumulative fluid intake, output, and net fluid balance into fluid_cumulative_balance.csv.
    """
    out_file = os.path.join(output_path, "fluid_cumulative_balance.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000000:
        print(f"\n[lab_and_fluid] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("fluid_cumulative_balance.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[lab_and_fluid] Loading pre-existing cumulative fluid from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[lab_and_fluid] Calculating cumulative fluid balance...")
    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'],
        parse_dates=['intime', 'outtime']
    )

    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['stay_id', 'storetime', 'amount', 'amountuom'],
        parse_dates=['storetime']
    )
    inputevents = inputevents[inputevents['amountuom'].isin(['cc', 'L', 'ml', 'uL'])]

    def convert_to_ml(row):
        uom = row['amountuom']
        amt = row['amount']
        if uom in ('ml', 'cc'):
            return amt
        elif uom == 'L':
            return amt * 1000.0
        elif uom == 'uL':
            return amt / 1000.0
        return None

    inputevents['amount_ml'] = inputevents.apply(convert_to_ml, axis=1)
    cv_input = inputevents.groupby(['stay_id', 'storetime'])['amount_ml'].sum().reset_index()
    cv_input.rename(columns={'amount_ml': 'in_amount'}, inplace=True)
    cv_input = pd.merge(cv_input, icustays, on='stay_id', how='inner')

    outputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/outputevents.csv.gz"),
        usecols=['stay_id', 'storetime', 'value', 'valueuom'],
        parse_dates=['storetime']
    )
    outputevents = outputevents[outputevents['valueuom'].isin(['ml', 'mL'])]
    output = outputevents.groupby(['stay_id', 'storetime'])['value'].sum().reset_index()
    output.rename(columns={'value': 'out_amount'}, inplace=True)
    output = pd.merge(output, icustays, on='stay_id', how='inner')

    all_fluid = pd.merge(cv_input, output, on=['stay_id', 'storetime'], how='outer')
    all_fluid = all_fluid.sort_values(['stay_id', 'storetime'])

    all_fluid['in_cum_amt'] = all_fluid.groupby('stay_id')['in_amount'].transform(lambda x: x.fillna(0).cumsum())
    all_fluid['out_cum_amt'] = all_fluid.groupby('stay_id')['out_amount'].transform(lambda x: x.fillna(0).cumsum())
    all_fluid['cum_fluid_balance'] = all_fluid['in_cum_amt'].fillna(0) - all_fluid['out_cum_amt'].fillna(0)

    # Standardize identifier columns
    subj_col = 'subject_id_y' if 'subject_id_y' in all_fluid.columns else ('subject_id_x' if 'subject_id_x' in all_fluid.columns else 'subject_id')
    hadm_col = 'hadm_id_y' if 'hadm_id_y' in all_fluid.columns else ('hadm_id_x' if 'hadm_id_x' in all_fluid.columns else 'hadm_id')
    all_fluid['subject_id'] = all_fluid[subj_col]
    all_fluid['hadm_id'] = all_fluid[hadm_col]
    all_fluid.rename(columns={'storetime': 'charttime'}, inplace=True)

    keep_cols = [c for c in ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'in_cum_amt', 'out_cum_amt', 'cum_fluid_balance'] if c in all_fluid.columns]
    all_fluid = all_fluid[keep_cols]

    all_fluid.to_csv(out_file, index=False)
    print(f"[lab_and_fluid] Cumulative fluid saved to: {out_file}")
    return all_fluid


def create_intravenous_fluids(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Retrieves intravenous fluid intake and exports as fluid_intravenous.csv.
    """
    out_file = os.path.join(output_path, "fluid_intravenous.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000000:
        print(f"\n[lab_and_fluid] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("fluid_intravenous.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[lab_and_fluid] Loading pre-existing intravenous from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[lab_and_fluid] Extracting intravenous fluids...")
    inputevents = pd.read_csv(
        os.path.join(mimic_path, "icu/inputevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'storetime', 'itemid', 'amount', 'amountuom',
                 'rate', 'ordercategoryname', 'secondaryordercategoryname', 'totalamount'],
        parse_dates=['storetime']
    )

    mv_filter_categories = [
        '03-IV Fluid Bolus', '02-Fluids (Crystalloids)', '04-Fluids (Colloids)', '07-Blood Products',
        'Intravenous', 'Intravenous Infusion', 'Intravenous Push', 'IV Drip', 'IV Piggyback'
    ]

    intra_mv = inputevents[
        (inputevents['ordercategoryname'].isin(mv_filter_categories)) |
        (inputevents['secondaryordercategoryname'].isin(mv_filter_categories))
    ]

    intra_mv_grouped = intra_mv.groupby(
        ['subject_id', 'hadm_id', 'stay_id', 'storetime'], as_index=False
    ).agg(
        amount=('totalamount', 'mean')
    ).rename(columns={'storetime': 'charttime'})

    intra_mv_grouped.to_csv(out_file, index=False)
    print(f"[lab_and_fluid] Intravenous fluids saved to: {out_file}")
    return intra_mv_grouped


def create_urine_output(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Retrieves urine output measurements from outputevents table.
    """
    out_file = os.path.join(output_path, "urine_output.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000000:
        print(f"\n[lab_and_fluid] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("urine_output.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[lab_and_fluid] Loading pre-existing urine_output from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[lab_and_fluid] Extracting urine output measurements...")
    urine_itemids = [
        40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651,
        225659, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489
    ]

    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/outputevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[chunk['itemid'].isin(urine_itemids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        print("[lab_and_fluid] No urine output data found.")
        return pd.DataFrame()

    outputevents = pd.concat(chunks, ignore_index=True)
    # GU irrigant volume (item 227488) counts as negative
    outputevents['adjusted_value'] = outputevents.apply(
        lambda row: -1 * row['value'] if row['itemid'] == 227488 and row['value'] > 0 else row['value'],
        axis=1
    )

    urine_output = outputevents.groupby(
        ['subject_id', 'hadm_id', 'stay_id', 'charttime']
    )['adjusted_value'].sum().reset_index()
    urine_output.rename(columns={'adjusted_value': 'urineOutput'}, inplace=True)
    urine_output['urineoutput'] = urine_output['urineOutput']

    urine_output.to_csv(out_file, index=False)
    print(f"[lab_and_fluid] Urine output saved to: {out_file}")
    return urine_output


# Backward compatibility aliases
create_cum_fluid = create_cumulative_fluid_balance
create_intravenous = create_intravenous_fluids


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR), force: bool = False):
    """Runs all lab and fluid extraction functions."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting lab_and_fluid extraction:\n  Drive Source: {mimic_path}\n  Local Output: {output_path}\n  Force Recompute: {force}")
    create_lab_values(mimic_path, output_path, force=force)
    create_cumulative_fluid_balance(mimic_path, output_path, force=force)
    create_intravenous_fluids(mimic_path, output_path, force=force)
    create_urine_output(mimic_path, output_path, force=force)
    print("\n[lab_and_fluid] All lab and fluid processing finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Lab and Fluid Data")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all tables")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, force=args.force)
