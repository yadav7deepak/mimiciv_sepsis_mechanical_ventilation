"""
Sampling and Table Aggregation Script (converted from sampling.ipynb).
Merges lab tables, vitals, fluids, vasopressors, and ventilation parameters,
performs 4-hour window binning/resampling, and merges patient demographics/scores.

Reads intermediate files from LOCAL_DATA_DIR (./data) with Drive fallback.
Reads raw icustays table from Google Drive (MIMIC_PATH).
Saves all outputs locally to LOCAL_DATA_DIR (./data).
"""

import os
import sys
import shutil
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


def create_merged_raw_labs(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Combines lab_values.csv and secondary_lab_values.csv into merged_raw_labs.csv.
    """
    out_file = os.path.join(output_path, "merged_raw_labs.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("merged_raw_labs.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing merged_raw_labs from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[sampling] Creating merged_raw_labs...")
    lab_f = resolve_intermediate_path("lab_values.csv")
    others_f = resolve_intermediate_path("secondary_lab_values.csv")

    lab = pd.read_csv(lab_f)
    others = pd.read_csv(others_f)

    id_cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime']

    lab_cols_fill = {
        'SGOT': np.nan, 'SGPT': np.nan, 'IonizedCalcium': np.nan,
        'MechVent': np.nan, 'FiO2': np.nan, 'urineoutput': np.nan,
        'rate_norepinephrine': np.nan, 'rate_epinephrine': np.nan, 'rate_phenylephrine': np.nan,
        'rate_vasopressin': np.nan, 'rate_dopamine': np.nan, 'vaso_total': np.nan,
        'iv_total': np.nan, 'cum_fluid_balance': np.nan,
        'PEEP': np.nan, 'tidal_volume': np.nan, 'plateau_pressure': np.nan,
        'gcs': np.nan, 'HeartRate': np.nan, 'SysBP': np.nan, 'DiasBP': np.nan,
        'MeanBP': np.nan, 'RespRate': np.nan, 'TempC': np.nan, 'SpO2': np.nan
    }
    for col in lab_cols_fill:
        if col not in lab.columns:
            lab[col] = lab_cols_fill[col]

    others_cols_fill = {
        'POTASSIUM': np.nan, 'SODIUM': np.nan, 'CHLORIDE': np.nan, 'GLUCOSE': np.nan,
        'BUN': np.nan, 'CREATININE': np.nan, 'MAGNESIUM': np.nan, 'CALCIUM': np.nan,
        'CARBONDIOXIDE': np.nan, 'BILIRUBIN': np.nan, 'ALBUMIN': np.nan, 'HEMOGLOBIN': np.nan,
        'WBC': np.nan, 'PLATELET': np.nan, 'PTT': np.nan, 'PT': np.nan, 'INR': np.nan, 'PH': np.nan,
        'PAO2': np.nan, 'PACO2': np.nan, 'BASE_EXCESS': np.nan, 'BICARBONATE': np.nan, 'LACTATE': np.nan,
        'BANDS': np.nan, 'MechVent': np.nan, 'FiO2': np.nan, 'urineoutput': np.nan,
        'rate_norepinephrine': np.nan, 'rate_epinephrine': np.nan, 'rate_phenylephrine': np.nan,
        'rate_vasopressin': np.nan, 'rate_dopamine': np.nan, 'vaso_total': np.nan,
        'iv_total': np.nan, 'cum_fluid_balance': np.nan,
        'PEEP': np.nan, 'tidal_volume': np.nan, 'plateau_pressure': np.nan,
        'gcs': np.nan, 'HeartRate': np.nan, 'SysBP': np.nan, 'DiasBP': np.nan,
        'MeanBP': np.nan, 'RespRate': np.nan, 'TempC': np.nan, 'SpO2': np.nan
    }
    for col in others_cols_fill:
        if col not in others.columns:
            others[col] = others_cols_fill[col]

    all_cols = list(set(id_cols + list(lab_cols_fill.keys()) + list(others_cols_fill.keys())))
    for col in all_cols:
        if col not in lab.columns:
            lab[col] = np.nan
        if col not in others.columns:
            others[col] = np.nan

    merged = pd.concat([lab[all_cols], others[all_cols]], ignore_index=True)

    agg_dict = {
        'gcs': 'mean', 'HeartRate': 'mean', 'SysBP': 'mean', 'DiasBP': 'mean', 'MeanBP': 'mean',
        'RespRate': 'mean', 'TempC': 'mean', 'SpO2': 'mean',
        'POTASSIUM': 'mean', 'SODIUM': 'mean', 'CHLORIDE': 'mean', 'GLUCOSE': 'mean',
        'BUN': 'mean', 'CREATININE': 'mean', 'MAGNESIUM': 'mean', 'CALCIUM': 'mean',
        'IonizedCalcium': 'mean',
        'CARBONDIOXIDE': 'mean', 'SGOT': 'mean', 'SGPT': 'mean',
        'BILIRUBIN': 'mean', 'ALBUMIN': 'mean', 'HEMOGLOBIN': 'mean', 'WBC': 'mean',
        'PLATELET': 'mean', 'PTT': 'mean', 'PT': 'mean', 'INR': 'mean', 'PH': 'mean',
        'PAO2': 'mean', 'PACO2': 'mean', 'BASE_EXCESS': 'mean', 'BICARBONATE': 'mean',
        'LACTATE': 'mean', 'BANDS': 'mean', 'MechVent': 'mean', 'FiO2': 'mean',
        'urineoutput': 'mean', 'rate_norepinephrine': 'mean', 'rate_epinephrine': 'mean',
        'rate_phenylephrine': 'mean', 'rate_vasopressin': 'mean', 'rate_dopamine': 'mean',
        'vaso_total': 'mean', 'iv_total': 'mean', 'cum_fluid_balance': 'mean',
        'PEEP': 'max', 'tidal_volume': 'max', 'plateau_pressure': 'max'
    }

    grouped = merged.groupby(id_cols, as_index=False).agg(agg_dict)
    grouped['shockindex'] = grouped['SysBP'] / grouped['HeartRate'].replace(0, pd.NA)
    grouped['PAO2FiO2ratio'] = grouped['PAO2'] / grouped['FiO2'].replace(0, pd.NA) * 100
    grouped['MechVent'] = (grouped['MechVent'] > 0).astype(int)
    grouped = grouped.sort_values(id_cols).reset_index(drop=True)

    out_file = os.path.join(output_path, "merged_raw_labs.csv")
    grouped.to_csv(out_file, index=False)
    print(f"[sampling] Saved merged_raw_labs to: {out_file}")
    return grouped


def create_merged_raw_vitals_interventions(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Combines vitals, ventilation, urine output, vasopressors, IV fluids, cum fluid, and vent params into merged_raw_vitals_interventions.csv.
    """
    out_file = os.path.join(output_path, "merged_raw_vitals_interventions.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("merged_raw_vitals_interventions.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing merged_raw_vitals_interventions from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return None

    print("\n[sampling] Creating merged_raw_vitals_interventions...")
    vitals = pd.read_csv(resolve_intermediate_path("patient_vital_signs.csv"))
    vent = pd.read_csv(resolve_intermediate_path("ventilation_status.csv"))
    lab = pd.read_csv(resolve_intermediate_path("lab_values.csv"))
    urine = pd.read_csv(resolve_intermediate_path("urine_output.csv"))
    vaso = pd.read_csv(resolve_intermediate_path("vasopressors_combined_by_stay.csv"))
    intravenous = pd.read_csv(resolve_intermediate_path("fluid_intravenous.csv"))
    intravenous.rename(columns={'amount': 'iv_total'}, inplace=True)
    cumfluid = pd.read_csv(resolve_intermediate_path("fluid_cumulative_balance.csv"))
    ventparams = pd.read_csv(resolve_intermediate_path("ventilation_parameters.csv"))

    if 'fio2_chartevents' in vent.columns:
        vent.rename(columns={'fio2_chartevents': 'FiO2'}, inplace=True)
    if 'urineOutput' in urine.columns:
        urine.rename(columns={'urineOutput': 'urineoutput'}, inplace=True)

    all_dfs = [vitals, vent, lab, urine, vaso, intravenous, cumfluid, ventparams]
    standard_cols = set()
    for df in all_dfs:
        standard_cols.update(df.columns)

    aligned_dfs = []
    for df in all_dfs:
        d = df.copy()
        for col in standard_cols:
            if col not in d.columns:
                d[col] = np.nan
        aligned_dfs.append(d)

    merged = pd.concat(aligned_dfs, ignore_index=True)
    agg_funcs = {
        'gcs': 'mean', 'HeartRate': 'mean', 'SysBP': 'mean', 'DiasBP': 'mean', 'MeanBP': 'mean',
        'RespRate': 'mean', 'TempC': 'mean', 'SpO2': 'mean',
        'POTASSIUM': 'mean', 'SODIUM': 'mean', 'CHLORIDE': 'mean', 'GLUCOSE': 'mean',
        'BUN': 'mean', 'CREATININE': 'mean', 'MAGNESIUM': 'mean', 'CALCIUM': 'mean',
        'CARBONDIOXIDE': 'mean',
        'SGOT': 'mean', 'SGPT': 'mean', 'BILIRUBIN': 'mean', 'ALBUMIN': 'mean', 'HEMOGLOBIN': 'mean', 'WBC': 'mean',
        'PLATELET': 'mean', 'PTT': 'mean', 'PT': 'mean', 'INR': 'mean', 'PH': 'mean', 'PAO2': 'mean', 'PACO2': 'mean',
        'BASE_EXCESS': 'mean', 'BICARBONATE': 'mean', 'LACTATE': 'mean', 'BANDS': 'mean',
        'MechVent': 'mean', 'FiO2': 'mean',
        'urineoutput': 'mean',
        'rate_norepinephrine': 'mean', 'rate_epinephrine': 'mean', 'rate_phenylephrine': 'mean', 'rate_vasopressin': 'mean',
        'rate_dopamine': 'mean', 'vaso_total': 'mean',
        'iv_total': 'mean', 'cum_fluid_balance': 'mean',
        'PEEP': 'max', 'tidal_volume': 'max', 'plateau_pressure': 'max'
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in merged.columns}

    grouped = merged.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime'], as_index=False).agg(agg_funcs)
    if 'SysBP' in grouped.columns and 'HeartRate' in grouped.columns:
        grouped['shockindex'] = grouped['SysBP'] / grouped['HeartRate'].replace(0, np.nan)
    if 'PAO2' in grouped.columns and 'FiO2' in grouped.columns:
        grouped['PAO2FiO2ratio'] = grouped['PAO2'] / grouped['FiO2'].replace(0, np.nan) * 100
    if 'MechVent' in grouped.columns:
        grouped['MechVent'] = (grouped['MechVent'] > 0).astype(int)

    grouped = grouped.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'charttime']).reset_index(drop=True)
    out_file = os.path.join(output_path, "merged_raw_vitals_interventions.csv")
    grouped.to_csv(out_file, index=False)
    print(f"[sampling] Saved merged_raw_vitals_interventions to: {out_file}")
    return grouped


def create_sampled_4h_labs(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Bins ICU encounter data into 4-hour intervals for lab values with ventilation parameters.
    Saves to sampled_4h_labs.csv.
    """
    out_file = os.path.join(output_path, "sampled_4h_labs.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("sampled_4h_labs.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing sampled_4h_labs from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[sampling] Creating 4-hour sampled lab table...")
    input_file = resolve_intermediate_path("merged_raw_labs.csv")
    df = pd.read_csv(input_file, parse_dates=["charttime"])

    id_cols = ["stay_id", "subject_id", "hadm_id"]
    df = df.sort_values(id_cols + ["charttime"]).set_index("charttime")

    mean_cols = [
        'ALBUMIN', 'BANDS', 'BASE_EXCESS', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CALCIUM', 'CARBONDIOXIDE',
        'CHLORIDE', 'CREATININE', 'cum_fluid_balance', 'DiasBP', 'gcs', 'GLUCOSE', 'HeartRate', 'HEMOGLOBIN',
        'INR', 'IonizedCalcium', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PACO2', 'PAO2', 'PAO2FiO2ratio',
        'PH', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SGOT', 'SGPT', 'shockindex', 'SODIUM',
        'SpO2', 'SysBP', 'TempC', 'WBC', 'FiO2'
    ]
    sum_cols = ["urineoutput", "iv_total"]
    max_cols = [
        "rate_norepinephrine", "rate_epinephrine", "rate_phenylephrine",
        "rate_vasopressin", "rate_dopamine", "vaso_total", "PEEP", "tidal_volume", "plateau_pressure"
    ]

    agg_map = {c: "mean" for c in mean_cols if c in df.columns}
    agg_map.update({c: "sum" for c in sum_cols if c in df.columns})
    agg_map.update({c: "max" for c in max_cols if c in df.columns})
    if "MechVent" in df.columns:
        agg_map["MechVent"] = "mean"

    grouped = (
        df.groupby(id_cols)
          .resample("4h")
          .agg(agg_map)
          .reset_index()
          .rename(columns={"charttime": "start_time"})
    )

    if "MechVent" in grouped:
        grouped["MechVent"] = (grouped["MechVent"] > 0).astype(int)
    if "gcs" in grouped:
        grouped["gcs"] = grouped["gcs"].round(0)
    if "SysBP" in grouped and "HeartRate" in grouped:
        grouped["shockindex"] = grouped["SysBP"] / grouped["HeartRate"].replace(0, np.nan)
    if "PAO2" in grouped and "FiO2" in grouped:
        grouped["PAO2FiO2ratio"] = grouped["PAO2"] / grouped["FiO2"].replace(0, np.nan) * 100

    grouped = grouped.sort_values(id_cols + ["start_time"]).reset_index(drop=True)
    grouped.to_csv(out_file, index=False)
    print(f"[sampling] Saved sampled_4h_labs to: {out_file}")
    return grouped


def create_sampled_4h_vitals_interventions(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Bins non-lab/vitals/interventions data into 4-hour intervals using vectorized resampling.
    Saves to sampled_4h_vitals_interventions.csv.
    """
    out_file = os.path.join(output_path, "sampled_4h_vitals_interventions.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("sampled_4h_vitals_interventions.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing sampled_4h_vitals_interventions from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[sampling] Creating 4-hour sampled vitals & interventions table...")
    input_file = resolve_intermediate_path("merged_raw_vitals_interventions.csv")
    df = pd.read_csv(input_file, parse_dates=["charttime"])

    id_cols = ["stay_id", "subject_id", "hadm_id"]
    df = df.sort_values(id_cols + ["charttime"]).set_index("charttime")

    mean_cols = [
        'ALBUMIN', 'BANDS', 'BASE_EXCESS', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CALCIUM', 'CARBONDIOXIDE',
        'CHLORIDE', 'CREATININE', 'cum_fluid_balance', 'DiasBP', 'gcs', 'GLUCOSE', 'HeartRate', 'HEMOGLOBIN',
        'INR', 'IonizedCalcium', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PACO2', 'PAO2', 'PAO2FiO2ratio',
        'PH', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SGOT', 'SGPT', 'shockindex', 'SODIUM',
        'SpO2', 'SysBP', 'TempC', 'WBC', 'FiO2'
    ]
    sum_cols = ["urineoutput", "iv_total"]
    max_cols = [
        "rate_norepinephrine", "rate_epinephrine", "rate_phenylephrine",
        "rate_vasopressin", "rate_dopamine", "vaso_total", "PEEP", "tidal_volume", "plateau_pressure"
    ]

    agg_map = {c: "mean" for c in mean_cols if c in df.columns}
    agg_map.update({c: "sum" for c in sum_cols if c in df.columns})
    agg_map.update({c: "max" for c in max_cols if c in df.columns})
    if "MechVent" in df.columns:
        agg_map["MechVent"] = "mean"

    grouped = (
        df.groupby(id_cols)
          .resample("4h")
          .agg(agg_map)
          .reset_index()
          .rename(columns={"charttime": "start_time"})
    )

    if "MechVent" in grouped:
        grouped["MechVent"] = (grouped["MechVent"] > 0).astype(int)
    if "gcs" in grouped:
        grouped["gcs"] = grouped["gcs"].round(0)
    if "SysBP" in grouped and "HeartRate" in grouped:
        grouped["shockindex"] = grouped["SysBP"] / grouped["HeartRate"].replace(0, np.nan)
    if "PAO2" in grouped and "FiO2" in grouped:
        grouped["PAO2FiO2ratio"] = grouped["PAO2"] / grouped["FiO2"].replace(0, np.nan) * 100

    grouped = grouped.sort_values(id_cols + ["start_time"]).reset_index(drop=True)
    grouped.to_csv(out_file, index=False)
    print(f"[sampling] Saved sampled_4h_vitals_interventions to: {out_file}")
    return grouped


def create_sampled_4h_combined(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Merges sampled lab and non-lab datasets into sampled_4h_combined.csv.
    """
    out_file = os.path.join(output_path, "sampled_4h_combined.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("sampled_4h_combined.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing sampled_4h_combined from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[sampling] Creating sampled_4h_combined...")
    f1 = resolve_intermediate_path("sampled_4h_labs.csv")
    f2 = resolve_intermediate_path("sampled_4h_vitals_interventions.csv")

    df1 = pd.read_csv(f1, parse_dates=["start_time"])
    df2 = pd.read_csv(f2, parse_dates=["start_time"])

    df = pd.concat([df1, df2], ignore_index=True)
    if "icustay_id" in df.columns:
        df.rename(columns={"icustay_id": "stay_id"}, inplace=True)

    id_cols = ["stay_id", "subject_id", "hadm_id"]
    df = df.sort_values(id_cols + ["start_time"]).set_index("start_time")

    mean_cols = [
        'ALBUMIN', 'BANDS', 'BASE_EXCESS', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CALCIUM', 'CARBONDIOXIDE',
        'CHLORIDE', 'CREATININE', 'cum_fluid_balance', 'DiasBP', 'gcs', 'GLUCOSE', 'HeartRate', 'HEMOGLOBIN',
        'INR', 'IonizedCalcium', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PACO2', 'PAO2', 'PAO2FiO2ratio',
        'PH', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SGOT', 'SGPT', 'shockindex', 'SODIUM',
        'SpO2', 'SysBP', 'TempC', 'WBC', 'FiO2'
    ]
    sum_cols = ["urineoutput", "iv_total"]
    max_cols = [
        "rate_norepinephrine", "rate_epinephrine", "rate_phenylephrine",
        "rate_vasopressin", "rate_dopamine", "vaso_total", "PEEP", "tidal_volume", "plateau_pressure"
    ]

    agg_map = {c: "mean" for c in mean_cols if c in df.columns}
    agg_map.update({c: "sum" for c in sum_cols if c in df.columns})
    agg_map.update({c: "max" for c in max_cols if c in df.columns})
    if "MechVent" in df.columns:
        agg_map["MechVent"] = "mean"

    grouped = (
        df.groupby(id_cols)
          .resample("4h")
          .agg(agg_map)
          .reset_index()
          .rename(columns={"start_time": "start_time"})
    )

    if "MechVent" in grouped:
        grouped["MechVent"] = (grouped["MechVent"] > 0).astype(int)
    if "gcs" in grouped:
        grouped["gcs"] = grouped["gcs"].round(0)
    if "SysBP" in grouped and "HeartRate" in grouped:
        grouped["shockindex"] = grouped["SysBP"] / grouped["HeartRate"].replace(0, np.nan)
    if "PAO2" in grouped and "FiO2" in grouped:
        grouped["PAO2FiO2ratio"] = grouped["PAO2"] / grouped["FiO2"].replace(0, np.nan) * 100

    grouped = grouped.sort_values(id_cols + ["start_time"]).reset_index(drop=True)
    grouped.to_csv(out_file, index=False)
    print(f"[sampling] Saved sampled_4h_combined to: {out_file}")
    return grouped


def create_final_master_patient_dataset(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Merges sampled 4-hour data with SIRS/SOFA scores, demographics, weight, and icustays.
    Saves to final_patient_dataset_master.csv.
    """
    out_file = os.path.join(output_path, "final_patient_dataset_master.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[sampling] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("final_patient_dataset_master.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[sampling] Copying pre-existing master patient dataset from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[sampling] Merging sampled dataset with scores and demographics...")
    samp = pd.read_csv(resolve_intermediate_path("sampled_4h_combined.csv"), parse_dates=['start_time'])
    sr = pd.read_csv(resolve_intermediate_path("sirs_scores.csv"), parse_dates=['start_time'])
    sf = pd.read_csv(resolve_intermediate_path("sofa_scores.csv"), parse_dates=['start_time'])
    dem = pd.read_csv(resolve_intermediate_path("patient_demographics.csv"))
    weig = pd.read_csv(resolve_intermediate_path("patient_weight.csv"))
    icu = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id']
    )

    merged = samp.merge(sr[['stay_id', 'start_time', 'SIRS']], on=['stay_id', 'start_time'], how='left')
    merged = merged.merge(sf[['stay_id', 'start_time', 'SOFA']], on=['stay_id', 'start_time'], how='left')
    merged = merged.merge(dem, on='stay_id', how='left', suffixes=('', '_dem'))
    merged = merged.merge(weig[['stay_id', 'weight']], on='stay_id', how='left')
    merged = merged.merge(icu[['stay_id', 'subject_id', 'hadm_id']], on='stay_id', how='inner', suffixes=('', '_icu'))

    outcols = [
        'stay_id', 'subject_id', 'hadm_id', 'start_time', 'first_admit_age',
        'gender', 'weight', 'ICU_readm', 'icu_readm', 'elixhauser_score', 'SOFA', 'SIRS',
        'gcs', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'shockindex', 'RespRate',
        'TempC', 'SpO2', 'POTASSIUM', 'SODIUM', 'CHLORIDE', 'GLUCOSE', 'BUN',
        'CREATININE', 'MAGNESIUM', 'CALCIUM', 'IonizedCalcium', 'ionizedcalcium', 'CARBONDIOXIDE',
        'SGOT', 'SGPT', 'BILIRUBIN', 'ALBUMIN', 'HEMOGLOBIN', 'WBC', 'PLATELET',
        'PTT', 'PT', 'INR', 'PH', 'PAO2', 'PACO2', 'BASE_EXCESS', 'BICARBONATE',
        'LACTATE', 'PAO2FiO2ratio', 'MechVent', 'FiO2', 'urineoutput',
        'vaso_total', 'iv_total', 'cum_fluid_balance', 'PEEP', 'tidal_volume',
        'plateau_pressure', 'hospmort90day', 'HospMort90day', 'dischtime', 'deathtime'
    ]
    final_cols = [col for col in outcols if col in merged.columns]
    seen = set()
    dedup_cols = []
    for c in final_cols:
        if c not in seen:
            seen.add(c)
            dedup_cols.append(c)

    out = merged[dedup_cols].sort_values(['stay_id', 'subject_id', 'hadm_id', 'start_time']).reset_index(drop=True)
    out.to_csv(out_file, index=False)
    print(f"[sampling] Saved master patient dataset to: {out_file}")
    return out


# Backward-compatibility aliases
create_overalltable_lab_withventparams = create_merged_raw_labs
create_overalltable_withoutLab_withventparams = create_merged_raw_vitals_interventions
create_sampled_lab_withventparams = create_sampled_4h_labs
create_sampled_withoutlab_withventparams_corrected = create_sampled_4h_vitals_interventions
create_sampled_all_withventparams_fast = create_sampled_4h_combined
merge_sampled_with_scdem_withventparams = create_final_master_patient_dataset


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR), force: bool = False, merge_final: bool = False):
    """Runs sampling pipeline steps."""
    mount_drive()
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting sampling pipeline:\n  Drive Source: {mimic_path}\n  Local Output: {output_path}\n  Force Recompute: {force}")
    create_merged_raw_labs(output_path, force=force)
    create_merged_raw_vitals_interventions(output_path, force=force)
    create_sampled_4h_labs(output_path, force=force)
    create_sampled_4h_vitals_interventions(output_path, force=force)
    create_sampled_4h_combined(output_path, force=force)

    if merge_final:
        verify_mimic_path(mimic_path, raise_error=True)
        create_final_master_patient_dataset(mimic_path, output_path, force=force)

    print("\n[sampling] Sampling stage finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Sampling and Resampling")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--merge-final", action="store_true", help="Perform final merge with demographics and scores")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all tables")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, force=args.force, merge_final=args.merge_final)

