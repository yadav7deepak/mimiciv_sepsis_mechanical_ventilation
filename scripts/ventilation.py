"""
Ventilation Parameters & Clinical Scores Script (converted from ventilation.ipynb).
Extracts:
- Ventilator parameters (PEEP, tidal volume, plateau pressure)
- Ventilation params (FiO2, MechVent status)
- Combined vasopressors table
- Others lab values (SGOT, SGPT, Ionized Calcium)
- SIRS scores on sampled data
- SOFA scores on sampled data

Reads raw tables from Google Drive (MIMIC_PATH).
Reads intermediate files from LOCAL_DATA_DIR (./data) with Drive fallback.
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


def create_vent_parameters(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts PEEP, tidal volume, and plateau pressure by charttime.
    """
    out_file = os.path.join(output_path, "ventilation_parameters.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("ventilation_parameters.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[ventilation] Loading pre-existing ventilation parameters from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[ventilation] Extracting ventilator parameters from chartevents...")
    PEEP_itemids = [60, 437, 505, 506, 686, 220339, 224699, 224700]
    tv_itemids = [639, 654, 681, 682, 683, 684, 224685, 224684, 224686]
    plateau_itemids = [543, 224696]
    target_ids = PEEP_itemids + tv_itemids + plateau_itemids

    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['stay_id', 'subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'value'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filt = (chunk['value'].notnull()) & (chunk['itemid'].isin(target_ids))
        sub = chunk.loc[filt, ['stay_id', 'subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']]
        if not sub.empty:
            chunks.append(sub)

    ce_filt = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['stay_id', 'subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']
    )

    ce_filt['PEEP'] = ce_filt.apply(lambda x: x['valuenum'] if x['itemid'] in PEEP_itemids else None, axis=1)
    ce_filt['tidal_volume'] = ce_filt.apply(lambda x: x['valuenum'] if x['itemid'] in tv_itemids else None, axis=1)
    ce_filt['plateau_pressure'] = ce_filt.apply(lambda x: x['valuenum'] if x['itemid'] in plateau_itemids else None, axis=1)

    aggregate = ce_filt.groupby(['stay_id', 'subject_id', 'hadm_id', 'charttime'], as_index=False).agg({
        'PEEP': 'mean',
        'tidal_volume': 'mean',
        'plateau_pressure': 'mean'
    }).sort_values(by=['stay_id', 'charttime'])

    aggregate.to_csv(out_file, index=False)
    print(f"[ventilation] Ventilator parameters saved to: {out_file}")
    return aggregate


def create_ventilation_params(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts FiO2 and mechanical ventilation status per charttime into ventilation_status.csv.
    """
    out_file = os.path.join(output_path, "ventilation_status.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("ventilation_status.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[ventilation] Loading pre-existing ventilation status from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[ventilation] Extracting ventilation status and FiO2...")
    FIO2_itemids = [223835, 3420, 3422, 190]
    MechVent_itemids = [
        720, 223848, 223849, 467, 445, 448, 449, 450, 1340, 1486, 1600, 224687,
        639, 654, 681, 682, 683, 684, 224685, 224684, 224686, 218, 436, 535,
        444, 459, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655,
        2000, 226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707,
        224709, 224705, 224706, 60, 437, 505, 506, 686, 220339, 224700, 3459,
        501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701
    ]
    target_all = FIO2_itemids + MechVent_itemids

    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filt = chunk['value'].notnull() & (chunk['itemid'].isin(target_all))
        sub = chunk.loc[filt, ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']]
        if not sub.empty:
            chunks.append(sub)

    ce_filt = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
    )

    def fio2_normalized(row):
        iid = row['itemid']
        vn = row['valuenum']
        if iid == 223835:
            if pd.notnull(vn):
                if 0 < vn <= 1:
                    return vn * 100.0
                elif 1 < vn < 21:
                    return None
                elif 21 <= vn <= 100:
                    return vn
        elif iid in [3420, 3422]:
            return vn
        elif iid == 190:
            if pd.notnull(vn) and 0.20 < vn < 1:
                return vn * 100.0
        return None

    ce_filt['fio2_chartevents'] = ce_filt.apply(fio2_normalized, axis=1)

    def mech_vent_flag(row):
        iid = row['itemid']
        value = str(row['value'])
        if pd.isnull(iid) or pd.isnull(row['value']):
            return 0
        if iid == 720 and value != 'Other/Remarks':
            return 1
        if iid == 223848 and value != 'Other':
            return 1
        if iid == 223849:
            return 1
        if iid == 467 and value == 'Ventilator':
            return 1
        if iid in MechVent_itemids:
            return 1
        return 0

    ce_filt['MechVent'] = ce_filt.apply(mech_vent_flag, axis=1)

    agg = ce_filt.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime'], as_index=False).agg({
        'fio2_chartevents': 'max',
        'MechVent': 'max'
    })

    out_file = os.path.join(output_path, "ventilation_status.csv")
    agg.to_csv(out_file, index=False)
    print(f"[ventilation] Ventilation status saved to: {out_file}")
    return agg


def create_vasopressors(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Combines individual vasopressor doses into a single table.
    """
    out_file = os.path.join(output_path, "vasopressors_combined.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("vasopressors_combined.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 1000:
        print(f"\n[ventilation] Loading pre-existing vasopressors_combined from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[ventilation] Combining vasopressor doses...")
    norepi_f = resolve_intermediate_path("vasopressor_norepinephrine.csv")
    epi_f = resolve_intermediate_path("vasopressor_epinephrine.csv")
    phenylep_f = resolve_intermediate_path("vasopressor_phenylephrine.csv")
    dopamine_f = resolve_intermediate_path("vasopressor_dopamine.csv")
    vasopressin_f = resolve_intermediate_path("vasopressor_vasopressin.csv")

    norepi = pd.read_csv(norepi_f, usecols=['icustay_id', 'starttime', 'vaso_rate'])
    epi = pd.read_csv(epi_f, usecols=['icustay_id', 'starttime', 'vaso_rate'])
    phenylep = pd.read_csv(phenylep_f, usecols=['icustay_id', 'starttime', 'vaso_rate'])
    dopamine = pd.read_csv(dopamine_f, usecols=['icustay_id', 'starttime', 'vaso_rate'])
    vasopressin = pd.read_csv(vasopressin_f, usecols=['icustay_id', 'starttime', 'vaso_rate'])

    norepi['rate_norepinephrine'] = norepi['vaso_rate']
    norepi = norepi.drop(columns=['vaso_rate'])
    for col in ['rate_epinephrine', 'rate_phenylephrine', 'rate_dopamine', 'rate_vasopressin']:
        norepi[col] = None

    epi['rate_epinephrine'] = epi['vaso_rate']
    epi = epi.drop(columns=['vaso_rate'])
    for col in ['rate_norepinephrine', 'rate_phenylephrine', 'rate_dopamine', 'rate_vasopressin']:
        epi[col] = None

    phenylep['rate_phenylephrine'] = phenylep['vaso_rate']
    phenylep = phenylep.drop(columns=['vaso_rate'])
    for col in ['rate_norepinephrine', 'rate_epinephrine', 'rate_dopamine', 'rate_vasopressin']:
        phenylep[col] = None

    dopamine['rate_dopamine'] = dopamine['vaso_rate']
    dopamine = dopamine.drop(columns=['vaso_rate'])
    for col in ['rate_norepinephrine', 'rate_epinephrine', 'rate_phenylephrine', 'rate_vasopressin']:
        dopamine[col] = None

    vasopressin['rate_vasopressin'] = vasopressin['vaso_rate']
    vasopressin = vasopressin.drop(columns=['vaso_rate'])
    for col in ['rate_norepinephrine', 'rate_epinephrine', 'rate_phenylephrine', 'rate_dopamine']:
        vasopressin[col] = None

    union_df = pd.concat([norepi, epi, phenylep, dopamine, vasopressin], ignore_index=True)
    rate_cols = ['rate_norepinephrine', 'rate_epinephrine', 'rate_phenylephrine', 'rate_dopamine', 'rate_vasopressin']
    for col in rate_cols:
        union_df[col] = pd.to_numeric(union_df[col], errors='coerce')

    vaso = union_df.groupby(['icustay_id', 'starttime'], as_index=False)[rate_cols].max()
    vaso['vaso_total'] = (
        vaso['rate_norepinephrine'].fillna(0) +
        vaso['rate_epinephrine'].fillna(0) +
        vaso['rate_phenylephrine'].fillna(0) / 2.2 +
        vaso['rate_dopamine'].fillna(0) / 100.0 +
        vaso['rate_vasopressin'].fillna(0) * 8.33
    )

    vaso = vaso.sort_values(['icustay_id', 'starttime']).reset_index(drop=True)
    out_file = os.path.join(output_path, "vasopressors_combined.csv")
    vaso.to_csv(out_file, index=False)
    print(f"[ventilation] Vasopressors combined saved to: {out_file}")
    return vaso


def create_vasopressors_charttable(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Joins combined vasopressors with icustays to get subject_id, hadm_id, stay_id, and charttime.
    """
    out_file = os.path.join(output_path, "vasopressors_combined_by_stay.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("vasopressors_combined_by_stay.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 1000:
        print(f"\n[ventilation] Loading pre-existing vasopressors_combined_by_stay from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[ventilation] Creating final vasopressors chart table...")
    vaso_f = resolve_intermediate_path("vasopressors_combined.csv")
    vaso = pd.read_csv(vaso_f, parse_dates=["starttime"])
    vaso = vaso.rename(columns={'starttime': 'charttime', 'icustay_id': 'stay_id'})

    icu = pd.read_csv(os.path.join(mimic_path, "icu/icustays.csv.gz"), usecols=['subject_id', 'hadm_id', 'stay_id'])
    df = vaso.merge(icu[['stay_id', 'subject_id', 'hadm_id']], on='stay_id', how='inner')

    df = df[[
        'subject_id', 'hadm_id', 'stay_id', 'charttime',
        'rate_norepinephrine', 'rate_epinephrine', 'rate_phenylephrine',
        'rate_vasopressin', 'rate_dopamine', 'vaso_total'
    ]]
    df.to_csv(out_file, index=False)
    print(f"[ventilation] Final vasopressors table saved to: {out_file}")
    return df


def create_secondary_lab_values(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts and averages SGOT, SGPT, and Ionized Calcium from chartevents into secondary_lab_values.csv.
    """
    out_file = os.path.join(output_path, "secondary_lab_values.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("secondary_lab_values.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[ventilation] Loading pre-existing secondary lab values from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[ventilation] Extracting SGOT, SGPT, and Ionized Calcium...")
    SGOT_ids = [220587]
    SGPT_ids = [220644]
    IonizedCalcium_ids = [225667]
    target_ids = SGOT_ids + SGPT_ids + IonizedCalcium_ids

    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'valuenum', 'itemid'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[chunk['itemid'].isin(target_ids)]
        if not filtered.empty:
            chunks.append(filtered)

    filtered_ce = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'valuenum', 'itemid']
    )
    filtered_ce['SGOT'] = filtered_ce.apply(lambda r: r['valuenum'] if r['itemid'] in SGOT_ids else None, axis=1)
    filtered_ce['SGPT'] = filtered_ce.apply(lambda r: r['valuenum'] if r['itemid'] in SGPT_ids else None, axis=1)
    filtered_ce['IonizedCalcium'] = filtered_ce.apply(lambda r: r['valuenum'] if r['itemid'] in IonizedCalcium_ids else None, axis=1)

    grouped = filtered_ce.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime'], as_index=False).agg({
        'SGOT': 'mean',
        'SGPT': 'mean',
        'IonizedCalcium': 'mean'
    })

    grouped.to_csv(out_file, index=False)
    print(f"[ventilation] Other lab values saved to: {out_file}")
    return grouped


def create_sirs_scores(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Calculates SIRS score on 4-hour sampled ventilation data into sirs_scores.csv.
    """
    out_file = os.path.join(output_path, "sirs_scores.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("sirs_scores.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 10000:
        print(f"\n[ventilation] Copying pre-existing SIRS from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[ventilation] Calculating SIRS score...")
    input_file = resolve_intermediate_path("sampled_4h_combined.csv")
    scorecomp = pd.read_csv(input_file, parse_dates=['start_time'])

    scorecalc = scorecomp.copy()
    scorecalc['Temp_score'] = scorecalc['TempC'].apply(
        lambda x: 1 if pd.notnull(x) and (x < 36.0 or x > 38.0) else (0 if pd.notnull(x) else None)
    )
    scorecalc['HeartRate_score'] = scorecalc['HeartRate'].apply(
        lambda x: 1 if pd.notnull(x) and x > 90.0 else (0 if pd.notnull(x) else None)
    )
    scorecalc['Resp_score'] = scorecalc.apply(
        lambda row: 1 if (pd.notnull(row['RespRate']) and row['RespRate'] > 20.0) or
                         (pd.notnull(row['PACO2']) and row['PACO2'] < 32.0)
                      else (0 if pd.notnull(row['RespRate']) or pd.notnull(row['PACO2']) else None),
        axis=1
    )
    scorecalc['WBC_score'] = scorecalc.apply(
        lambda row: 1 if (pd.notnull(row['WBC']) and (row['WBC'] < 4.0 or row['WBC'] > 12.0)) or
                         (pd.notnull(row['BANDS']) and row['BANDS'] > 10)
                      else (0 if pd.notnull(row['WBC']) or pd.notnull(row['BANDS']) else None),
        axis=1
    )

    final_df = scorecalc[['stay_id', 'subject_id', 'hadm_id', 'start_time',
                          'Temp_score', 'HeartRate_score', 'Resp_score', 'WBC_score']].copy()
    final_df['SIRS'] = final_df[['Temp_score', 'HeartRate_score', 'Resp_score', 'WBC_score']].fillna(0).sum(axis=1)
    final_df = final_df[['stay_id', 'subject_id', 'hadm_id', 'start_time', 'SIRS',
                         'Temp_score', 'HeartRate_score', 'Resp_score', 'WBC_score']]

    final_df.to_csv(out_file, index=False)
    print(f"[ventilation] SIRS scores saved to: {out_file}")
    return final_df


def create_sofa_scores(output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Calculates SOFA score on 4-hour sampled ventilation data into sofa_scores.csv.
    """
    out_file = os.path.join(output_path, "sofa_scores.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        print(f"\n[ventilation] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("sofa_scores.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 10000:
        print(f"\n[ventilation] Copying pre-existing SOFA scores from Drive: {drive_file}")
        shutil.copyfile(drive_file, out_file)
        return pd.read_csv(out_file)

    print("\n[ventilation] Calculating SOFA score...")
    input_file = resolve_intermediate_path("sampled_4h_combined.csv")
    scorecomp = pd.read_csv(input_file, parse_dates=['start_time'])

    scorecalc = scorecomp.copy()

    # Coagulation
    conditions_coag = [
        scorecalc['PLATELET'] < 20,
        scorecalc['PLATELET'] < 50,
        scorecalc['PLATELET'] < 100,
        scorecalc['PLATELET'] < 150,
        scorecalc['PLATELET'].isnull()
    ]
    choices_coag = [4, 3, 2, 1, None]
    scorecalc['coagulation'] = np.select(conditions_coag, choices_coag, default=0)

    # Renal
    conditions_renal = [
        (scorecalc['CREATININE'] >= 5.0) | (scorecalc['urineoutput'] < 200),
        (scorecalc['CREATININE'] >= 3.5) | (scorecalc['urineoutput'] < 500),
        (scorecalc['CREATININE'] >= 2.0),
        (scorecalc['CREATININE'] >= 1.2),
        scorecalc['urineoutput'].isnull() & scorecalc['CREATININE'].isnull()
    ]
    choices_renal = [4, 3, 2, 1, None]
    scorecalc['renal'] = np.select(conditions_renal, choices_renal, default=0)

    # Respiration
    conditions_resp = [
        (pd.notnull(scorecalc['PAO2FiO2ratio'])) & (scorecalc['PAO2FiO2ratio'] < 100) & (scorecalc['MechVent'] == 1),
        (pd.notnull(scorecalc['PAO2FiO2ratio'])) & (scorecalc['PAO2FiO2ratio'] < 200) & (scorecalc['MechVent'] == 1),
        (pd.notnull(scorecalc['PAO2FiO2ratio'])) & (scorecalc['PAO2FiO2ratio'] < 300),
        (pd.notnull(scorecalc['PAO2FiO2ratio'])) & (scorecalc['PAO2FiO2ratio'] < 400),
        scorecalc['PAO2FiO2ratio'].isnull()
    ]
    choices_resp = [4, 3, 2, 1, None]
    scorecalc['respiration'] = np.select(conditions_resp, choices_resp, default=0)

    # CNS / GCS
    conditions_cns = [
        (scorecalc['gcs'] >= 13) & (scorecalc['gcs'] <= 14),
        (scorecalc['gcs'] >= 10) & (scorecalc['gcs'] <= 12),
        (scorecalc['gcs'] >= 6) & (scorecalc['gcs'] <= 9),
        scorecalc['gcs'] < 6,
        scorecalc['gcs'].isnull()
    ]
    choices_cns = [1, 2, 3, 4, None]
    scorecalc['cns'] = np.select(conditions_cns, choices_cns, default=0)

    # Cardiovascular
    conditions_cardio = [
        (scorecalc['rate_dopamine'] > 15) | (scorecalc['rate_epinephrine'] > 0.1) | (scorecalc['rate_norepinephrine'] > 0.1),
        (scorecalc['rate_dopamine'] > 5) | ((scorecalc['rate_epinephrine'] <= 0.1) & (scorecalc['rate_epinephrine'] > 0)) | ((scorecalc['rate_norepinephrine'] <= 0.1) & (scorecalc['rate_norepinephrine'] > 0)),
        (scorecalc['rate_dopamine'] <= 5) & (scorecalc['rate_dopamine'] > 0),
        scorecalc['MeanBP'] < 70,
        scorecalc['rate_dopamine'].isnull() & scorecalc['rate_epinephrine'].isnull() & scorecalc['rate_norepinephrine'].isnull() & scorecalc['MeanBP'].isnull()
    ]
    choices_cardio = [4, 3, 2, 1, None]
    scorecalc['cardiovascular'] = np.select(conditions_cardio, choices_cardio, default=0)

    # Liver
    conditions_liver = [
        scorecalc['BILIRUBIN'] >= 12.0,
        scorecalc['BILIRUBIN'] >= 6.0,
        scorecalc['BILIRUBIN'] >= 2.0,
        scorecalc['BILIRUBIN'] >= 1.2,
        scorecalc['BILIRUBIN'].isnull()
    ]
    choices_liver = [4, 3, 2, 1, None]
    scorecalc['liver'] = np.select(conditions_liver, choices_liver, default=0)

    scorecalc['SOFA'] = scorecalc[['respiration', 'cns', 'cardiovascular', 'liver', 'coagulation', 'renal']].fillna(0).sum(axis=1)

    final_columns = [
        'stay_id', 'subject_id', 'hadm_id', 'start_time',
        'PAO2FiO2ratio', 'MechVent', 'gcs', 'MeanBP', 'rate_dopamine',
        'rate_norepinephrine', 'rate_epinephrine', 'BILIRUBIN',
        'PLATELET', 'CREATININE', 'urineoutput',
        'respiration', 'cns', 'cardiovascular', 'liver',
        'coagulation', 'renal', 'SOFA'
    ]
    for col in final_columns:
        if col not in scorecalc.columns:
            scorecalc[col] = None

    final_df = scorecalc[final_columns].copy().sort_values(by=['stay_id', 'subject_id', 'hadm_id', 'start_time']).reset_index(drop=True)
    out_file = os.path.join(output_path, "sofa_scores.csv")
    final_df.to_csv(out_file, index=False)
    print(f"[ventilation] SOFA scores saved to: {out_file}")
    return final_df


# Backward compatibility aliases
create_ventilation_parameters = create_vent_parameters
create_ventilation_status = create_ventilation_params
create_others = create_secondary_lab_values
create_SIRS_sampled_withventparams = create_sirs_scores
create_SOFA_sampled_withventparams = create_sofa_scores


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR), force: bool = False, include_scores: bool = False):
    """Runs all ventilation preparation functions."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting ventilation extraction:\n  Drive Source: {mimic_path}\n  Local Output: {output_path}\n  Force Recompute: {force}")
    create_vent_parameters(mimic_path, output_path, force=force)
    create_ventilation_params(mimic_path, output_path, force=force)
    create_vasopressors(output_path, force=force)
    create_vasopressors_charttable(mimic_path, output_path, force=force)
    create_secondary_lab_values(mimic_path, output_path, force=force)

    if include_scores:
        create_sirs_scores(output_path, force=force)
        create_sofa_scores(output_path, force=force)
    print("\n[ventilation] Ventilation processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Ventilation Data")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--include-scores", action="store_true", help="Calculate SIRS and SOFA scores (requires sampled_all_withventparams.csv)")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all tables")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, force=args.force, include_scores=args.include_scores)
