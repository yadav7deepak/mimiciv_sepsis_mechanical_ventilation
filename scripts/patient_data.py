"""
Patient Data Preparation Script (converted from patient_data.ipynb).
Extracts demographics, Elixhauser comorbidity scores, echo data, GCS,
vital signs, weights, and hospital mortality / ICU stay times from MIMIC-IV.

Reads raw tables from Google Drive (MIMIC_PATH).
Saves all processed outputs locally to LOCAL_DATA_DIR (./data).
"""

import os
import sys
import re
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


def create_elixhauser_score(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Calculate Elixhauser comorbidities and scores from MIMIC-IV diagnoses_icd and admissions.
    """
    output_file = os.path.join(output_path, "patient_elixhauser_scores.csv")
    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 1000000:
        print(f"\n[patient_data] Reusing existing {output_file} ({os.path.getsize(output_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(output_file)

    print("\n[patient_data] Calculating Elixhauser comorbidity scores...")
    diagnoses_path = os.path.join(mimic_path, "hosp/diagnoses_icd.csv.gz")
    admissions_path = os.path.join(mimic_path, "hosp/admissions.csv.gz")

    diagnoses = pd.read_csv(diagnoses_path, usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
    admissions = pd.read_csv(admissions_path, usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime'])

    diagnoses = diagnoses.dropna(subset=['icd_code']).copy()
    diagnoses['icd_code'] = diagnoses['icd_code'].str.strip()

    comorbidities_icd9 = {
        'CHF': ['4280', '4281', '4289', '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843', '4289'],
        'CARDIAC_ARRHYTHMIAS': ['42610', '42611', '42613', '4262', '4263', '4264', '42651', '42652', '42653', '4266', '4267', '4268', '4270', '4272', '42731', '42760', '4279', '7850', 'V450', 'V533'],
        'VALVULAR_DISEASE': ['0932', '394', '395', '396', '397', '424', '7463', '7464', '7465', '7466', 'V422', 'V433'],
        'PULMONARY_CIRCULATION': ['416', '4179'],
        'PERIPHERAL_VASCULAR': ['440', '441', '442', '4431', '4432', '4438', '4439', '4471', '5571', '5579', 'V434'],
        'HYPERTENSION_UNCOMPLICATED': ['401'],
        'HYPERTENSION_COMPLICATED': ['402', '403', '404', '405'],
        'PARALYSIS': ['342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'],
        'OTHER_NEUROLOGICAL': ['3319', '3320', '3334', '3335', '334', '335', '340', '341', '345', '3481', '3483', '7803', '7843'],
        'CHRONIC_PULMONARY': ['490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505', '5064', '5081', '5088'],
        'DIABETES_UNCOMPLICATED': ['2500', '2501', '2502', '2503'],
        'DIABETES_COMPLICATED': ['2504', '2505', '2506', '2507', '2508', '2509'],
        'HYPOTHYROIDISM': ['243', '244', '2461', '2468'],
        'RENAL_FAILURE': ['585', '586', 'V420', 'V451', 'V56'],
        'LIVER_DISEASE': ['07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570', '571', '5722', '5723', '5724', '5728', '5733', '5738', '5739', 'V427'],
        'PEPTIC_ULCER': ['531', '532', '533', '534'],
        'AIDS_HIV': ['042'],
        'LYMPHOMA': ['200', '201', '202', 'V107'],
        'METASTATIC_CANCER': ['196', '197', '198', '199'],
        'SOLID_TUMOR': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195'],
        'RHEUMATOID_ARTHRITIS': ['7010', '7100', '7101', '7104', '7140', '7141', '7142', '7148', '720', '725'],
        'COAGULOPATHY': ['286', '2871', '2873', '2874', '2875'],
        'OBESITY': ['2780'],
        'WEIGHT_LOSS': ['260', '261', '262', '263', '7832'],
        'FLUID_ELECTROLYTE': ['276'],
        'BLOOD_LOSS_ANEMIA': ['2800'],
        'DEFICIENCY_ANEMIAS': ['2801', '2802', '2803', '2804', '2805', '2806', '2807', '2808', '2809', '281'],
        'ALCOHOL_ABUSE': ['291', '303', '3050', 'V113'],
        'DRUG_ABUSE': ['292', '304', '3052', '3053', '3054', '3055', '3056', '3057', '3058', '3059', 'V6542'],
        'PSYCHOSES': ['295', '29604', '29614', '29644', '29654', '297', '298'],
        'DEPRESSION': ['3004', '30112', '3090', '3091', '311']
    }

    comorbidities_icd10 = {
        'CHF': ['I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'I43', 'I50', 'P290'],
        'CARDIAC_ARRHYTHMIAS': ['I441', 'I442', 'I443', 'I456', 'I459', 'I47', 'I48', 'I49', 'R000', 'R001', 'R008', 'T821', 'Z450', 'Z950'],
        'VALVULAR_DISEASE': ['A520', 'I05', 'I06', 'I07', 'I08', 'I091', 'I098', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'Q230', 'Q231', 'Q232', 'Q233', 'Z952', 'Z953', 'Z954'],
        'PULMONARY_CIRCULATION': ['I26', 'I27', 'I280', 'I288', 'I289'],
        'PERIPHERAL_VASCULAR': ['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'],
        'HYPERTENSION_UNCOMPLICATED': ['I10'],
        'HYPERTENSION_COMPLICATED': ['I11', 'I12', 'I13', 'I15'],
        'PARALYSIS': ['G041', 'G114', 'G801', 'G802', 'G81', 'G82', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'],
        'OTHER_NEUROLOGICAL': ['G10', 'G11', 'G12', 'G13', 'G20', 'G21', 'G22', 'G254', 'G255', 'G312', 'G318', 'G319', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'G931', 'G934', 'R470', 'R56'],
        'CHRONIC_PULMONARY': ['I278', 'I279', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J684', 'J701', 'J703'],
        'DIABETES_UNCOMPLICATED': ['E100', 'E101', 'E109', 'E110', 'E111', 'E119', 'E120', 'E121', 'E129', 'E130', 'E131', 'E139', 'E140', 'E141', 'E149'],
        'DIABETES_COMPLICATED': ['E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148'],
        'HYPOTHYROIDISM': ['E00', 'E01', 'E02', 'E03', 'E890'],
        'RENAL_FAILURE': ['I120', 'I131', 'N18', 'N19', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'],
        'LIVER_DISEASE': ['B18', 'I85', 'I864', 'I982', 'K70', 'K711', 'K713', 'K714', 'K715', 'K717', 'K72', 'K73', 'K74', 'K760', 'K762', 'K763', 'K764', 'K765', 'K766', 'K767', 'K768', 'K769', 'Z944'],
        'PEPTIC_ULCER': ['K25', 'K26', 'K27', 'K28'],
        'AIDS_HIV': ['B20', 'B21', 'B22', 'B24'],
        'LYMPHOMA': ['C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C96', 'C900', 'C902'],
        'METASTATIC_CANCER': ['C77', 'C78', 'C79', 'C80'],
        'SOLID_TUMOR': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C97'],
        'RHEUMATOID_ARTHRITIS': ['M05', 'M06', 'M315', 'M32', 'M33', 'M34', 'M351', 'M353', 'M360'],
        'COAGULOPATHY': ['D65', 'D66', 'D67', 'D68', 'D691', 'D693', 'D694', 'D695', 'D696'],
        'OBESITY': ['E66'],
        'WEIGHT_LOSS': ['E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'R634', 'R64'],
        'FLUID_ELECTROLYTE': ['E222', 'E86', 'E87'],
        'BLOOD_LOSS_ANEMIA': ['D500'],
        'DEFICIENCY_ANEMIAS': ['D508', 'D509', 'D51', 'D52', 'D53'],
        'ALCOHOL_ABUSE': ['F10', 'E52', 'G621', 'I426', 'K292', 'K700', 'K703', 'K709', 'T510', 'T511', 'T512', 'T513', 'T518', 'T519', 'Z502', 'Z714', 'Z721'],
        'DRUG_ABUSE': ['F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19', 'Z715', 'Z722'],
        'PSYCHOSES': ['F20', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29', 'F302', 'F312', 'F315'],
        'DEPRESSION': ['F313', 'F314', 'F315', 'F32', 'F33', 'F341', 'F412', 'F432']
    }

    def code_in(icd_code, code_list, length):
        prefix = icd_code[:length]
        return prefix in code_list

    def assign_comorbidities(row):
        code = str(row['icd_code'])
        version = row['icd_version']
        flags = {}
        if version == 9:
            for comorb, prefixes in comorbidities_icd9.items():
                flags[comorb] = any(code_in(code, prefixes, len(p)) for p in prefixes)
        elif version == 10:
            for comorb, prefixes in comorbidities_icd10.items():
                flags[comorb] = any(code_in(code, prefixes, len(p)) for p in prefixes)
        else:
            for comorb in comorbidities_icd9:
                flags[comorb] = False
        return pd.Series(flags)

    print("Mapping comorbidities across diagnosis codes...")
    comorb_flags = diagnoses.apply(assign_comorbidities, axis=1)
    diagnoses_with_flags = pd.concat([diagnoses[['subject_id', 'hadm_id']], comorb_flags], axis=1)

    patient_comorbidities = diagnoses_with_flags.groupby(['subject_id', 'hadm_id']).max().reset_index()

    weights = {
        'CHF': 7, 'CARDIAC_ARRHYTHMIAS': 5, 'VALVULAR_DISEASE': -1, 'PULMONARY_CIRCULATION': 4,
        'PERIPHERAL_VASCULAR': 2, 'HYPERTENSION_UNCOMPLICATED': -1, 'HYPERTENSION_COMPLICATED': -1,
        'PARALYSIS': 7, 'OTHER_NEUROLOGICAL': 6, 'CHRONIC_PULMONARY': 3, 'DIABETES_UNCOMPLICATED': 0,
        'DIABETES_COMPLICATED': 0, 'HYPOTHYROIDISM': 0, 'RENAL_FAILURE': 5, 'LIVER_DISEASE': 11,
        'PEPTIC_ULCER': 0, 'AIDS_HIV': 0, 'LYMPHOMA': 9, 'METASTATIC_CANCER': 12, 'SOLID_TUMOR': 4,
        'RHEUMATOID_ARTHRITIS': 0, 'COAGULOPATHY': 3, 'OBESITY': -4, 'WEIGHT_LOSS': 6,
        'FLUID_ELECTROLYTE': 5, 'BLOOD_LOSS_ANEMIA': -2, 'DEFICIENCY_ANEMIAS': -2, 'ALCOHOL_ABUSE': 0,
        'DRUG_ABUSE': -7, 'PSYCHOSES': -4, 'DEPRESSION': -3
    }

    patient_comorbidities['elixhauser_score'] = 0
    for comorb, weight in weights.items():
        if comorb in patient_comorbidities.columns:
            patient_comorbidities['elixhauser_score'] += patient_comorbidities[comorb].astype(int) * weight

    elixhauser = pd.merge(admissions[['subject_id', 'hadm_id']], patient_comorbidities, on=['subject_id', 'hadm_id'], how='left')
    elixhauser['elixhauser_score'] = elixhauser['elixhauser_score'].fillna(0)

    output_file = os.path.join(output_path, "elixhauser_score.csv")
    elixhauser.to_csv(output_file, index=False)
    print(f"[patient_data] Elixhauser score saved to: {output_file}")
    return elixhauser


def create_patient_demographics(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Computes patient admission age, mortality flags, ICU readmissions,
    and merges Elixhauser comorbidity scores into patient_demographics.csv.
    """
    output_file = os.path.join(output_path, "patient_demographics.csv")
    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
        print(f"\n[patient_data] Reusing existing {output_file} ({os.path.getsize(output_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(output_file)

    drive_file = resolve_intermediate_path("patient_demographics.csv")
    if not force and os.path.exists(drive_file) and drive_file != output_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[patient_data] Loading pre-existing demographics from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(output_file, index=False)
        return df

    print("\n[patient_data] Creating demographics dataframe...")
    patients = pd.read_csv(
        os.path.join(mimic_path, "hosp/patients.csv.gz"),
        usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'],
        parse_dates=['dod']
    )
    admissions = pd.read_csv(
        os.path.join(mimic_path, "hosp/admissions.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
                 'admission_type', 'admission_location', 'discharge_location', 'insurance',
                 'language', 'marital_status', 'hospital_expire_flag'],
        parse_dates=['admittime', 'dischtime', 'deathtime']
    )
    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'],
        parse_dates=['intime', 'outtime']
    )

    elixhauser_file = resolve_intermediate_path("elixhauser_score.csv")
    elixhauser = pd.read_csv(elixhauser_file)

    # Join icustays, admissions, and patients
    base = icustays.merge(admissions, on=['hadm_id', 'subject_id'], how='inner')
    base = base.merge(patients, on='subject_id', how='inner')

    # Calculate first admittime and first admit age
    base['first_admittime'] = base.groupby('subject_id')['admittime'].transform('min')
    base['birth_year'] = base['anchor_year'] - base['anchor_age']

    # Age at ICU admission
    base['first_admit_age'] = base['intime'].dt.year - base['birth_year']
    base['first_admit_age'] = (base['first_admit_age'] - 0.5).round().astype(int)

    # Cap age > 89 at median 91.4 for deidentification
    base['first_admit_age'] = np.where(base['first_admit_age'] > 89, 91.4, base['first_admit_age'])

    # Mortality calculations using pd.Timedelta (matching notebook logic)
    base['ICUMort'] = np.where(
        (pd.notnull(base['dod'])) & (base['dod'] >= base['intime']) & (base['dod'] <= base['outtime']),
        1, 0
    )
    base['HospMort'] = base['hospital_expire_flag'].fillna(0).astype(int)
    base['HospMort28day'] = np.where(
        pd.notnull(base['dod']) & (base['dod'] <= base['admittime'] + pd.Timedelta(days=28)), 1, 0
    )
    base['HospMort90day'] = np.where(
        pd.notnull(base['dod']) & (base['dod'] <= base['admittime'] + pd.Timedelta(days=90)), 1, 0
    )
    base['HospMort1year'] = np.where(
        pd.notnull(base['dod']) & (base['dod'] <= base['admittime'] + pd.Timedelta(days=365)), 1, 0
    )

    # ICU readmission: 1 if >1 icustay per subject
    icu_counts = base.groupby('subject_id')['stay_id'].count().reset_index()
    icu_counts['ICU_readm'] = np.where(icu_counts['stay_id'] > 1, 1, 0)
    base = base.merge(icu_counts[['subject_id', 'ICU_readm']], on='subject_id', how='left')

    # Merge Elixhauser scores
    score_col = 'elixhauser_score' if 'elixhauser_score' in elixhauser.columns else (
        'elixhauser_vanwalraven' if 'elixhauser_vanwalraven' in elixhauser.columns else elixhauser.columns[-1]
    )
    elix_subset = elixhauser[['subject_id', 'hadm_id', score_col]].drop_duplicates(subset=['subject_id', 'hadm_id'])
    base = base.merge(elix_subset, on=['subject_id', 'hadm_id'], how='left')
    base['elixhauser_score'] = base[score_col].fillna(0)
    base['elixhauser_vanwalraven'] = base['elixhauser_score']

    # Keep both lower and upper casing for downstream compatibility
    base['icu_readm'] = base['ICU_readm']
    base['icumort'] = base['ICUMort']
    base['hospmort'] = base['HospMort']
    base['hospmort28day'] = base['HospMort28day']
    base['hospmort90day'] = base['HospMort90day']
    base['hospmort1year'] = base['HospMort1year']

    out_cols = [
        'subject_id', 'hadm_id', 'stay_id',
        'first_admit_age', 'gender', 'ICU_readm', 'icu_readm',
        'elixhauser_score', 'elixhauser_vanwalraven',
        'ICUMort', 'icumort', 'HospMort', 'hospmort',
        'HospMort28day', 'hospmort28day', 'HospMort90day', 'hospmort90day', 'hospmort1year',
        'dischtime', 'deathtime'
    ]
    out_cols = [c for c in out_cols if c in base.columns]
    merged = base[out_cols].sort_values(['subject_id', 'hadm_id', 'stay_id']).reset_index(drop=True)

    output_file = os.path.join(output_path, "patient_demographics.csv")
    merged.to_csv(output_file, index=False)
    print(f"[patient_data] Demographics saved to: {output_file}")
    return merged


def create_patient_echo_notes(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts echocardiogram findings from discharge and radiology notes into patient_echo_notes.csv.
    """
    out_file = os.path.join(output_path, "patient_echo_notes.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"\n[patient_data] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_echo = resolve_intermediate_path("patient_echo_notes.csv")
    if not force and os.path.exists(drive_echo) and drive_echo != out_file and os.path.getsize(drive_echo) > 1000:
        print(f"\n[patient_data] Loading pre-existing echo data from Drive: {drive_echo}")
        df = pd.read_csv(drive_echo)
        df.to_csv(out_file, index=False)
        return df

    print("\n[patient_data] Extracting echocardiogram data from notes...")
    note_dir = os.path.join(mimic_path, "mimic-iv-note/2.2/note")
    discharge_path = os.path.join(note_dir, "discharge.csv.gz")
    radiology_path = os.path.join(note_dir, "radiology.csv.gz")

    if os.path.exists(discharge_path) and os.path.exists(radiology_path):
        print("  Reading notes in chunks and filtering for 'echo'...")
        echo_chunks = []
        for path in [discharge_path, radiology_path]:
            for chunk in pd.read_csv(path, usecols=['subject_id', 'hadm_id', 'charttime', 'text'], chunksize=50000):
                matched = chunk[chunk['text'].str.contains('echo', case=False, na=False)].copy()
                if not matched.empty:
                    echo_chunks.append(matched)
        echo_notes = pd.concat(echo_chunks, ignore_index=True) if echo_chunks else pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime', 'text'])

        # Extract 'Indication'
        echo_notes['indication'] = echo_notes['text'].str.extract(r'Indication:\s*(.*?)\n', flags=re.IGNORECASE)[0]

        # Height
        height1 = echo_notes["text"].str.extract(r"Height:\s*\(in\)\s*([^\n\*]*)\n", flags=re.IGNORECASE)[0]
        height2 = echo_notes["text"].str.extract(r"height\s*[:=]?\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
        echo_notes["height"] = pd.to_numeric(height1, errors='coerce')
        missing_h = echo_notes["height"].isnull()
        echo_notes.loc[missing_h, "height"] = pd.to_numeric(height2[missing_h], errors='coerce')
        echo_notes.loc[echo_notes["height"].astype(str).str.contains(r'\*', na=False), "height"] = None

        # Weight
        weight1 = echo_notes["text"].str.extract(r"Weight\s*\(lb\):\s*([^\n\*]*)\n", flags=re.IGNORECASE)[0]
        weight2 = echo_notes["text"].str.extract(r"weight\s*[:=]\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
        echo_notes["weight"] = pd.to_numeric(weight1, errors='coerce')
        missing_w = echo_notes["weight"].isnull()
        echo_notes.loc[missing_w, "weight"] = pd.to_numeric(weight2[missing_w], errors='coerce')
        echo_notes.loc[echo_notes["weight"].astype(str).str.contains(r'\*', na=False), "weight"] = None

        # BSA
        bsa1 = echo_notes["text"].str.extract(r"BSA\s*\(m2\):\s*([^\s\*]+)", flags=re.IGNORECASE)[0]
        bsa2 = echo_notes["text"].str.extract(r"BSA\s*\(m2\)\s*:?=?\s*([0-9]+\.?\d*)", flags=re.IGNORECASE)[0]
        echo_notes["bsa"] = pd.to_numeric(bsa1, errors='coerce')
        missing_b = echo_notes["bsa"].isnull()
        echo_notes.loc[missing_b, "bsa"] = pd.to_numeric(bsa2[missing_b], errors='coerce')
        echo_notes.loc[echo_notes["bsa"].astype(str).str.contains(r'\*', na=False), "bsa"] = None

        # BP
        echo_notes["bp"] = echo_notes["text"].str.extract(r'BP\s*\(mm Hg\):\s*([^\n]*)', flags=re.IGNORECASE)[0]
        systolic = echo_notes["text"].str.extract(r'BP\s*\(mm Hg\):\s*([0-9]+)\s*/\s*[0-9]+\s*\n', flags=re.IGNORECASE)[0]
        diastolic = echo_notes["text"].str.extract(r'BP\s*\(mm Hg\):\s*[0-9]+\s*/\s*([0-9]+)\s*\n', flags=re.IGNORECASE)[0]
        echo_notes["bp_sys"] = pd.to_numeric(systolic, errors='coerce')
        echo_notes["bp_dias"] = pd.to_numeric(diastolic, errors='coerce')

        # HR
        hr1 = echo_notes["text"].str.extract(r'HR\s*\(bpm\):\s*([^\n\*]*)\n', flags=re.IGNORECASE)[0]
        echo_notes["hr"] = pd.to_numeric(hr1, errors='coerce')
        hr2 = echo_notes["text"].str.extract(r'HR\s*\(bpm\)?\s*[:=]?\s*([0-9]+\.?\d*)', flags=re.IGNORECASE)[0]
        missing_hr = echo_notes["hr"].isnull()
        echo_notes.loc[missing_hr, "hr"] = pd.to_numeric(hr2[missing_hr], errors='coerce')
        echo_notes.loc[echo_notes["hr"].astype(str).str.contains(r'\*', na=False), "hr"] = None

        # Descriptive fields
        echo_notes['status'] = echo_notes['text'].str.extract(r'Status:\s*(.*?)\n', flags=re.IGNORECASE)[0]
        echo_notes['tissue'] = echo_notes['text'].str.extract(r'Tissue:\s*(.*?)\n', flags=re.IGNORECASE)[0]
        echo_notes['doppler'] = echo_notes['text'].str.extract(r'Doppler:\s*(.*?)\n', flags=re.IGNORECASE)[0]
        echo_notes['contrast'] = echo_notes['text'].str.extract(r'Contrast:\s*(.*?)\n', flags=re.IGNORECASE)[0]
        echo_notes['technical_quality'] = echo_notes['text'].str.extract(r'Technical Quality:\s*(.*?)\n', flags=re.IGNORECASE)[0]

        echo_data = echo_notes[[
            'subject_id', 'hadm_id', 'charttime', 'indication', 'height', 'weight', 'bsa',
            'bp', 'bp_sys', 'bp_dias', 'hr', 'status', 'tissue', 'doppler', 'contrast', 'technical_quality'
        ]].copy()
    else:
        print(f"  Note files not found at {note_dir}. Checking for pre-existing echo data...")
        existing_echo = resolve_intermediate_path("patient_echo_notes.csv")
        echo_data = pd.read_csv(existing_echo)

    out_file = os.path.join(output_path, "patient_echo_notes.csv")
    echo_data.to_csv(out_file, index=False)
    print(f"[patient_data] Echo notes saved to: {out_file}")
    return echo_data


def create_patient_gcs(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Implements GCS extraction and imputation logic for MIMIC-IV chartevents into patient_gcs.csv.
    """
    out_file = os.path.join(output_path, "patient_gcs.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[patient_data] Reusing existing {out_file} ({os.path.getsize(output_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("patient_gcs.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[patient_data] Loading pre-existing GCS data from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[patient_data] Extracting Glasgow Coma Scale (GCS)...")
    ce_file = os.path.join(mimic_path, "icu/chartevents.csv.gz")

    metavision_ids = [223900, 223901, 220739]
    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        ce_file,
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[chunk['itemid'].isin(metavision_ids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        print("[patient_data] No GCS events found.")
        return pd.DataFrame()

    ce = pd.concat(chunks, ignore_index=True)

    # Pivot to GCSMotor, GCSVerbal, GCSEyes columns
    piv = ce.copy()
    piv['GCSMotor'] = np.where(piv['itemid'] == 223901, piv['valuenum'], np.nan)
    piv['GCSVerbal'] = np.where(piv['itemid'] == 223900, piv['valuenum'], np.nan)
    piv['GCSEyes'] = np.where(piv['itemid'] == 220739, piv['valuenum'], np.nan)
    piv['EndoTrachFlag'] = np.where(
        (piv['itemid'] == 223900) & (piv['value'] == 'No Response-ETT'), 1, 0
    )

    base = piv.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime']).agg({
        'GCSMotor': 'max',
        'GCSVerbal': 'max',
        'GCSEyes': 'max',
        'EndoTrachFlag': 'max'
    }).reset_index()

    base = base.sort_values(['subject_id', 'hadm_id', 'stay_id', 'charttime']).copy()

    # Lookback values within 6 hours
    base['GCSVerbalPrev'] = base.groupby('stay_id')['GCSVerbal'].shift(1)
    base['GCSMotorPrev'] = base.groupby('stay_id')['GCSMotor'].shift(1)
    base['GCSEyesPrev'] = base.groupby('stay_id')['GCSEyes'].shift(1)
    base['EndoTrachFlagPrev'] = base.groupby('stay_id')['EndoTrachFlag'].shift(1)
    base['charttimePrev'] = base.groupby('stay_id')['charttime'].shift(1)

    time_diff = base['charttime'] - base['charttimePrev']
    within_6h = time_diff < pd.Timedelta(hours=6)
    for c in ['GCSVerbalPrev', 'GCSMotorPrev', 'GCSEyesPrev', 'EndoTrachFlagPrev']:
        base.loc[~within_6h, c] = np.nan

    def compute_gcs(row):
        if pd.notnull(row['GCSVerbal']) and row['GCSVerbal'] == 0:
            return 15
        elif pd.isnull(row['GCSVerbal']) and pd.notnull(row['GCSVerbalPrev']) and row['GCSVerbalPrev'] == 0:
            return 15
        elif pd.notnull(row['GCSVerbalPrev']) and row['GCSVerbalPrev'] == 0:
            return (
                (row['GCSMotor'] if pd.notnull(row['GCSMotor']) else 6) +
                (row['GCSVerbal'] if pd.notnull(row['GCSVerbal']) else 5) +
                (row['GCSEyes'] if pd.notnull(row['GCSEyes']) else 4)
            )
        else:
            gm = row['GCSMotor'] if pd.notnull(row['GCSMotor']) else (
                 row['GCSMotorPrev'] if pd.notnull(row['GCSMotorPrev']) else 6)
            gv = row['GCSVerbal'] if pd.notnull(row['GCSVerbal']) else (
                 row['GCSVerbalPrev'] if pd.notnull(row['GCSVerbalPrev']) else 5)
            ge = row['GCSEyes'] if pd.notnull(row['GCSEyes']) else (
                 row['GCSEyesPrev'] if pd.notnull(row['GCSEyesPrev']) else 4)
            return gm + gv + ge

    base['GCS'] = base.apply(compute_gcs, axis=1)
    base['components_measured'] = (
        base[['GCSMotor', 'GCSVerbal', 'GCSEyes']].notnull().sum(axis=1) +
        base[['GCSMotorPrev', 'GCSVerbalPrev', 'GCSEyesPrev']].notnull().sum(axis=1)
    )
    base['GCSMotor_full'] = base['GCSMotor'].combine_first(base['GCSMotorPrev'])
    base['GCSVerbal_full'] = base['GCSVerbal'].combine_first(base['GCSVerbalPrev'])
    base['GCSEyes_full'] = base['GCSEyes'].combine_first(base['GCSEyesPrev'])
    base['EndoTrachFlag'] = base['EndoTrachFlag'].combine_first(base['EndoTrachFlagPrev'])

    base['priority'] = base['components_measured'] * 10 + (1 - base['EndoTrachFlag']) - base['GCS']
    base = base.sort_values(['subject_id', 'hadm_id', 'stay_id', 'charttime', 'priority'])

    final = base.groupby(['subject_id', 'hadm_id', 'stay_id', 'charttime'], as_index=False).first()

    results = final[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'GCS',
                     'GCSMotor_full', 'GCSVerbal_full', 'GCSEyes_full', 'EndoTrachFlag']]
    results = results.rename(columns={
        'GCSMotor_full': 'GCSMotor',
        'GCSVerbal_full': 'GCSVerbal',
        'GCSEyes_full': 'GCSEyes'
    })
    results = results.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    out_file = os.path.join(output_path, "patient_gcs.csv")
    results.to_csv(out_file, index=False)
    print(f"[patient_data] GCS data saved to: {out_file}")
    return results


def create_patient_vital_signs(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Retrieves vital signs (HR, SysBP, DiasBP, MeanBP, RespRate, TempC, SpO2) and merges GCS into patient_vital_signs.csv.
    """
    out_file = os.path.join(output_path, "patient_vital_signs.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[patient_data] Reusing existing {out_file} ({os.path.getsize(output_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("patient_vital_signs.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[patient_data] Loading pre-existing vital signs from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[patient_data] Extracting vital signs from chartevents...")
    vital_itemids = {
        'HeartRate': [220045],
        'SysBP': [220179, 220050],
        'DiasBP': [220180, 220051],
        'MeanBP': [220052, 220181, 225312],
        'RespRate': [220210, 224690],
        'TempC': [223761, 223762, 676, 678],
        'SpO2': [220277]
    }
    all_itemids = [item for sublist in vital_itemids.values() for item in sublist]

    chunk_size = 1000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'],
        parse_dates=['charttime'],
        chunksize=chunk_size
    ):
        filtered = chunk[chunk['itemid'].isin(all_itemids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        print("[patient_data] No vital signs data found.")
        return pd.DataFrame()

    chartevents = pd.concat(chunks, ignore_index=True)
    vitals = pd.DataFrame()
    vitals['subject_id'] = chartevents['subject_id']
    vitals['hadm_id'] = chartevents['hadm_id']
    vitals['stay_id'] = chartevents['stay_id']
    vitals['charttime'] = chartevents['charttime']

    # Heart Rate
    m = chartevents['itemid'].isin(vital_itemids['HeartRate']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] < 300)
    vitals.loc[m, 'HeartRate'] = chartevents.loc[m, 'valuenum']

    # Systolic BP
    m = chartevents['itemid'].isin(vital_itemids['SysBP']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] < 400)
    vitals.loc[m, 'SysBP'] = chartevents.loc[m, 'valuenum']

    # Diastolic BP
    m = chartevents['itemid'].isin(vital_itemids['DiasBP']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] < 300)
    vitals.loc[m, 'DiasBP'] = chartevents.loc[m, 'valuenum']

    # Mean BP
    m = chartevents['itemid'].isin(vital_itemids['MeanBP']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] < 300)
    vitals.loc[m, 'MeanBP'] = chartevents.loc[m, 'valuenum']

    # Resp Rate
    m = chartevents['itemid'].isin(vital_itemids['RespRate']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] < 70)
    vitals.loc[m, 'RespRate'] = chartevents.loc[m, 'valuenum']

    # Temp Fahrenheit to Celsius
    m_f = chartevents['itemid'].isin([223761, 678]) & (chartevents['valuenum'] > 70) & (chartevents['valuenum'] < 120)
    vitals.loc[m_f, 'TempC'] = (chartevents.loc[m_f, 'valuenum'] - 32) / 1.8

    # Temp Celsius
    m_c = chartevents['itemid'].isin([223762, 676]) & (chartevents['valuenum'] > 10) & (chartevents['valuenum'] < 50)
    vitals.loc[m_c, 'TempC'] = chartevents.loc[m_c, 'valuenum']

    # SpO2
    m = chartevents['itemid'].isin(vital_itemids['SpO2']) & (chartevents['valuenum'] > 0) & (chartevents['valuenum'] <= 100)
    vitals.loc[m, 'SpO2'] = chartevents.loc[m, 'valuenum']

    # Merge GCS if available
    gcs_path = resolve_intermediate_path("patient_gcs.csv")
    if os.path.exists(gcs_path):
        try:
            gcs = pd.read_csv(gcs_path, parse_dates=["charttime"])
            vitals = vitals.merge(
                gcs[["subject_id", "hadm_id", "stay_id", "charttime", "GCS"]],
                how="left",
                on=["subject_id", "hadm_id", "stay_id", "charttime"]
            )
            vitals.rename(columns={"GCS": "gcs"}, inplace=True)
        except Exception as e:
            print(f"Could not merge GCS: {e}")
            vitals['gcs'] = np.nan
    else:
        vitals['gcs'] = np.nan

    vital_signs = vitals.groupby(["subject_id", "hadm_id", "stay_id", "charttime"]).agg({
        'gcs': 'mean',
        'HeartRate': 'mean',
        'SysBP': 'mean',
        'DiasBP': 'mean',
        'MeanBP': 'mean',
        'RespRate': 'mean',
        'TempC': 'mean',
        'SpO2': 'mean',
    }).reset_index()

    out_file = os.path.join(output_path, "patient_vital_signs.csv")
    vital_signs.to_csv(out_file, index=False)
    print(f"[patient_data] Vital signs saved to: {out_file}")
    return vital_signs


def create_patient_weight(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts weights for ICU patients from chartevents and optional echo notes into patient_weight.csv.
    """
    out_file = os.path.join(output_path, "patient_weight.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[patient_data] Reusing existing {out_file} ({os.path.getsize(output_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("patient_weight.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[patient_data] Loading pre-existing weight data from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[patient_data] Calculating patient weights...")
    weight_itemids = [762, 226512, 763, 224639]
    chunk_size = 2000000
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic_path, "icu/chartevents.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'],
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
        columns=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum']
    )

    wt_stg['weight_type'] = wt_stg['itemid'].apply(lambda x: 'admit' if x in [762, 226512] else 'daily')
    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'],
        parse_dates=['intime', 'outtime']
    )

    wt_stg['rn'] = wt_stg.groupby(['stay_id', 'weight_type']).cumcount() + 1
    wt_stg2 = pd.merge(wt_stg, icustays, on=['subject_id', 'hadm_id', 'stay_id'], how='inner')

    wt_stg2['starttime'] = wt_stg2.apply(
        lambda row: row['intime'] - timedelta(hours=2) if row['weight_type'] == 'admit' and row['rn'] == 1 else row['charttime'],
        axis=1
    )
    wt_stg2 = wt_stg2[~((wt_stg2['weight_type'] == 'admit') & (wt_stg2['rn'] == 1))]

    wt_stg3 = wt_stg2.sort_values(['stay_id', 'starttime'])
    wt_stg3['endtime'] = wt_stg3.groupby(['stay_id'])['starttime'].shift(-1)
    wt_stg3['endtime'] = wt_stg3['endtime'].fillna(wt_stg3['outtime'] + timedelta(hours=2))

    wt_stg3 = wt_stg3[['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'valuenum']]
    wt_stg3.rename(columns={'valuenum': 'weight'}, inplace=True)

    # Try to augment with echo data
    echo_file = resolve_intermediate_path("echo_data.csv")
    if os.path.exists(echo_file):
        try:
            echo_data = pd.read_csv(echo_file, parse_dates=["charttime"])
            echo_weights = echo_data[['subject_id', 'hadm_id', 'charttime', 'weight']].dropna(subset=['weight'])
            echo_weights = pd.merge(echo_weights, icustays, on=['subject_id', 'hadm_id'], how='inner')
            missing_weights = set(icustays['stay_id']) - set(wt_stg3['stay_id'])
            echo_weights = echo_weights[echo_weights['stay_id'].isin(missing_weights)]
            echo_weights['weight'] = echo_weights['weight'] * 0.453592
            if not echo_weights.empty:
                echo_weights = echo_weights[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'weight']]
                echo_weights.rename(columns={'charttime': 'starttime'}, inplace=True)
                echo_weights['endtime'] = echo_weights['starttime'] + timedelta(hours=24)
                weight_data = pd.concat([wt_stg3, echo_weights[wt_stg3.columns]], ignore_index=True)
            else:
                weight_data = wt_stg3
        except Exception:
            weight_data = wt_stg3
    else:
        weight_data = wt_stg3

    weight_avg = weight_data.groupby(['subject_id', 'hadm_id', 'stay_id'])['weight'].mean().reset_index()
    out_file = os.path.join(output_path, "patient_weight.csv")
    weight_avg.to_csv(out_file, index=False)
    print(f"[patient_data] Weight data saved to: {out_file}")
    return weight_avg


def create_patient_icu_stay_times(mimic_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Extracts ICU in/out times and hospital mortality indicators into patient_icu_stay_times.csv.
    """
    out_file = os.path.join(output_path, "patient_icu_stay_times.csv")
    if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print(f"\n[patient_data] Reusing existing {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)...")
        return pd.read_csv(out_file)

    drive_file = resolve_intermediate_path("patient_icu_stay_times.csv")
    if not force and os.path.exists(drive_file) and drive_file != out_file and os.path.getsize(drive_file) > 100000:
        print(f"\n[patient_data] Loading pre-existing hospital mortality times from Drive: {drive_file}")
        df = pd.read_csv(drive_file)
        df.to_csv(out_file, index=False)
        return df

    print("\n[patient_data] Extracting hospital mortality and stay times...")
    patients = pd.read_csv(os.path.join(mimic_path, "hosp/patients.csv.gz"), usecols=['subject_id', 'dod'])
    admissions = pd.read_csv(
        os.path.join(mimic_path, "hosp/admissions.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']
    )
    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu/icustays.csv.gz"),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']
    )

    for df, col in [(patients, 'dod'), (admissions, 'admittime'), (admissions, 'dischtime'),
                    (admissions, 'deathtime'), (icustays, 'intime'), (icustays, 'outtime')]:
        df[col] = pd.to_datetime(df[col])

    hosp_mort = pd.merge(icustays, admissions, on=['subject_id', 'hadm_id'], how='inner')
    hosp_mort = pd.merge(hosp_mort, patients, on='subject_id', how='inner')

    hosp_mort_and_times = hosp_mort[[
        'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime',
        'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag', 'dod'
    ]]
    out_file = os.path.join(output_path, "patient_icu_stay_times.csv")
    hosp_mort_and_times.to_csv(out_file, index=False)
    print(f"[patient_data] Hospital mortality & times saved to: {out_file}")
    return hosp_mort_and_times


# Backward compatibility aliases
create_demographics = create_patient_demographics
create_demographics2 = create_patient_demographics
create_echo_data = create_patient_echo_notes
create_echo_data2 = create_patient_echo_notes
create_gcs = create_patient_gcs
create_GCS2 = create_patient_gcs
create_vital_signs = create_patient_vital_signs
create_vital_signs2 = create_patient_vital_signs
create_weight = create_patient_weight
create_icu_stay_times = create_patient_icu_stay_times
create_hosp_mort_and_in_out_times = create_patient_icu_stay_times


def run_all(mimic_path: str = MIMIC_PATH, output_path: str = str(LOCAL_DATA_DIR), force: bool = False):
    """Runs all patient data extraction functions in dependency order."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting patient_data extraction:\n  Drive Source: {mimic_path}\n  Local Output: {output_path}\n  Force Recompute: {force}")
    create_elixhauser_score(mimic_path, output_path, force=force)
    create_patient_demographics(mimic_path, output_path, force=force)
    create_patient_echo_notes(mimic_path, output_path, force=force)
    create_patient_gcs(mimic_path, output_path, force=force)
    create_patient_vital_signs(mimic_path, output_path, force=force)
    create_patient_weight(mimic_path, output_path, force=force)
    create_patient_icu_stay_times(mimic_path, output_path, force=force)
    print("\n[patient_data] All patient data processing finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Patient Data")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all tables")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, force=args.force)
