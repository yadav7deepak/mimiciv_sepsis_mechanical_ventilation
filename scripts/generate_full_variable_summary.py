"""
Generates comprehensive distribution summary statistics for all variables (~52 clinical variables)
across both the unfiltered Full Master Dataset (final_patient_dataset_master.csv)
and the MDP Research Cohort (final_patient_dataset_mdp.csv).

Includes:
- Categorical variables (Gender: Male/Female counts & percentages)
- Continuous variables: Valid count, Missing count, Missing %, Min, P25, Median, P75, Max, IQR
- Raw Mean and Raw Standard Deviation
- Capping and Flooring (1st and 99th percentiles / Winsorization) to eliminate extreme chart typos
- Capped Mean and Capped Standard Deviation

Saves:
1. data/patient_variables_summary_all.csv (Combined long format with capping)
2. data/patient_variables_summary_comparison.csv (Side-by-side Table 1 comparison with raw and capped values)
3. data/patient_vitals_labs_distribution.csv (Updated canonical distribution CSV)
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    from .config import LOCAL_DATA_DIR, resolve_intermediate_path
except ImportError:
    from config import LOCAL_DATA_DIR, resolve_intermediate_path


VARIABLES_CATALOG = [
    # Demographics & Comorbidity
    ("Demographics", "gender", "Patient Gender (M/F)", "categorical"),
    ("Demographics", "first_admit_age", "Age at First Admission (years)", "continuous"),
    ("Demographics", "weight", "Patient Weight (kg)", "continuous"),
    ("Demographics", "ICU_readm", "ICU Readmission Flag (0/1)", "binary"),
    ("Demographics", "elixhauser_score", "Elixhauser Comorbidity Score", "continuous"),
    ("Demographics", "HospMort90day", "90-Day In-Hospital Mortality Flag (0/1)", "binary"),

    # Severity & Neurological Scores
    ("Severity Scores", "SOFA", "Sequential Organ Failure Assessment Score", "continuous"),
    ("Severity Scores", "SIRS", "Systemic Inflammatory Response Syndrome Score", "continuous"),
    ("Severity Scores", "gcs", "Glasgow Coma Scale Total", "continuous"),

    # Vital Signs
    ("Vital Signs", "HeartRate", "Heart Rate (bpm)", "continuous"),
    ("Vital Signs", "SysBP", "Systolic Blood Pressure (mmHg)", "continuous"),
    ("Vital Signs", "DiasBP", "Diastolic Blood Pressure (mmHg)", "continuous"),
    ("Vital Signs", "MeanBP", "Mean Arterial Pressure (mmHg)", "continuous"),
    ("Vital Signs", "shockindex", "Shock Index (SysBP / HeartRate)", "continuous"),
    ("Vital Signs", "RespRate", "Respiratory Rate (bpm)", "continuous"),
    ("Vital Signs", "TempC", "Body Temperature (°C)", "continuous"),
    ("Vital Signs", "SpO2", "Pulse Oximetry Saturation (%)", "continuous"),

    # Arterial Blood Gas & Acid-Base
    ("Blood Gas & Acid-Base", "PH", "Arterial Blood Gas pH", "continuous"),
    ("Blood Gas & Acid-Base", "PAO2", "Partial Pressure of Oxygen (PaO2, mmHg)", "continuous"),
    ("Blood Gas & Acid-Base", "PACO2", "Partial Pressure of Carbon Dioxide (PaCO2, mmHg)", "continuous"),
    ("Blood Gas & Acid-Base", "BASE_EXCESS", "Base Excess (mEq/L)", "continuous"),
    ("Blood Gas & Acid-Base", "BICARBONATE", "Serum Bicarbonate (mEq/L)", "continuous"),
    ("Blood Gas & Acid-Base", "LACTATE", "Serum Lactate (mmol/L)", "continuous"),
    ("Blood Gas & Acid-Base", "PAO2FiO2ratio", "PaO2 / FiO2 Ratio (Horowitz Index)", "continuous"),

    # Electrolytes & Chemistries
    ("Electrolytes & Chemistries", "POTASSIUM", "Serum Potassium (mEq/L)", "continuous"),
    ("Electrolytes & Chemistries", "SODIUM", "Serum Sodium (mEq/L)", "continuous"),
    ("Electrolytes & Chemistries", "CHLORIDE", "Serum Chloride (mEq/L)", "continuous"),
    ("Electrolytes & Chemistries", "GLUCOSE", "Serum Glucose (mg/dL)", "continuous"),
    ("Electrolytes & Chemistries", "BUN", "Blood Urea Nitrogen (mg/dL)", "continuous"),
    ("Electrolytes & Chemistries", "CREATININE", "Serum Creatinine (mg/dL)", "continuous"),
    ("Electrolytes & Chemistries", "MAGNESIUM", "Serum Magnesium (mg/dL)", "continuous"),
    ("Electrolytes & Chemistries", "CALCIUM", "Serum Calcium (mg/dL)", "continuous"),
    ("Electrolytes & Chemistries", "CARBONDIOXIDE", "Total Carbon Dioxide (mEq/L)", "continuous"),

    # Liver Function Tests & Proteins
    ("Liver & Proteins", "SGOT", "Aspartate Aminotransferase (AST/SGOT, IU/L)", "continuous"),
    ("Liver & Proteins", "SGPT", "Alanine Aminotransferase (ALT/SGPT, IU/L)", "continuous"),
    ("Liver & Proteins", "BILIRUBIN", "Total Bilirubin (mg/dL)", "continuous"),
    ("Liver & Proteins", "ALBUMIN", "Serum Albumin (g/dL)", "continuous"),

    # Hematology & Coagulation
    ("Hematology & Coagulation", "HEMOGLOBIN", "Hemoglobin (g/dL)", "continuous"),
    ("Hematology & Coagulation", "WBC", "White Blood Cell Count (10^9/L)", "continuous"),
    ("Hematology & Coagulation", "PLATELET", "Platelet Count (10^9/L)", "continuous"),
    ("Hematology & Coagulation", "PTT", "Partial Thromboplastin Time (seconds)", "continuous"),
    ("Hematology & Coagulation", "PT", "Prothrombin Time (seconds)", "continuous"),
    ("Hematology & Coagulation", "INR", "International Normalized Ratio (INR)", "continuous"),

    # Mechanical Ventilation & Respiratory Parameters
    ("Mechanical Ventilation", "MechVent", "Mechanical Ventilation Active (0/1)", "binary"),
    ("Mechanical Ventilation", "FiO2", "Fraction of Inspired Oxygen (%)", "continuous"),
    ("Mechanical Ventilation", "PEEP", "Positive End-Expiratory Pressure (cmH2O)", "continuous"),
    ("Mechanical Ventilation", "tidal_volume", "Exhaled Tidal Volume (mL)", "continuous"),
    ("Mechanical Ventilation", "plateau_pressure", "Plateau Pressure (cmH2O)", "continuous"),

    # Fluid Management & Vasopressors
    ("Fluids & Vasopressors", "urineoutput", "Urine Output Volume (mL/4h)", "continuous"),
    ("Fluids & Vasopressors", "iv_total", "Intravenous Fluid Intake (mL/4h)", "continuous"),
    ("Fluids & Vasopressors", "cum_fluid_balance", "Cumulative Fluid Balance (mL)", "continuous"),
    ("Fluids & Vasopressors", "vaso_total", "Total Vasopressor Equivalent Rate", "continuous"),

    # MDP & RL Specific
    ("RL State & Actions", "cluster_label", "K-Means State Cluster Label (0-499)", "continuous"),
    ("RL State & Actions", "PEEP_binned", "PEEP Action Quintile Bin (0-4)", "continuous"),
    ("RL State & Actions", "FiO2_binned", "FiO2 Action Quintile Bin (0-4)", "continuous"),
    ("RL State & Actions", "tidal_volume_binned", "Tidal Volume Action Quintile Bin (0-4)", "continuous"),
    ("RL State & Actions", "RespRate_binned", "Resp Rate Action Quintile Bin (0-4)", "continuous"),
    ("RL State & Actions", "subject_has_deathtime", "Mortality Outcome Indicator (-1/+1 Reward)", "binary"),
]


def summarize_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Computes summary statistics with capping & flooring (1st and 99th percentiles)."""
    records = []
    total_n = len(df)

    for category, var_name, label, vtype in VARIABLES_CATALOG:
        matched_col = None
        for col_cand in [var_name, var_name.lower(), var_name.upper()]:
            if col_cand in df.columns:
                matched_col = col_cand
                break

        if matched_col is None:
            continue

        raw_series = df[matched_col]

        # Handle Categorical (Gender)
        if vtype == "categorical" or var_name == "gender":
            valid_cnt = int(raw_series.notna().sum())
            miss_cnt = int(total_n - valid_cnt)
            miss_pct = round((miss_cnt / total_n) * 100.0, 2)

            m_cnt = int((raw_series == "M").sum())
            f_cnt = int((raw_series == "F").sum())
            m_pct = round((m_cnt / valid_cnt) * 100.0, 2) if valid_cnt > 0 else 0.0
            f_pct = round((f_cnt / valid_cnt) * 100.0, 2) if valid_cnt > 0 else 0.0

            records.append({
                "Dataset": dataset_name,
                "Category": category,
                "Variable": var_name,
                "Description": label,
                "Total_Rows": total_n,
                "Valid_Count": valid_cnt,
                "Missing_Count": miss_cnt,
                "Missing_Percent": miss_pct,
                "Raw_Mean": np.nan,
                "Raw_Std": np.nan,
                "Min": np.nan,
                "P25": np.nan,
                "Median": np.nan,
                "P75": np.nan,
                "Max": np.nan,
                "IQR": np.nan,
                "Floor_P1": np.nan,
                "Cap_P99": np.nan,
                "Capped_Mean": np.nan,
                "Capped_Std": np.nan,
                "Display_Summary": f"Male: {m_cnt:,} ({m_pct}%), Female: {f_cnt:,} ({f_pct}%)"
            })
            continue

        # Continuous or Binary Numeric variables
        series = pd.to_numeric(raw_series, errors='coerce')
        valid_cnt = int(series.count())
        miss_cnt = int(total_n - valid_cnt)
        miss_pct = round((miss_cnt / total_n) * 100.0, 2) if total_n > 0 else 0.0

        if valid_cnt > 0:
            raw_mean = round(float(series.mean()), 2)
            raw_std = round(float(series.std()), 2)
            min_val = round(float(series.min()), 2)
            p25_val = round(float(series.quantile(0.25)), 2)
            median_val = round(float(series.median()), 2)
            p75_val = round(float(series.quantile(0.75)), 2)
            max_val = round(float(series.max()), 2)
            iqr_val = round(float(p75_val - p25_val), 2)

            # 1st and 99th percentile capping & flooring (Winsorization)
            if vtype == "continuous":
                floor_p1 = round(float(series.quantile(0.01)), 2)
                cap_p99 = round(float(series.quantile(0.99)), 2)
                series_capped = series.clip(lower=floor_p1, upper=cap_p99)
                capped_mean = round(float(series_capped.mean()), 2)
                capped_std = round(float(series_capped.std()), 2)
            else:
                # Binary flags don't need percentile clipping
                floor_p1 = min_val
                cap_p99 = max_val
                capped_mean = raw_mean
                capped_std = raw_std

            disp_summary = f"{capped_mean} ± {capped_std} (Med: {median_val} [{p25_val}-{p75_val}])"
        else:
            raw_mean = raw_std = min_val = p25_val = median_val = p75_val = max_val = iqr_val = np.nan
            floor_p1 = cap_p99 = capped_mean = capped_std = np.nan
            disp_summary = "-"

        records.append({
            "Dataset": dataset_name,
            "Category": category,
            "Variable": var_name,
            "Description": label,
            "Total_Rows": total_n,
            "Valid_Count": valid_cnt,
            "Missing_Count": miss_cnt,
            "Missing_Percent": miss_pct,
            "Raw_Mean": raw_mean,
            "Raw_Std": raw_std,
            "Min": min_val,
            "P25": p25_val,
            "Median": median_val,
            "P75": p75_val,
            "Max": max_val,
            "IQR": iqr_val,
            "Floor_P1": floor_p1,
            "Cap_P99": cap_p99,
            "Capped_Mean": capped_mean,
            "Capped_Std": capped_std,
            "Display_Summary": disp_summary
        })

    return pd.DataFrame(records)


def run_summary(output_dir: str = str(LOCAL_DATA_DIR)):
    master_path = resolve_intermediate_path("final_patient_dataset_master.csv")
    mdp_path = resolve_intermediate_path("final_patient_dataset_mdp.csv")

    print(f"Loading Master Dataset: {master_path}")
    df_master = pd.read_csv(master_path)
    print(f"  Master dataset loaded: {df_master.shape[0]:,} rows x {df_master.shape[1]} columns")

    print(f"Loading MDP Clustered Dataset: {mdp_path}")
    df_mdp = pd.read_csv(mdp_path)
    print(f"  MDP dataset loaded: {df_mdp.shape[0]:,} rows x {df_mdp.shape[1]} columns")

    print("\nComputing statistics with Capping & Flooring for Full Master Dataset (Unfiltered)...")
    master_summary = summarize_dataset(df_master, "Full_Dataset_Unfiltered")

    print("Computing statistics with Capping & Flooring for MDP Research Cohort...")
    mdp_summary = summarize_dataset(df_mdp, "MDP_Cohort")

    # 1. Combined Long Table
    df_all = pd.concat([master_summary, mdp_summary], ignore_index=True)
    out_all_csv = os.path.join(output_dir, "patient_variables_summary_all.csv")
    df_all.to_csv(out_all_csv, index=False)
    print(f"\n[1] Saved comprehensive variable summary (long format) to:\n    {out_all_csv}")

    # Canonical distribution CSV
    canonical_dist_csv = os.path.join(output_dir, "patient_vitals_labs_distribution.csv")
    df_all.to_csv(canonical_dist_csv, index=False)
    print(f"[2] Updated canonical distribution CSV:\n    {canonical_dist_csv}")

    # 2. Side-by-side Table 1 comparison
    comp_records = []
    master_dict = {r["Variable"]: r for _, r in master_summary.iterrows()}
    mdp_dict = {r["Variable"]: r for _, r in mdp_summary.iterrows()}

    all_vars = []
    seen = set()
    for _, var_name, _, _ in VARIABLES_CATALOG:
        if var_name not in seen:
            seen.add(var_name)
            all_vars.append(var_name)

    for var in all_vars:
        m_row = master_dict.get(var)
        mdp_row = mdp_dict.get(var)

        cat = m_row["Category"] if m_row is not None else (mdp_row["Category"] if mdp_row is not None else "Other")
        desc = m_row["Description"] if m_row is not None else (mdp_row["Description"] if mdp_row is not None else var)

        if var == "gender":
            comp_records.append({
                "Category": cat,
                "Variable": var,
                "Description": desc,
                "Full_Missing_Pct": m_row["Missing_Percent"] if m_row is not None else np.nan,
                "Full_Raw_Mean_Std": "-",
                "Full_Capped_Mean_Std": "-",
                "Full_Median_IQR": m_row["Display_Summary"] if m_row is not None else "-",
                "Full_Min_Max": "-",
                "MDP_Missing_Pct": mdp_row["Missing_Percent"] if mdp_row is not None else np.nan,
                "MDP_Raw_Mean_Std": "-",
                "MDP_Capped_Mean_Std": "-",
                "MDP_Median_IQR": mdp_row["Display_Summary"] if mdp_row is not None else "-",
                "MDP_Min_Max": "-"
            })
            continue

        comp_records.append({
            "Category": cat,
            "Variable": var,
            "Description": desc,

            # Full Master Dataset (Unfiltered)
            "Full_Missing_Pct": m_row["Missing_Percent"] if m_row is not None else np.nan,
            "Full_Raw_Mean_Std": f"{m_row['Raw_Mean']} ± {m_row['Raw_Std']}" if m_row is not None and pd.notna(m_row['Raw_Mean']) else "-",
            "Full_Capped_Mean_Std": f"{m_row['Capped_Mean']} ± {m_row['Capped_Std']} [P1: {m_row['Floor_P1']} | P99: {m_row['Cap_P99']}]" if m_row is not None and pd.notna(m_row['Capped_Mean']) else "-",
            "Full_Median_IQR": f"{m_row['Median']} [{m_row['P25']} - {m_row['P75']}]" if m_row is not None and pd.notna(m_row['Median']) else "-",
            "Full_Min_Max": f"{m_row['Min']} to {m_row['Max']}" if m_row is not None and pd.notna(m_row['Min']) else "-",

            # MDP Cohort
            "MDP_Missing_Pct": mdp_row["Missing_Percent"] if mdp_row is not None else np.nan,
            "MDP_Raw_Mean_Std": f"{mdp_row['Raw_Mean']} ± {mdp_row['Raw_Std']}" if mdp_row is not None and pd.notna(mdp_row['Raw_Mean']) else "-",
            "MDP_Capped_Mean_Std": f"{mdp_row['Capped_Mean']} ± {mdp_row['Capped_Std']} [P1: {mdp_row['Floor_P1']} | P99: {mdp_row['Cap_P99']}]" if mdp_row is not None and pd.notna(mdp_row['Capped_Mean']) else "-",
            "MDP_Median_IQR": f"{mdp_row['Median']} [{mdp_row['P25']} - {mdp_row['P75']}]" if mdp_row is not None and pd.notna(mdp_row['Median']) else "-",
            "MDP_Min_Max": f"{mdp_row['Min']} to {mdp_row['Max']}" if mdp_row is not None and pd.notna(mdp_row['Min']) else "-",
        })

    df_comp = pd.DataFrame(comp_records)
    out_comp_csv = os.path.join(output_dir, "patient_variables_summary_comparison.csv")
    df_comp.to_csv(out_comp_csv, index=False)
    print(f"[3] Saved side-by-side Table 1 comparison with capping & flooring to:\n    {out_comp_csv}")

    print("\nEnhanced summary generation finished successfully.")
    return df_all, df_comp


if __name__ == "__main__":
    run_summary()
