"""
Patient Cohort Data Distribution Analysis Script.
Analyzes demographics, clinical vitals, laboratory tests, severity scores,
mechanical ventilation parameters, fluid balance, and MDP state-action spaces
for both the master 4-hour sampled dataset and the preprocessed/clustered MDP dataset.
Generates CSV distribution tables and multi-panel high-resolution figures.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .config import LOCAL_DATA_DIR, resolve_intermediate_path
except ImportError:
    from config import LOCAL_DATA_DIR, resolve_intermediate_path


def compute_distribution_table(df: pd.DataFrame, columns: list, dataset_name: str) -> pd.DataFrame:
    """Computes distribution statistics (count, mean, std, median, 25%, 75%, min, max, % missing)."""
    rows = []
    total_len = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        valid_count = series.count()
        missing_count = total_len - valid_count
        missing_pct = (missing_count / total_len) * 100 if total_len > 0 else 0.0

        if valid_count > 0:
            mean_val = series.mean()
            std_val = series.std()
            q25 = series.quantile(0.25)
            median_val = series.median()
            q75 = series.quantile(0.75)
            min_val = series.min()
            max_val = series.max()
        else:
            mean_val = std_val = q25 = median_val = q75 = min_val = max_val = np.nan

        rows.append({
            'dataset': dataset_name,
            'variable': col,
            'total_rows': total_len,
            'valid_count': valid_count,
            'missing_count': missing_count,
            'missing_percent': round(missing_pct, 2),
            'mean': round(mean_val, 2) if pd.notna(mean_val) else np.nan,
            'std': round(std_val, 2) if pd.notna(std_val) else np.nan,
            'p25': round(q25, 2) if pd.notna(q25) else np.nan,
            'median': round(median_val, 2) if pd.notna(median_val) else np.nan,
            'p75': round(q75, 2) if pd.notna(q75) else np.nan,
            'min': round(min_val, 2) if pd.notna(min_val) else np.nan,
            'max': round(max_val, 2) if pd.notna(max_val) else np.nan,
        })
    return pd.DataFrame(rows)


def analyze_cohort_distributions(output_dir: str = str(LOCAL_DATA_DIR)):
    """Computes and saves clinical distribution metrics and plots."""
    master_path = resolve_intermediate_path("final_patient_dataset_master.csv")
    mdp_path = resolve_intermediate_path("final_patient_dataset_mdp.csv")

    print(f"Loading master patient dataset: {master_path}")
    df_master = pd.read_csv(master_path)
    print(f"  Master dataset loaded: {df_master.shape[0]:,} rows x {df_master.shape[1]} columns")

    print(f"Loading MDP clustered dataset: {mdp_path}")
    df_mdp = pd.read_csv(mdp_path)
    print(f"  MDP dataset loaded: {df_mdp.shape[0]:,} rows x {df_mdp.shape[1]} columns")

    # 1. High-level cohort summary
    summary_data = []

    # Master Cohort
    m_stays = df_master['stay_id'].nunique()
    m_subjects = df_master['subject_id'].nunique()
    m_admissions = df_master['hadm_id'].nunique()
    m_mort_col = 'HospMort90day' if 'HospMort90day' in df_master.columns else 'hospmort90day'
    m_mort = df_master.groupby('subject_id')[m_mort_col].max().mean() * 100 if m_mort_col in df_master.columns else np.nan
    m_male_pct = (df_master.groupby('subject_id')['gender'].first() == 'M').mean() * 100 if 'gender' in df_master.columns else np.nan
    m_age_mean = df_master.groupby('subject_id')['first_admit_age'].first().mean() if 'first_admit_age' in df_master.columns else np.nan

    summary_data.append({
        'Cohort': 'Master Cohort (All 4h Sampled)',
        'Total_Timesteps': len(df_master),
        'Unique_Patients': m_subjects,
        'Unique_Admissions': m_admissions,
        'Unique_ICU_Stays': m_stays,
        'Mean_Age_Years': round(m_age_mean, 2),
        'Male_Percent': round(m_male_pct, 2),
        '90Day_Mortality_Percent': round(m_mort, 2),
    })

    # MDP Cohort
    mdp_stays = df_mdp['stay_id'].nunique()
    mdp_subjects = df_mdp['subject_id'].nunique()
    mdp_admissions = df_mdp['hadm_id'].nunique()
    mdp_mort_col = 'subject_has_deathtime' if 'subject_has_deathtime' in df_mdp.columns else ('HospMort90day' if 'HospMort90day' in df_mdp.columns else 'hospmort90day')
    mdp_mort = df_mdp.groupby('subject_id')[mdp_mort_col].max().mean() * 100 if mdp_mort_col in df_mdp.columns else np.nan
    mdp_male_pct = (df_mdp.groupby('subject_id')['gender'].first() == 'M').mean() * 100 if 'gender' in df_mdp.columns else np.nan
    mdp_age_mean = df_mdp.groupby('subject_id')['first_admit_age'].first().mean() if 'first_admit_age' in df_mdp.columns else np.nan

    summary_data.append({
        'Cohort': 'MDP Cohort (Adults, >=24h Vent, Imputed, Clustered)',
        'Total_Timesteps': len(df_mdp),
        'Unique_Patients': mdp_subjects,
        'Unique_Admissions': mdp_admissions,
        'Unique_ICU_Stays': mdp_stays,
        'Mean_Age_Years': round(mdp_age_mean, 2),
        'Male_Percent': round(mdp_male_pct, 2),
        '90Day_Mortality_Percent': round(mdp_mort, 2),
    })

    df_summary = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, "patient_cohort_distribution_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nCohort Overview:\n{df_summary.to_string(index=False)}")

    # 2. Detailed clinical variables distribution
    clinical_vars = [
        # Demographics & Comorbidities
        'first_admit_age', 'weight', 'elixhauser_score', 'SOFA', 'SIRS', 'gcs',
        # Vitals
        'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'shockindex', 'RespRate', 'TempC', 'SpO2',
        # Gas & Acid-Base
        'PH', 'PAO2', 'PACO2', 'BASE_EXCESS', 'BICARBONATE', 'LACTATE', 'PAO2FiO2ratio',
        # Electrolytes & Chemistries
        'POTASSIUM', 'SODIUM', 'CHLORIDE', 'GLUCOSE', 'BUN', 'CREATININE', 'MAGNESIUM', 'CALCIUM',
        # CBC & Coagulation
        'HEMOGLOBIN', 'WBC', 'PLATELET', 'PTT', 'PT', 'INR',
        # Respiratory & Ventilation Interventions
        'MechVent', 'FiO2', 'PEEP', 'tidal_volume', 'plateau_pressure',
        # Fluid & Vasopressors
        'urineoutput', 'iv_total', 'cum_fluid_balance', 'vaso_total'
    ]

    dist_master = compute_distribution_table(df_master, clinical_vars, "Master_Cohort")
    dist_mdp = compute_distribution_table(df_mdp, clinical_vars, "MDP_Cohort")
    dist_combined = pd.concat([dist_master, dist_mdp], ignore_index=True)

    dist_csv = os.path.join(output_dir, "patient_vitals_labs_distribution.csv")
    dist_combined.to_csv(dist_csv, index=False)
    print(f"\nSaved detailed distribution to: {dist_csv}")

    # 3. Create high-resolution multi-panel distribution figure
    print("\nGenerating publication-quality distribution plots...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), dpi=200)

    # 1. Age Distribution (Patient level)
    ax1 = axes[0, 0]
    m_ages = df_master.groupby('subject_id')['first_admit_age'].first().dropna()
    mdp_ages = df_mdp.groupby('subject_id')['first_admit_age'].first().dropna()
    sns.histplot(m_ages, kde=True, color='skyblue', label='Master Cohort', ax=ax1, stat='density', alpha=0.5, bins=30)
    sns.histplot(mdp_ages, kde=True, color='darkorange', label='MDP Cohort (Adult)', ax=ax1, stat='density', alpha=0.5, bins=30)
    ax1.set_title('A. Patient Age at First Admission', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Density')
    ax1.legend()

    # 2. Gender & Mortality
    ax2 = axes[0, 1]
    mort_data = pd.DataFrame({
        'Cohort': ['Master Cohort', 'MDP Cohort'],
        'Mortality %': [m_mort, mdp_mort]
    })
    bars = sns.barplot(x='Cohort', y='Mortality %', data=mort_data, ax=ax2, palette=['skyblue', 'coral'])
    ax2.set_title('B. 90-Day Hospital Mortality Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mortality (%)')
    ax2.set_ylim(0, max(m_mort, mdp_mort) * 1.35)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.0),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. ICU Stay Duration (hours)
    ax3 = axes[0, 2]
    # Calculate stay duration per stay_id
    if 'start_time' in df_mdp.columns:
        df_mdp_time = df_mdp.copy()
        df_mdp_time['start_time'] = pd.to_datetime(df_mdp_time['start_time'])
        stay_hours = df_mdp_time.groupby('stay_id')['start_time'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600.0)
        stay_hours = stay_hours[stay_hours <= 500]  # trim extreme outliers for plot
        sns.histplot(stay_hours, kde=True, color='teal', ax=ax3, bins=35)
        ax3.set_title('C. MDP ICU Stay Duration Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Duration (hours, capped at 500h)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(stay_hours.median(), color='red', linestyle='--', label=f'Median: {stay_hours.median():.0f}h')
        ax3.legend()

    # 4. SOFA Score Distribution
    ax4 = axes[1, 0]
    sns.countplot(x='SOFA', data=df_mdp[df_mdp['SOFA'] <= 18], ax=ax4, color='royalblue')
    ax4.set_title('D. Sequential Organ Failure Assessment (SOFA)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('SOFA Score')
    ax4.set_ylabel('Timestep Count')

    # 5. SIRS Score Distribution
    ax5 = axes[1, 1]
    sns.countplot(x='SIRS', data=df_mdp, ax=ax5, color='cornflowerblue')
    ax5.set_title('E. Systemic Inflammatory Response (SIRS)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('SIRS Score')
    ax5.set_ylabel('Timestep Count')

    # 6. Mean Arterial Pressure (MAP) & Shock Index
    ax6 = axes[1, 2]
    map_vals = df_mdp['MeanBP'].dropna()
    map_vals = map_vals[(map_vals >= 30) & (map_vals <= 150)]
    sns.histplot(map_vals, kde=True, color='mediumseagreen', ax=ax6, bins=40)
    ax6.axvline(65, color='darkred', linestyle='--', linewidth=2, label='Target MAP (65 mmHg)')
    ax6.set_title('F. Mean Arterial Pressure (MeanBP)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('MeanBP (mmHg)')
    ax6.set_ylabel('Count')
    ax6.legend()

    # 7. PaO2 / FiO2 Ratio (Horowitz index of ARDS severity)
    ax7 = axes[2, 0]
    pao2_fio2 = df_mdp['PAO2FiO2ratio'].dropna()
    pao2_fio2 = pao2_fio2[(pao2_fio2 >= 20) & (pao2_fio2 <= 600)]
    sns.histplot(pao2_fio2, kde=True, color='purple', ax=ax7, bins=40)
    ax7.axvline(300, color='gold', linestyle='--', label='Mild ARDS (<=300)')
    ax7.axvline(200, color='orange', linestyle='--', label='Moderate ARDS (<=200)')
    ax7.axvline(100, color='red', linestyle='--', label='Severe ARDS (<=100)')
    ax7.set_title('G. PaO2 / FiO2 Ratio (ARDS Severity)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('PaO2 / FiO2 Ratio')
    ax7.set_ylabel('Count')
    ax7.legend(loc='upper right', fontsize=9)

    # 8. PEEP Distribution
    ax8 = axes[2, 1]
    peep_vals = df_mdp['PEEP'].dropna()
    peep_vals = peep_vals[(peep_vals >= 0) & (peep_vals <= 25)]
    sns.histplot(peep_vals, kde=False, color='steelblue', ax=ax8, bins=26)
    ax8.set_title('H. Positive End-Expiratory Pressure (PEEP)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('PEEP (cmH2O)')
    ax8.set_ylabel('Count')

    # 9. FiO2 Distribution
    ax9 = axes[2, 2]
    fio2_vals = df_mdp['FiO2'].dropna()
    if fio2_vals.max() > 1.5:
        fio2_vals = fio2_vals / 100.0
    fio2_vals = fio2_vals[(fio2_vals >= 0.2) & (fio2_vals <= 1.0)]
    sns.histplot(fio2_vals, kde=True, color='indianred', ax=ax9, bins=25)
    ax9.set_title('I. Fraction of Inspired Oxygen (FiO2)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('FiO2 (fraction: 0.21 - 1.0)')
    ax9.set_ylabel('Count')

    # 10. Tidal Volume Distribution
    ax10 = axes[3, 0]
    tv_vals = df_mdp['tidal_volume'].dropna()
    tv_vals = tv_vals[(tv_vals >= 100) & (tv_vals <= 1000)]
    sns.histplot(tv_vals, kde=True, color='mediumorchid', ax=ax10, bins=35)
    ax10.set_title('J. Exhaled Tidal Volume (TV)', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Tidal Volume (mL)')
    ax10.set_ylabel('Count')

    # 11. Serum Lactate & Arterial pH
    ax11 = axes[3, 1]
    lac_vals = df_mdp['LACTATE'].dropna()
    lac_vals = lac_vals[(lac_vals >= 0) & (lac_vals <= 15)]
    sns.histplot(lac_vals, kde=True, color='crimson', ax=ax11, bins=35)
    ax11.axvline(2.0, color='black', linestyle='--', label='Normal threshold (2.0 mmol/L)')
    ax11.set_title('K. Serum Lactate Distribution', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Lactate (mmol/L)')
    ax11.set_ylabel('Count')
    ax11.legend()

    # 12. Discrete Action Bins Distribution
    ax12 = axes[3, 2]
    if 'PEEP_binned' in df_mdp.columns and 'FiO2_binned' in df_mdp.columns:
        action_counts = pd.DataFrame({
            'PEEP Bin': df_mdp['PEEP_binned'].value_counts(normalize=True) * 100,
            'FiO2 Bin': df_mdp['FiO2_binned'].value_counts(normalize=True) * 100,
            'TV Bin': df_mdp['tidal_volume_binned'].value_counts(normalize=True) * 100,
            'RR Bin': df_mdp['RespRate_binned'].value_counts(normalize=True) * 100,
        }).reset_index().rename(columns={'index': 'Bin_Index'})
        df_melted = pd.melt(action_counts, id_vars=['Bin_Index'], var_name='Variable', value_name='Frequency_%')
        sns.barplot(x='Bin_Index', y='Frequency_%', hue='Variable', data=df_melted, ax=ax12, palette='Set2')
        ax12.set_title('L. MDP Discrete Action Bin Frequencies', fontsize=12, fontweight='bold')
        ax12.set_xlabel('Quintile Bin (0 = Lowest, 4 = Highest)')
        ax12.set_ylabel('Frequency (%)')
        ax12.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "patient_data_distribution.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {plot_path}")

    # Copy plot to artifact directory if available
    artifact_dir = "/Users/deepakyadav/.gemini/antigravity/brain/0bedc334-897f-470b-95ae-ed5582986857"
    if os.path.exists(artifact_dir):
        artifact_plot = os.path.join(artifact_dir, "patient_data_distribution.png")
        shutil.copyfile(plot_path, artifact_plot)
        print(f"Copied distribution plot to artifact directory: {artifact_plot}")

    return df_summary, dist_combined


if __name__ == "__main__":
    analyze_cohort_distributions()
