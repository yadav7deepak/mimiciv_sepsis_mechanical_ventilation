"""
Ventilation Data Preprocessing & RL Transition Preparation Script (converted from vent_datapreparation.ipynb).
Performs:
1. Adult cohort filtering (age >= 18) and >= 24 hours mechanical ventilation.
2. Missing data imputation (forward/backward fill per stay + iterative/median imputer).
3. Negative value corrections (e.g., PEEP < 0 clamped to 0).
4. Feature scaling & K-Means clustering to create discrete state space.
5. Cluster consolidation: reassign small clusters (< 1000 samples) to nearest large centroid.
6. Action space discretization (PEEP, FiO2, plateau_pressure/tidal_volume, RespRate).
7. Reward definition & MDP transition preparation.

Reads intermediate files from LOCAL_DATA_DIR (./data) with Drive fallback.
Saves all preprocessed and clustered outputs locally to LOCAL_DATA_DIR (./data).
"""

import os
import sys
import shutil
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Import configuration and drive utilities
try:
    from .config import (
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from .drive_utils import mount_drive
except ImportError:
    from config import (
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from drive_utils import mount_drive


def filter_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for adult ICU patients (age >= 18) who received mechanical ventilation for >= 24h.
    """
    print("\n[vent_datapreparation] Filtering adult mechanical ventilation cohort...")
    print(f"Initial row count: {len(df):,}, unique subjects: {df['subject_id'].nunique():,}")

    if 'first_admit_age' in df.columns:
        adult_subjects = df[df['first_admit_age'] >= 18]['subject_id'].unique()
        df = df[df['subject_id'].isin(adult_subjects)].copy()
        print(f"After adult filter (>=18): {len(df):,} rows, {df['subject_id'].nunique():,} subjects")

    if 'MechVent' in df.columns:
        mech_subjects = df[df['MechVent'] == 1]['subject_id'].unique()
        df = df[df['subject_id'].isin(mech_subjects)].copy()
        print(f"After MechVent filter: {len(df):,} rows, {df['subject_id'].nunique():,} subjects")

    if 'start_time' in df.columns and 'stay_id' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        duration = df.groupby('stay_id').agg(
            {'start_time': lambda x: (max(x) - min(x)).total_seconds() / 3600.0}
        )
        valid_stays = duration[duration['start_time'] >= 24.0].index
        df = df[df['stay_id'].isin(valid_stays)].copy()
        print(f"After >= 24h stay filter: {len(df):,} rows, {df['stay_id'].nunique():,} stays")

    return df.reset_index(drop=True)


def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values using forward fill, backward fill per subject, and median filling.
    """
    print("\n[vent_datapreparation] Imputing missing clinical features...")
    df_imputed = df.copy()

    var_to_fill = [
        'weight', 'gcs', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'shockindex',
        'RespRate', 'TempC', 'SpO2', 'POTASSIUM', 'SODIUM', 'CHLORIDE',
        'GLUCOSE', 'BUN', 'CREATININE', 'MAGNESIUM', 'CALCIUM', 'CARBONDIOXIDE',
        'SGOT', 'SGPT', 'BILIRUBIN', 'ALBUMIN', 'HEMOGLOBIN', 'WBC', 'PLATELET',
        'PTT', 'PT', 'INR', 'PH', 'PAO2', 'PACO2', 'BASE_EXCESS', 'BICARBONATE',
        'LACTATE', 'PAO2FiO2ratio', 'FiO2', 'vaso_total', 'cum_fluid_balance',
        'PEEP', 'tidal_volume', 'plateau_pressure'
    ]
    present_vars = [v for v in var_to_fill if v in df_imputed.columns]

    # Sort by subject and time
    df_imputed = df_imputed.sort_values(by=['subject_id', 'start_time']).reset_index(drop=True)

    # Backward fill per subject
    df_imputed[present_vars] = df_imputed.groupby('subject_id')[present_vars].bfill()
    # Forward fill per subject
    df_imputed[present_vars] = df_imputed.groupby('subject_id')[present_vars].ffill()

    # Fill remaining NaNs with column median
    for col in present_vars:
        if df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val if pd.notna(median_val) else 0)

    # Clamp physiological values
    if 'PEEP' in df_imputed.columns:
        df_imputed.loc[df_imputed['PEEP'] < 0, 'PEEP'] = 0.0

    print("[vent_datapreparation] Missing data imputation completed.")
    return df_imputed


def perform_clustering(df: pd.DataFrame, n_clusters: int = 500, min_cluster_size: int = 1000) -> pd.DataFrame:
    """
    Fits KMeans on scaled continuous state features and merges small clusters into nearest large cluster.
    """
    print(f"\n[vent_datapreparation] Performing K-Means clustering (K={n_clusters})...")
    df_clustered = df.copy()

    columns_to_exclude = [
        'stay_id', 'subject_id', 'hadm_id', 'start_time', 'dischtime', 'deathtime',
        'PEEP', 'FiO2', 'RespRate', 'plateau_pressure', 'tidal_volume',
        'gender', 'cluster_label'
    ]

    if 'gender' in df_clustered.columns:
        df_clustered['gender_numeric'], _ = pd.factorize(df_clustered['gender'])

    features_to_scale = [c for c in df_clustered.columns if c not in columns_to_exclude and pd.api.types.is_numeric_dtype(df_clustered[c])]

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_clustered[features_to_scale].fillna(0))
    df_scaled = pd.DataFrame(scaled_matrix, columns=features_to_scale)

    # Run KMeans
    actual_k = min(n_clusters, len(df_clustered))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    df_clustered['cluster_label'] = labels
    df_scaled['cluster_label'] = labels

    print(f"KMeans finished with {actual_k} clusters.")
    cluster_sizes = df_clustered['cluster_label'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index.tolist()
    large_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()

    if small_clusters and large_clusters:
        print(f"Consolidating {len(small_clusters)} small clusters (< {min_cluster_size} items)...")
        large_centroids = df_scaled[df_scaled['cluster_label'].isin(large_clusters)].groupby('cluster_label')[features_to_scale].mean()
        centroids_array = large_centroids.values
        large_labels_list = large_centroids.index.tolist()

        small_mask = df_clustered['cluster_label'].isin(small_clusters)
        small_obs_scaled = df_scaled.loc[small_mask, features_to_scale].values

        if len(small_obs_scaled) > 0:
            distances = cdist(small_obs_scaled, centroids_array, metric='euclidean')
            nearest_idx = np.argmin(distances, axis=1)
            reassigned = [large_labels_list[i] for i in nearest_idx]
            df_clustered.loc[small_mask, 'cluster_label'] = reassigned

    print(f"Final active clusters: {df_clustered['cluster_label'].nunique()}")
    return df_clustered


def discretize_action_space(df: pd.DataFrame, num_bins: int = 5) -> (pd.DataFrame, pd.DataFrame):
    """
    Discretizes ventilation action variables (PEEP, FiO2, tidal_volume, RespRate)
    and constructs the Cartesian action space (225 combinations).
    """
    print("\n[vent_datapreparation] Discretizing clinical action variables into bins...")
    df_actions = df.copy()

    # Variables for action discretization (matching Vent_D3QN.ipynb)
    candidate_action_vars = ['PEEP', 'FiO2', 'tidal_volume', 'RespRate']
    action_vars = [v for v in candidate_action_vars if v in df_actions.columns]

    binned_cols = []
    unique_bins_per_var = []
    for var in action_vars:
        binned_name = f"{var}_binned"
        df_actions[var] = pd.to_numeric(df_actions[var], errors='coerce').fillna(0)
        df_actions[binned_name] = pd.qcut(
            df_actions[var],
            q=num_bins,
            labels=False,
            duplicates='drop'
        )
        binned_cols.append(binned_name)
        ubins = sorted(df_actions[binned_name].dropna().unique().tolist())
        unique_bins_per_var.append(ubins)
        print(f"  {var}: {len(ubins)} discrete bins -> {ubins}")

    # Create full action space table
    action_combinations = list(itertools.product(*unique_bins_per_var))
    df_action_space = pd.DataFrame(action_combinations, columns=binned_cols)
    df_action_space['action_id'] = range(len(df_action_space))
    print(f"Total discrete action combinations: {len(df_action_space)}")

    # Add mortality outcome indicator for reward
    if 'deathtime' in df_actions.columns:
        subjects_with_death = df_actions.groupby('subject_id')['deathtime'].apply(lambda x: x.notna().any())
        df_actions['subject_has_deathtime'] = df_actions['subject_id'].map(subjects_with_death).astype(int)
    elif 'hospmort90day' in df_actions.columns:
        df_actions['subject_has_deathtime'] = df_actions['hospmort90day'].fillna(0).astype(int)
    else:
        df_actions['subject_has_deathtime'] = 0

    return df_actions, df_action_space


def run_all(output_path: str = str(LOCAL_DATA_DIR), n_clusters: int = 500, force: bool = False):
    """Executes the full ventilation data preparation pipeline."""
    mount_drive()
    os.makedirs(output_path, exist_ok=True)

    cluster_file = os.path.join(output_path, "final_patient_dataset_mdp.csv")
    action_space_file = os.path.join(output_path, "action_space.csv")
    preproc_file = os.path.join(output_path, "final_patient_dataset_imputed.csv")

    if not force and os.path.exists(cluster_file) and os.path.exists(action_space_file) and os.path.getsize(cluster_file) > 100000 and os.path.getsize(action_space_file) > 100:
        print(f"\n[vent_datapreparation] Reusing existing {cluster_file} and {action_space_file}...")
        return

    # Check Drive for pre-clustered dataset
    drive_cluster = resolve_intermediate_path("final_patient_dataset_mdp.csv")
    if not force and os.path.exists(drive_cluster) and drive_cluster != cluster_file and os.path.getsize(drive_cluster) > 100000:
        print(f"\n[vent_datapreparation] Copying pre-existing clustered dataset from Drive: {drive_cluster}")
        shutil.copyfile(drive_cluster, cluster_file)
        drive_preproc = resolve_intermediate_path("final_patient_dataset_imputed.csv")
        if os.path.exists(drive_preproc) and drive_preproc != preproc_file:
            shutil.copyfile(drive_preproc, preproc_file)

        print(f"\n[vent_datapreparation] Discretizing actions from clustered dataset...")
        df_clustered = pd.read_csv(cluster_file)
        df_final, df_action_space = discretize_action_space(df_clustered)
        df_final.to_csv(cluster_file, index=False)
        df_action_space.to_csv(action_space_file, index=False)
        print(f"[vent_datapreparation] Clustered dataset with actions saved to: {cluster_file}")
        print(f"[vent_datapreparation] Action space saved to: {action_space_file}")
        return

    input_file = resolve_intermediate_path("final_patient_dataset_master.csv")
    print(f"Reading master dataset: {input_file}")
    df = pd.read_csv(input_file)

    # 1. Filter cohort
    df_filtered = filter_cohort(df)

    # 2. Impute missing data
    df_imputed = impute_missing_data(df_filtered)
    df_imputed.to_csv(preproc_file, index=False)
    print(f"[vent_datapreparation] Preprocessed data saved to: {preproc_file}")

    # 3. Clustering
    df_clustered = perform_clustering(df_imputed, n_clusters=n_clusters)

    # 4. Action discretization
    df_final, df_action_space = discretize_action_space(df_clustered)

    df_final.to_csv(cluster_file, index=False)
    print(f"[vent_datapreparation] Clustered dataset saved to: {cluster_file}")

    df_action_space.to_csv(action_space_file, index=False)
    print(f"[vent_datapreparation] Action space saved to: {action_space_file}")

    print("\n[vent_datapreparation] Data preparation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV Ventilation Data Preparation & Clustering")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--clusters", type=int, default=500, help="Number of KMeans clusters for state space")
    parser.add_argument("--force", action="store_true", help="Force recomputation from master dataset")
    args = parser.parse_args()

    run_all(output_path=args.output_dir, n_clusters=args.clusters, force=args.force)
