"""
Central configuration module for the MIMIC-IV Data Preparation and RL Pipeline.
Loads credentials and directory paths from .env file and environment variables.
"""

import os
import sys
from pathlib import Path


def load_env_file(dotenv_path=None):
    """
    Lightweight .env file parser with zero external dependencies.
    Falls back to python-dotenv if available.
    """
    if dotenv_path is None:
        # Check current working directory, script directory, and project root
        candidate_names = [".env", "config.env"]
        candidate_paths = []
        for name in candidate_names:
            candidate_paths.extend([
                Path.cwd() / name,
                Path(__file__).resolve().parent.parent / name,
                Path(__file__).resolve().parent / name,
            ])
        for p in candidate_paths:
            if p.is_file():
                dotenv_path = p
                break

    if not dotenv_path or not Path(dotenv_path).is_file():
        return

    # Try python-dotenv first if installed
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    except ImportError:
        pass

    # Pure Python fallback parser
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Remove quotes if present
                if len(val) >= 2 and (
                    (val.startswith('"') and val.endswith('"')) or
                    (val.startswith("'") and val.endswith("'"))
                ):
                    val = val[1:-1]
                if key not in os.environ:
                    os.environ[key] = val


# Load environment variables upon module import
load_env_file()

# ------------------------------------------------------------------------------
# Google Drive & Mount Configuration
# ------------------------------------------------------------------------------
GDRIVE_EMAIL = os.getenv("GDRIVE_EMAIL", "")
GDRIVE_PASSWORD = os.getenv("GDRIVE_PASSWORD", "")
GDRIVE_MOUNT_PATH = os.getenv("GDRIVE_MOUNT_PATH", "/content/drive")

# ------------------------------------------------------------------------------
# Input Dataset Paths (READ DATA FROM GOOGLE DRIVE ONLY)
# ------------------------------------------------------------------------------
# Primary MIMIC-IV directory on Google Drive
DEFAULT_MIMIC_PATH = os.path.join(
    GDRIVE_MOUNT_PATH, "My Drive/mimic-iv/physionet.org/files/mimiciv/3.0"
)
MIMIC_PATH = os.path.expanduser(os.getenv("MIMIC_PATH", DEFAULT_MIMIC_PATH))

# If user specified path ending in 'mimiciv', automatically check for '3.0' subfolder
if os.path.isdir(os.path.join(MIMIC_PATH, "3.0")):
    MIMIC_PATH = os.path.join(MIMIC_PATH, "3.0")

# Optional path for pre-existing ventilation data on Google Drive (fallback)
DEFAULT_VENTILATION_DATA_PATH = os.path.join(
    MIMIC_PATH, "ventilation_data/patient_data"
)
GDRIVE_VENTILATION_DATA_PATH = os.path.expanduser(
    os.getenv("GDRIVE_VENTILATION_DATA_PATH", DEFAULT_VENTILATION_DATA_PATH)
)

# ------------------------------------------------------------------------------
# Local Output Configuration (STORES OUTPUTS LOCALLY IN ./data)
# ------------------------------------------------------------------------------
# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local data output folder
LOCAL_DATA_DIR_RAW = os.getenv("LOCAL_DATA_DIR", "./data")
if os.path.isabs(LOCAL_DATA_DIR_RAW):
    LOCAL_DATA_DIR = Path(LOCAL_DATA_DIR_RAW)
else:
    LOCAL_DATA_DIR = (PROJECT_ROOT / LOCAL_DATA_DIR_RAW).resolve()

# Ensure local data output folder exists
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# External API Credentials
# ------------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ------------------------------------------------------------------------------
# Canonical Dataset File Constants
# ------------------------------------------------------------------------------
FILE_PATIENT_DEMOGRAPHICS = "patient_demographics.csv"
FILE_PATIENT_ELIXHAUSER = "patient_elixhauser_scores.csv"
FILE_PATIENT_VITALS = "patient_vital_signs.csv"
FILE_PATIENT_GCS = "patient_gcs.csv"
FILE_PATIENT_ECHO_NOTES = "patient_echo_notes.csv"
FILE_PATIENT_WEIGHT = "patient_weight.csv"
FILE_PATIENT_STAY_TIMES = "patient_icu_stay_times.csv"

FILE_LAB_VALUES = "lab_values.csv"
FILE_FLUID_CUMULATIVE = "fluid_cumulative_balance.csv"
FILE_FLUID_INTRAVENOUS = "fluid_intravenous.csv"
FILE_URINE_OUTPUT = "urine_output.csv"

FILE_VASOPRESSOR_WEIGHTS = "vasopressor_weights.csv"
FILE_VASOPRESSOR_DOPAMINE = "vasopressor_dopamine.csv"
FILE_VASOPRESSOR_EPINEPHRINE = "vasopressor_epinephrine.csv"
FILE_VASOPRESSOR_NOREPINEPHRINE = "vasopressor_norepinephrine.csv"
FILE_VASOPRESSOR_PHENYLEPHRINE = "vasopressor_phenylephrine.csv"
FILE_VASOPRESSOR_VASOPRESSIN = "vasopressor_vasopressin.csv"

FILE_VENTILATION_PARAMETERS = "ventilation_parameters.csv"
FILE_VENTILATION_STATUS = "ventilation_status.csv"
FILE_VASOPRESSORS_COMBINED = "vasopressors_combined.csv"
FILE_VASOPRESSORS_BY_STAY = "vasopressors_combined_by_stay.csv"
FILE_SECONDARY_LABS = "secondary_lab_values.csv"

FILE_MERGED_RAW_LABS = "merged_raw_labs.csv"
FILE_MERGED_RAW_VITALS = "merged_raw_vitals_interventions.csv"
FILE_SAMPLED_4H_LABS = "sampled_4h_labs.csv"
FILE_SAMPLED_4H_VITALS = "sampled_4h_vitals_interventions.csv"
FILE_SAMPLED_4H_COMBINED = "sampled_4h_combined.csv"

FILE_SIRS_SCORES = "sirs_scores.csv"
FILE_SOFA_SCORES = "sofa_scores.csv"

FILE_FINAL_MASTER = "final_patient_dataset_master.csv"
FILE_FINAL_IMPUTED = "final_patient_dataset_imputed.csv"
FILE_FINAL_MDP = "final_patient_dataset_mdp.csv"
FILE_ACTION_SPACE = "action_space.csv"

# Bidirectional alias mapping between new canonical names and legacy Colab names
FILE_ALIASES = {
    # Canonical -> Legacy
    "patient_demographics.csv": "demographics2.csv",
    "patient_elixhauser_scores.csv": "elixhauser_score.csv",
    "patient_vital_signs.csv": "vital_signs2.csv",
    "patient_gcs.csv": "gcs2.csv",
    "patient_echo_notes.csv": "echo_data.csv",
    "patient_weight.csv": "weight.csv",
    "patient_icu_stay_times.csv": "hosp_mort_and_in_out_times.csv",
    "fluid_cumulative_balance.csv": "cum_fluid.csv",
    "fluid_intravenous.csv": "intravenous.csv",
    "vasopressor_weights.csv": "weightdurations.csv",
    "vasopressor_dopamine.csv": "dopamine_dose.csv",
    "vasopressor_epinephrine.csv": "epinephrine_dose.csv",
    "vasopressor_norepinephrine.csv": "norepinephrine_dose.csv",
    "vasopressor_phenylephrine.csv": "phenylephrine_dose.csv",
    "vasopressor_vasopressin.csv": "vasopressin_dose.csv",
    "ventilation_parameters.csv": "vent_parameters.csv",
    "ventilation_status.csv": "ventilation_params.csv",
    "vasopressors_combined_by_stay.csv": "vasopressors_combined_final.csv",
    "secondary_lab_values.csv": "others_lab_values.csv",
    "merged_raw_labs.csv": "overalltable_Lab_withventparams.csv",
    "merged_raw_vitals_interventions.csv": "overalltable_withoutLab_withventparams2.csv",
    "sampled_4h_labs.csv": "sampled_lab_withventparams.csv",
    "sampled_4h_vitals_interventions.csv": "sampled_withoutlab_withventparams.csv",
    "sampled_4h_combined.csv": "sampled_all_withventparams.csv",
    "sirs_scores.csv": "SIRS_sampled_withventparams.csv",
    "sofa_scores.csv": "SOFA_sampled_withventparams.csv",
    "final_patient_dataset_master.csv": "sampled_with_scdem_withventparams.csv",
    "final_patient_dataset_imputed.csv": "sampled_with_scdem_withventparams_preprocess.csv",
    "final_patient_dataset_mdp.csv": "sampled_with_scdem_withventparams_cluster.csv",
    "action_space.csv": "df_action_space.csv",
    # Legacy -> Canonical
    "demographics2.csv": "patient_demographics.csv",
    "elixhauser_score.csv": "patient_elixhauser_scores.csv",
    "vital_signs2.csv": "patient_vital_signs.csv",
    "gcs2.csv": "patient_gcs.csv",
    "echo_data.csv": "patient_echo_notes.csv",
    "echo_dat2.csv": "patient_echo_notes.csv",
    "weightdurations.csv": "vasopressor_weights.csv",
    "dopamine_dose.csv": "vasopressor_dopamine.csv",
    "epinephrine_dose.csv": "vasopressor_epinephrine.csv",
    "norepinephrine_dose.csv": "vasopressor_norepinephrine.csv",
    "phenylephrine_dose.csv": "vasopressor_phenylephrine.csv",
    "vasopressin_dose.csv": "vasopressor_vasopressin.csv",
    "cum_fluid.csv": "fluid_cumulative_balance.csv",
    "intravenous.csv": "fluid_intravenous.csv",
    "vent_parameters.csv": "ventilation_parameters.csv",
    "ventilation_params.csv": "ventilation_status.csv",
    "vasopressors_combined_final.csv": "vasopressors_combined_by_stay.csv",
    "others_lab_values.csv": "secondary_lab_values.csv",
    "overalltable_Lab_withventparams.csv": "merged_raw_labs.csv",
    "overalltable_withoutLab_withventparams2.csv": "merged_raw_vitals_interventions.csv",
    "sampled_lab_withventparams.csv": "sampled_4h_labs.csv",
    "sampled_withoutlab_withventparams.csv": "sampled_4h_vitals_interventions.csv",
    "sampled_all_withventparams.csv": "sampled_4h_combined.csv",
    "SIRS_sampled_withventparams.csv": "sirs_scores.csv",
    "SOFA_sampled_withventparams.csv": "sofa_scores.csv",
    "sampled_with_scdem_withventparams.csv": "final_patient_dataset_master.csv",
    "sampled_with_scdem_withventparams_preprocess.csv": "final_patient_dataset_imputed.csv",
    "sampled_with_scdem_withventparams_cluster.csv": "final_patient_dataset_mdp.csv",
    "df_action_space.csv": "action_space.csv",
}


# ------------------------------------------------------------------------------
# Helper Path Functions
# ------------------------------------------------------------------------------
def get_output_path(filename: str) -> str:
    """
    Returns the absolute path for saving an output file into the local data directory.
    Ensures the target folder exists.
    """
    path = LOCAL_DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_intermediate_path(filename: str, fallback_to_drive: bool = True) -> str:
    """
    Resolves an intermediate data file path:
    1. Checks local data folder first (./data/<filename>).
    2. Checks alias in local data folder (e.g. canonical vs legacy name).
    3. If not found locally and fallback_to_drive is True, checks Google Drive path.
    4. Defaults to local path if neither exists yet.
    """
    # 1. Direct match in local data
    local_path = LOCAL_DATA_DIR / filename
    if local_path.is_file():
        return str(local_path)

    # 2. Check alias in local data
    alias = FILE_ALIASES.get(filename)
    if alias:
        alias_path = LOCAL_DATA_DIR / alias
        if alias_path.is_file():
            return str(alias_path)

    # 3. Check Google Drive paths
    if fallback_to_drive and GDRIVE_VENTILATION_DATA_PATH:
        drive_path = os.path.join(GDRIVE_VENTILATION_DATA_PATH, filename)
        if os.path.exists(drive_path):
            return drive_path
        if alias:
            drive_alias_path = os.path.join(GDRIVE_VENTILATION_DATA_PATH, alias)
            if os.path.exists(drive_alias_path):
                return drive_alias_path

    # Fallback to local path
    return str(local_path)


def resolve_raw_mimic_path(subfolder: str, filename: str) -> str:
    """
    Returns the path to a raw MIMIC-IV file strictly on Google Drive.
    subfolder can be 'icu', 'hosp', or 'mimic-iv-note/2.2/note', etc.
    """
    if subfolder:
        return os.path.join(MIMIC_PATH, subfolder, filename)
    return os.path.join(MIMIC_PATH, filename)


def print_config_summary():
    """Prints a summary of the current configuration and active paths."""
    print("=" * 60)
    print("MIMIC-IV Data Pipeline Configuration")
    print("=" * 60)
    print(f"Project Root:                 {PROJECT_ROOT}")
    print(f"Google Drive Mount Path:      {GDRIVE_MOUNT_PATH}")
    print(f"Google Drive Email:           {GDRIVE_EMAIL or '(Not set)'}")
    print(f"Google Drive Password:        {'[SET]' if GDRIVE_PASSWORD else '(Not set)'}")
    print(f"MIMIC-IV Raw Path (Drive):    {MIMIC_PATH}")
    print(f"Drive Vent Data Path:         {GDRIVE_VENTILATION_DATA_PATH}")
    print(f"Local Output Directory:       {LOCAL_DATA_DIR}")
    print(f"Gemini API Key:               {'[SET]' if GEMINI_API_KEY else '(Not set)'}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
