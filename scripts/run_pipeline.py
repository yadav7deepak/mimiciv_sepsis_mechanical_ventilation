"""
Master Orchestration Script for MIMIC-IV Data Preparation and RL Pipeline.
Allows running the entire pipeline end-to-end or running individual stages.

Usage:
  python scripts/run_pipeline.py --stage all
  python scripts/run_pipeline.py --stage patient
  python scripts/run_pipeline.py --stage lab
  python scripts/run_pipeline.py --stage vasopressors
  python scripts/run_pipeline.py --stage ventilation
  python scripts/run_pipeline.py --stage sampling
  python scripts/run_pipeline.py --stage prep
  python scripts/run_pipeline.py --stage rl
  python scripts/run_pipeline.py --stage echo-regex
  python scripts/run_pipeline.py --stage echo-genai
"""

import os
import sys
import time
import argparse
from typing import Optional
from pathlib import Path

# Add scripts folder to python path
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from config import (
    MIMIC_PATH,
    LOCAL_DATA_DIR,
    print_config_summary,
)
from drive_utils import mount_drive, verify_mimic_path


def run_stage_patient(mimic_path: str, output_dir: str, force: bool = False):
    import patient_data
    patient_data.run_all(mimic_path=mimic_path, output_path=output_dir, force=force)


def run_stage_lab(mimic_path: str, output_dir: str, force: bool = False):
    import lab_and_fluid
    lab_and_fluid.run_all(mimic_path=mimic_path, output_path=output_dir, force=force)


def run_stage_vasopressors(mimic_path: str, output_dir: str, force: bool = False):
    import vasopressors_6files
    vasopressors_6files.run_all(mimic_path=mimic_path, output_path=output_dir, force=force)


def run_stage_ventilation(mimic_path: str, output_dir: str, include_scores: bool = False, force: bool = False):
    import ventilation
    ventilation.run_all(mimic_path=mimic_path, output_path=output_dir, include_scores=include_scores, force=force)


def run_stage_sampling(mimic_path: str, output_dir: str, merge_final: bool = False, force: bool = False):
    import sampling
    sampling.run_all(mimic_path=mimic_path, output_path=output_dir, merge_final=merge_final, force=force)


def run_stage_prep(output_dir: str, clusters: int = 500):
    import vent_datapreparation
    vent_datapreparation.run_all(output_path=output_dir, n_clusters=clusters)


def run_stage_rl(
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 64,
    update_freq: int = 4,
    max_transitions: Optional[int] = None
):
    import vent_d3qn
    vent_d3qn.run_all(
        output_path=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        update_freq=update_freq,
        max_transitions=max_transitions
    )


def run_stage_echo_regex(mimic_path: str, output_dir: str):
    import data_controls
    data_controls.run_all(mimic_path=mimic_path, output_path=output_dir)


def run_stage_echo_genai(mimic_path: str, output_dir: str):
    import echo_notes_genai
    echo_notes_genai.run_all(mimic_path=mimic_path, output_path=output_dir)


def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Ventilation RL Pipeline Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all", "patient", "lab", "vasopressors", "ventilation",
            "sampling", "scores", "prep", "rl", "echo-regex", "echo-genai"
        ],
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory (./data)")
    parser.add_argument("--clusters", type=int, default=500, help="Number of clusters for state space")
    parser.add_argument("--epochs", type=int, default=2, help="Number of D3QN training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for D3QN training")
    parser.add_argument("--update-freq", type=int, default=4, help="Frequency of gradient updates per transition (default: 4)")
    parser.add_argument("--max-transitions", type=int, default=None, help="Optional transition limit for testing")
    parser.add_argument("--force", action="store_true", help="Force recomputation of intermediate tables instead of reusing local or Drive cached tables")
    args = parser.parse_args()

    print_config_summary()
    mount_drive()
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    stage = args.stage

    if stage in ["all", "patient"]:
        print("\n" + "=" * 50)
        print("STAGE 1: Patient Data Extraction")
        print("=" * 50)
        run_stage_patient(args.mimic_path, args.output_dir, force=args.force)

    if stage in ["all", "lab"]:
        print("\n" + "=" * 50)
        print("STAGE 2: Lab & Fluid Extraction")
        print("=" * 50)
        run_stage_lab(args.mimic_path, args.output_dir, force=args.force)

    if stage in ["all", "vasopressors"]:
        print("\n" + "=" * 50)
        print("STAGE 3: Vasopressors Dose Calculation")
        print("=" * 50)
        run_stage_vasopressors(args.mimic_path, args.output_dir, force=args.force)

    if stage in ["all", "ventilation"]:
        print("\n" + "=" * 50)
        print("STAGE 4: Ventilation Parameters & Tables")
        print("=" * 50)
        run_stage_ventilation(args.mimic_path, args.output_dir, include_scores=False, force=args.force)

    if stage in ["all", "sampling"]:
        print("\n" + "=" * 50)
        print("STAGE 5: 4-Hour Resampling & Aggregation")
        print("=" * 50)
        run_stage_sampling(args.mimic_path, args.output_dir, merge_final=False, force=args.force)

    if stage in ["all", "scores"]:
        print("\n" + "=" * 50)
        print("STAGE 6: Clinical Scores (SIRS & SOFA)")
        print("=" * 50)
        import ventilation
        ventilation.create_sirs_scores(args.output_dir, force=args.force)
        ventilation.create_sofa_scores(args.output_dir, force=args.force)

        print("\nMerging with demographics to generate master dataset...")
        import sampling
        sampling.create_final_master_patient_dataset(args.mimic_path, args.output_dir, force=args.force)

    if stage in ["all", "prep"]:
        print("\n" + "=" * 50)
        print("STAGE 7: Ventilation Cohort Preprocessing & Clustering")
        print("=" * 50)
        run_stage_prep(args.output_dir, clusters=args.clusters)

    if stage in ["all", "rl"]:
        print("\n" + "=" * 50)
        run_stage_rl(
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            update_freq=args.update_freq,
            max_transitions=args.max_transitions
        )

    if stage == "echo-regex":
        print("\n" + "=" * 50)
        print("OPTIONAL STAGE: Echo Notes Regex Extraction")
        print("=" * 50)
        run_stage_echo_regex(args.mimic_path, args.output_dir)

    if stage == "echo-genai":
        print("\n" + "=" * 50)
        print("OPTIONAL STAGE: Echo Notes GenAI Extraction")
        print("=" * 50)
        run_stage_echo_genai(args.mimic_path, args.output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Pipeline stage '{stage}' completed in {elapsed:.2f} seconds.")
    print(f"All local outputs are saved in: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
