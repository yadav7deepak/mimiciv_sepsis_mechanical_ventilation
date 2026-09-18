"""
Computes and displays the exact bin cutoffs, ranges, counts, and percentages
for all 4 clinical action variables (PEEP, FiO2, tidal_volume, RespRate)
used in the discrete action space (225 actions) of the D3QN model.

Outputs:
- results/action_variables_bin_cutoffs.csv
- data/action_variables_bin_cutoffs.csv
"""

import os
import sys
import pandas as pd

try:
    from .config import LOCAL_DATA_DIR, resolve_intermediate_path
except ImportError:
    from config import LOCAL_DATA_DIR, resolve_intermediate_path


def compute_action_bin_cutoffs(output_dir: str = str(LOCAL_DATA_DIR)):
    mdp_file = resolve_intermediate_path("final_patient_dataset_mdp.csv")
    print(f"Loading MDP dataset: {mdp_file}")
    df_mdp = pd.read_csv(mdp_file)

    bin_rows = []
    action_vars = ['PEEP', 'FiO2', 'tidal_volume', 'RespRate']

    print("\n" + "=" * 80)
    print("EXACT ACTION VARIABLES BIN CUTOFFS & DISTRIBUTION")
    print("=" * 80)

    for var in action_vars:
        bin_col = f"{var}_binned"
        if bin_col not in df_mdp.columns:
            continue

        print(f"\n--- {var} ({bin_col}) ---")
        unique_bins = sorted(df_mdp[bin_col].dropna().unique())
        for b in unique_bins:
            sub = df_mdp[df_mdp[bin_col] == b][var]
            cnt = len(sub)
            pct = (cnt / len(df_mdp)) * 100.0
            min_v = float(sub.min())
            max_v = float(sub.max())
            mean_v = float(sub.mean())
            med_v = float(sub.median())

            print(f"  Bin {int(b)}: Range=[{min_v:.2f}, {max_v:.2f}] | Median={med_v:.2f} | Count={cnt:,} ({pct:.2f}%)")

            bin_rows.append({
                'Variable': var,
                'Bin_Index': int(b),
                'Min_Value': round(min_v, 2),
                'Max_Value': round(max_v, 2),
                'Mean_Value': round(mean_v, 2),
                'Median_Value': round(med_v, 2),
                'Timestep_Count': cnt,
                'Percentage': round(pct, 2)
            })

    df_cutoffs = pd.DataFrame(bin_rows)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "action_variables_bin_cutoffs.csv")
    df_cutoffs.to_csv(out_csv, index=False)
    print(f"\nSaved action bin cutoffs to: {out_csv}")

    results_dir = os.path.join(os.path.dirname(output_dir), "results")
    if os.path.exists(results_dir):
        res_csv = os.path.join(results_dir, "action_variables_bin_cutoffs.csv")
        df_cutoffs.to_csv(res_csv, index=False)
        print(f"Saved copy to: {res_csv}")

    return df_cutoffs


if __name__ == "__main__":
    compute_action_bin_cutoffs()
