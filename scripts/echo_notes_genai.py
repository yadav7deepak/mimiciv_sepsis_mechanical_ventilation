"""
Echo Notes Feature Extraction via Generative AI (converted from echo_notes_genai.ipynb).
Uses Google Gemini API (google.generativeai) to extract structured clinical parameters:
- Height, Weight, BSA, BPSys, BPDias, HR, Status, Test, Doppler, Contrast, Technical Quality.

Configured with GEMINI_API_KEY from .env.
Reads raw note files from Google Drive (MIMIC_PATH/mimic-iv-note/2.2/note).
Saves output locally to LOCAL_DATA_DIR/echo_features_genai.csv.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

# Import configuration and drive utilities
try:
    from .config import (
        MIMIC_PATH,
        LOCAL_DATA_DIR,
        GEMINI_API_KEY,
        get_output_path,
    )
    from .drive_utils import mount_drive, verify_mimic_path
except ImportError:
    from config import (
        MIMIC_PATH,
        LOCAL_DATA_DIR,
        GEMINI_API_KEY,
        get_output_path,
    )
    from drive_utils import mount_drive, verify_mimic_path


def generate_llm_prompt(note_text: str) -> str:
    """Constructs prompt for Gemini LLM to extract echo measurements as structured JSON."""
    return f"""Extract the following medical parameters from the provided echo note text and return the output in valid JSON format.
If a variable is not found, return null for that specific variable. Ensure numerical values are returned as numbers where appropriate.

Parameters to extract:
- Height: (numerical value or string with unit)
- Weight: (numerical value or string with unit)
- BSA: (Body Surface Area, numerical value)
- BPSys: (Systolic Blood Pressure, numerical value)
- BPDias: (Diastolic Blood Pressure, numerical value)
- HR: (Heart Rate, numerical value)
- Status: (string e.g., 'Completed', 'Limited')
- Test: (string value)
- Doppler: (string value)
- Contrast: (string value)
- TechnicalQuality: (string value)

Echo Note Text:
\"\"\"
{note_text}
\"\"\"

Return ONLY a valid JSON object matching this schema:
{{
  "Height": null,
  "Weight": null,
  "BSA": null,
  "BPSys": null,
  "BPDias": null,
  "HR": null,
  "Status": null,
  "Test": null,
  "Doppler": null,
  "Contrast": null,
  "TechnicalQuality": null
}}
"""


def extract_parameters_with_gemini(
    sample_notes: pd.DataFrame,
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    delay_sec: float = 1.0
) -> pd.DataFrame:
    """Configures Gemini API and extracts parameters for given notes."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai is not installed. Please run: pip install google-generativeai"
        )

    if not api_key:
        raise ValueError(
            "Gemini API key is required. Please set GEMINI_API_KEY in your .env file or pass via --api-key."
        )

    print(f"\n[echo_notes_genai] Configuring Google Generative AI (model: {model_name})...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    extracted_records = []
    total = len(sample_notes)
    print(f"[echo_notes_genai] Processing {total} notes with LLM...")

    for i, (idx, row) in enumerate(sample_notes.iterrows()):
        note_text = row.get('text', '')
        note_id = row.get('note_id', idx)
        subject_id = row.get('subject_id', None)
        hadm_id = row.get('hadm_id', None)

        prompt = generate_llm_prompt(note_text)
        record = {
            "note_id": note_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "Height": None,
            "Weight": None,
            "BSA": None,
            "BPSys": None,
            "BPDias": None,
            "HR": None,
            "Status": None,
            "Test": None,
            "Doppler": None,
            "Contrast": None,
            "TechnicalQuality": None,
        }

        try:
            response = model.generate_content(prompt)
            resp_text = response.text.strip()

            # Parse JSON block
            json_str = resp_text
            if "```json" in resp_text:
                start = resp_text.find("```json") + 7
                end = resp_text.find("```", start)
                json_str = resp_text[start:end if end != -1 else None].strip()
            elif "```" in resp_text:
                start = resp_text.find("```") + 3
                end = resp_text.find("```", start)
                json_str = resp_text[start:end if end != -1 else None].strip()

            data = json.loads(json_str)
            for k, v in data.items():
                if k in record:
                    record[k] = v
        except Exception as e:
            print(f"  Note {i+1}/{total} (note_id {note_id}) failed: {e}")

        extracted_records.append(record)
        if delay_sec > 0:
            time.sleep(delay_sec)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Processed {i+1}/{total} notes...")

    return pd.DataFrame(extracted_records)


def run_all(
    mimic_path: str = MIMIC_PATH,
    output_path: str = str(LOCAL_DATA_DIR),
    api_key: str = GEMINI_API_KEY,
    sample_limit: int = 50
):
    """Executes GenAI extraction on echo notes."""
    mount_drive()
    verify_mimic_path(mimic_path, raise_error=True)
    os.makedirs(output_path, exist_ok=True)

    note_dir = os.path.join(mimic_path, "mimic-iv-note/2.2/note")
    discharge_file = os.path.join(note_dir, "discharge.csv.gz")

    print(f"[echo_notes_genai] Reading notes from: {discharge_file}")
    df = pd.read_csv(discharge_file, nrows=sample_limit * 5)
    echo_df = df[df['text'].str.contains('Echo', case=False, na=False)].head(sample_limit)

    results_df = extract_parameters_with_gemini(echo_df, api_key=api_key)
    out_file = os.path.join(output_path, "echo_features_genai.csv")
    results_df.to_csv(out_file, index=False)
    print(f"[echo_notes_genai] Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Echo Report Features using Google Gemini")
    parser.add_argument("--mimic-path", type=str, default=MIMIC_PATH, help="Path to MIMIC-IV on Google Drive")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--api-key", type=str, default=GEMINI_API_KEY, help="Google Gemini API key")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample echo notes to process")
    args = parser.parse_args()

    run_all(mimic_path=args.mimic_path, output_path=args.output_dir, api_key=args.api_key, sample_limit=args.limit)
