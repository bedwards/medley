#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

import sys
import pandas as pd
import numpy as np
import re


def clean(file_path, file_type, output_path, interim_type):
    """Clean either perfinterim or perftca data file."""
    # Read file without headers
    df = pd.read_csv(file_path, header=None)

    # Initialize clean dataframe with student metadata
    clean_df = df.iloc[3:, :5].copy()
    clean_df.columns = [h.replace(" ", "_") for h in df.iloc[1, :5]]
    clean_df["student_id"] = clean_df.iloc[:, 1]  # Local_ID column

    if file_type == "interim":
        # Process TEK columns
        for i, header in enumerate(df.iloc[2, 5:-4]):
            if pd.isna(header):
                continue
            col_name = f"{interim_type}: {header}"
            col_idx = i + 5
            clean_df[col_name] = process_percentage(df.iloc[3:, col_idx])

    else:  # tca
        # Find TCA sections
        tca_sections = []
        for i, val in enumerate(df.iloc[0]):
            match = re.search(r"TCA (\d+)", str(val)) if pd.notna(val) else None
            if match:
                tca_sections.append((f"TCA {match.group(1)}", i))

        # Process each section's columns
        for tca_name, start_idx in tca_sections:
            for i, header in enumerate(df.iloc[2, start_idx:]):
                if pd.isna(header) or not isinstance(header, str):
                    continue
                col_idx = start_idx + i

                # Skip columns after the next TCA section starts
                next_section_starts = False
                for j in range(start_idx + 1, col_idx):
                    if (
                        j < len(df.iloc[0])
                        and isinstance(df.iloc[0, j], str)
                        and "TCA" in df.iloc[0, j]
                    ):
                        next_section_starts = True
                        break
                if next_section_starts:
                    continue

                col_name = f"{tca_name}: {header}"
                if "[" in header:  # TEK column
                    clean_df[col_name] = process_percentage(df.iloc[3:, col_idx])
                elif "Score" in header:
                    clean_df[col_name] = pd.to_numeric(
                        df.iloc[3:, col_idx], errors="coerce"
                    )
                elif any(x in header for x in ["Approaches", "Meets", "Masters"]):
                    clean_df[col_name] = df.iloc[3:, col_idx].map(
                        {"Yes": 1, "No": 0, "-": np.nan}
                    )

    clean_df = clean_df.drop(
        columns=[
            "Student_Name",
            "Local_ID",
            "Special_Ed_Indicator",
            "Emergent_Bilingual",
            "Ethnicity",
        ]
    )

    print(clean_df)
    print(clean_df.columns)

    clean_df.to_csv(output_path, index=False)


def process_percentage(series):
    """Convert percentage strings to numeric values (0-1 scale)."""
    series = series.replace(["--", "-"], np.nan)
    if series.dtype == "object":
        series = series.str.replace("%", "").astype(float) / 100
    return series


if __name__ == "__main__":
    if not sys.argv[1:]:
        print("need input name", file=sys.stderr)
        sys.exit(1)

    n = sys.argv[1]

    if not (("interim" in n) ^ ("tca" in n)):
        print("need exactly one of 'interim' or 'tca' in input name")

    file_type = "interim" if "interim" in n else "tca"

    if file_type == "interim":
        interim_type = "Fall" if "fall" in n else "Spring"
    else:
        interim_type = None

    d = "../input/medley"
    clean(f"{d}/{n}.csv", file_type, f"{d}/{n}_clean.csv", interim_type)
