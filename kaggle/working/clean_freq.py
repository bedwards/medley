#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

import pandas as pd
import re
import sys


def clean(input_file, output_file):
    # Read the raw CSV file with flexible parsing
    df = pd.read_csv(input_file, header=None, skiprows=2)

    # These are the actual column positions in the input file
    df = df.rename(columns={0: "Day", 1: "Topic", 2: "TEKS", 3: "R/S/NT", 4: "Skill"})

    # Extract year information and identify test columns
    year_row = df.iloc[0]
    test_cols = []

    # Expected years in order
    expected_years = [
        "2024",
        "2023",
        "2022",
        "2021",
        "2019",
        "2018",
        "2017",
        "2016",
        "2015",
        "2014",
        "2013",
    ]
    test_types = ["fall", "winter", "field", "spring", "staar"]

    # Find columns with years as headers
    year_columns = {}
    for i in range(5, len(df.columns)):
        if i in df.columns:
            year_val = str(year_row[i]).strip()
            if re.match(r"^\d{4}$", year_val):  # Year like "2024"
                year_columns[year_val] = i

    # Process each expected year
    for year in expected_years:
        if year in year_columns:
            start_col = year_columns[year]

            # The next 5 columns are test columns for this year
            for j, test_type in enumerate(test_types):
                if start_col + j + 1 in df.columns:
                    test_col = f"{year}_{j}_{test_type}"
                    df = df.rename(columns={start_col + j + 1: test_col})
                    test_cols.append(test_col)

    # Remove the first few rows which were used for headers
    df = df.iloc[1:]

    # Remove metadata rows (legends, page breaks, etc.)
    metadata_patterns = [
        r"R\s*=\s*Readiness Standard",  # Legend row
        r"S\s*=\s*Supporting Standard",  # Legend row
        r"NT\s*=\s*Not Tested",  # Legend row
        r"White Cells = Reporting Category",  # Category description
        r"^Skill$",  # Column header repeats
    ]

    # First identify rows containing legend text
    legend_rows = df[
        df["TEKS"].str.contains(r"R\s*=|S\s*=|NT\s*=", regex=True, na=False)
        | df["Skill"].str.contains(r"R\s*=|S\s*=|NT\s*=", regex=True, na=False)
        | df["R/S/NT"].str.contains(r"R\s*=|S\s*=|NT\s*=", regex=True, na=False)
    ]

    # Print and remove the legend rows
    print(f"Removing {len(legend_rows)} legend rows")
    df = df.drop(legend_rows.index)

    # Remove rows matching other metadata patterns in Skill column
    for pattern in metadata_patterns:
        pattern_rows = df[df["Skill"].str.contains(pattern, regex=True, na=False)]
        if not pattern_rows.empty:
            print(f"Removing {len(pattern_rows)} rows matching pattern: {pattern}")
            df = df[~df["Skill"].str.contains(pattern, regex=True, na=False)]

    # Also remove completely empty rows
    df = df.dropna(subset=["Day", "Topic", "TEKS", "Skill"], how="all")

    # Fill down missing values in Unit and Day columns
    df["Unit"] = df["Day"].str.extract(r"^(Unit \d+:.*?)(?=Day \d+|$)", expand=False)
    df["Unit"] = df["Unit"].fillna(method="ffill")
    df["Day"] = df["Day"].fillna(method="ffill")

    # Convert test markers ('x') to binary values
    for col in test_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == "x" else 0)

    # Handle rows with "Unit TEKS"
    unit_teks_rows = df[df["TEKS"].str.contains("Unit TEKS", case=False, na=False)]
    for _, row in unit_teks_rows.iterrows():
        rs_value = str(row["R/S/NT"]).strip()
        tested = "R" in rs_value or "S" in rs_value
        status = "was" if tested else "was not"
        print(f"Deleted row with Unit TEKS that {status} tested.")

    # Remove "Unit TEKS" rows
    df = df[~df["TEKS"].str.contains("Unit TEKS", case=False, na=False)]

    # Handle missing TEK values and missing Skill values
    # First get the rows with missing TEKS or Skills
    missing_tek_rows = df[df["TEKS"].isna() | (df["TEKS"] == "")]
    missing_skill_rows = df[df["Skill"].isna() | (df["Skill"] == "")]

    print(f"Found {len(missing_tek_rows)} rows with missing TEK values")
    print(f"Found {len(missing_skill_rows)} rows with missing Skill values")

    # Handle rows with missing values by forward-filling from previous valid rows
    # This assumes that missing values are meant to be the same as the previous row

    # Create a temporary column to identify sections in the file
    df["section"] = df["Day"].str.contains("Unit", na=False).cumsum()

    # Within each section, fill missing values
    for section, group in df.groupby("section"):
        # Get indices for this section
        section_indices = group.index

        # Forward fill TEKS and R/S/NT within section
        last_valid_tek = None
        last_valid_rs = None

        for idx in section_indices:
            if pd.isna(df.at[idx, "TEKS"]) or df.at[idx, "TEKS"] == "":
                if last_valid_tek is not None:
                    df.at[idx, "TEKS"] = last_valid_tek
                    df.at[idx, "R/S/NT"] = last_valid_rs
            else:
                last_valid_tek = df.at[idx, "TEKS"]
                last_valid_rs = df.at[idx, "R/S/NT"]

        # Forward fill Skill within section
        last_valid_skill = None

        for idx in section_indices:
            if pd.isna(df.at[idx, "Skill"]) or df.at[idx, "Skill"] == "":
                if last_valid_skill is not None:
                    df.at[idx, "Skill"] = last_valid_skill
            else:
                last_valid_skill = df.at[idx, "Skill"]

    # Drop the temporary column
    df = df.drop(columns=["section"])

    # Check if we still have missing values
    still_missing_tek = df[df["TEKS"].isna() | (df["TEKS"] == "")]
    still_missing_skill = df[df["Skill"].isna() | (df["Skill"] == "")]

    print(f"After filling: {len(still_missing_tek)} rows with missing TEK values")
    print(f"After filling: {len(still_missing_skill)} rows with missing Skill values")

    # Create new rows for multi-TEK entries
    new_rows = []
    rows_to_drop = []

    for i, row in df.iterrows():
        teks_value = str(row["TEKS"]).strip() if not pd.isna(row["TEKS"]) else ""
        rs_value = str(row["R/S/NT"]).strip() if not pd.isna(row["R/S/NT"]) else ""

        if "," in teks_value:
            # Split the TEKs
            teks_list = [t.strip() for t in teks_value.split(",")]

            # Split R/S/NT values if multiple
            rs_list = []
            if "," in rs_value:
                rs_list = [t.strip() for t in rs_value.split(",")]
                # If we have fewer R/S/NT values than TEKs, repeat the last one
                if len(rs_list) < len(teks_list):
                    rs_list += [rs_list[-1]] * (len(teks_list) - len(rs_list))
            else:
                # If only one R/S/NT value, map it to all TEKs
                rs_list = [rs_value] * len(teks_list)

            # Create new rows for each TEK
            for j, (tek, rs) in enumerate(zip(teks_list, rs_list[: len(teks_list)])):
                new_row = row.copy()
                new_row["TEKS"] = tek
                new_row["R/S/NT"] = rs
                new_rows.append(new_row)

            rows_to_drop.append(i)

    # Drop the original multi-TEK rows
    df = df.drop(index=rows_to_drop)

    # Add the new rows
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Format TEK IDs to match performance CSV headers
    def format_tek_id(tek, rs_nt):
        if pd.isna(tek) or tek == "":
            return ""

        tek = str(tek).strip()
        rs_nt = str(rs_nt).strip()

        # Check if the TEK contains legend text and skip it
        if (
            "Readiness Standard" in tek
            or "Supporting Standard" in tek
            or "Not Tested" in tek
        ):
            print(f"Skipping legend text in TEK: {tek}")
            return ""

        # Determine TEK type
        tek_type = "NT"  # Default
        if rs_nt == "R":
            tek_type = "R"
        elif rs_nt == "S":
            tek_type = "S"
        # Handle case when R/S/NT contains the actual indication like "R = Readiness"
        elif "R =" in rs_nt or "Readiness" in rs_nt:
            tek_type = "R"
        elif "S =" in rs_nt or "Supporting" in rs_nt:
            tek_type = "S"

        # Handle complex TEK formats (e.g., "10Di" -> "6.10(D.i)")
        if re.match(r"^\d+[A-Z]", tek):
            match = re.match(r"(\d+)([A-Z])(.*)", tek)
            if match:
                number, letter, suffix = match.groups()
                if suffix:
                    formatted = f"6.{number}({letter}.{suffix.lower()})"
                else:
                    formatted = f"6.{number}({letter})"
            else:
                formatted = f"6.{tek}"
        else:
            # Handle numeric-only or other formats
            formatted = f"6.{tek}"

        return f"{formatted} [{tek_type}]"

    # Add the formatted TEK_ID column
    df["TEK"] = df.apply(lambda row: format_tek_id(row["TEKS"], row["R/S/NT"]), axis=1)

    # Remove rows with empty TEK after formatting
    empty_tek_rows = df[df["TEK"] == ""]
    print(f"Removing {len(empty_tek_rows)} rows with empty formatted TEK")
    df = df[df["TEK"] != ""]

    # Check for any remaining problematic rows
    print(f"\nFinal checks:")
    print(
        f"Rows containing 'Readiness Standard': {len(df[df.apply(lambda row: 'Readiness Standard' in str(row['Skill']), axis=1)])}"
    )
    print(f"Rows with missing TEK: {len(df[df['TEK'] == ''])}")
    print(
        f"Rows with missing Skill: {len(df[df['Skill'].isna() | (df['Skill'] == '')])}"
    )

    # Reorder columns - keep the correct Skill column
    cols = ["TEK", "Skill"] + test_cols
    df = df[cols]

    print(f"Processed {len(df)} rows with valid TEKs")
    print(f"Columns: {df.columns.tolist()}")
    print(df)

    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")


if __name__ == "__main__":
    d = "../input/medley"
    n = "freq-rla-6-202425"
    clean(f"{d}/{n}.csv", f"{d}/{n}-clean.csv")
