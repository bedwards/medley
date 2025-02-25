#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import random
import ast
from statistics import mean

data_dir = "../input/medley"


def read_csv(n):
    return pd.read_csv(f"{data_dir}/{n}.csv")


# Load data files
periods = read_csv("periods")
periods.columns = ["period", "student_id"]  # Rename columns to match the new format
periods = periods.set_index("student_id")

tca = read_csv("perf-tca-clean")
interim = read_csv("perf-interim-clean")
freq = read_csv("freq-rla-6-202425-clean")

# Process frequency data
freq["TEK"] = freq["TEK"].str.replace("\\", "")  # Remove backslashes from TEK column

# Extract unique TEKs from performance data
tca_cols = list(tca.columns)
tca_cols.remove("student_id")
interim_cols = list(interim.columns)
interim_cols.remove("student_id")

# Process student data and calculate weighted averages
tca["tca_avg"] = tca[tca_cols].mean(axis=1)
interim["interim_avg"] = interim[interim_cols].mean(axis=1)

# Combine TCA and interim data with more weight on interim
all_students = pd.merge(
    tca[["student_id", "tca_avg"]],
    interim[["student_id", "interim_avg"]],
    on="student_id",
    how="outer",
)
all_students["weighted_avg"] = all_students.apply(
    lambda row: (
        0.4 * row["tca_avg"] + 0.6 * row["interim_avg"]
        if pd.notna(row["tca_avg"]) and pd.notna(row["interim_avg"])
        else row["tca_avg"] if pd.notna(row["tca_avg"]) else row["interim_avg"]
    ),
    axis=1,
)

# Add period information
all_students = pd.merge(
    all_students, periods, left_on="student_id", right_index=True, how="inner"
)

# Create student groups within each period
student_groups = []
for period, group_df in all_students.groupby("period"):
    sorted_students = group_df.sort_values("weighted_avg", ascending=False)
    num_students = len(sorted_students)
    group_size = max(1, num_students // 4)  # Divide into 4 groups

    for i, (idx, student) in enumerate(sorted_students.iterrows()):
        group_num = min(i // group_size + 1, 4)  # Limit to max 4 groups
        student_groups.append(
            {
                "period": period,
                "group": group_num,
                "student_id": student["student_id"],
                "Weighted_avg_score": student["weighted_avg"],
            }
        )

student_groups_df = pd.DataFrame(student_groups)
student_groups_df.to_csv(f"{data_dir}/student_groups.csv", index=False)

# Calculate TEK priorities
tek_performance = {}
for tek in tca_cols + interim_cols:
    tek_code = tek.split(":")[1].strip() if ":" in tek else tek

    # Find corresponding frequency data
    freq_row = freq[freq["TEK"] == tek_code]
    if len(freq_row) > 0:
        times_tested = freq_row["2024_4_staar"].values[0]
        skill = freq_row["Skill"].values[0]
    else:
        times_tested = 0
        skill = ""

    # Calculate performance on this TEK
    performances = []
    if tek in tca_cols:
        performances.extend(tca[tek].dropna().tolist())
    if tek in interim_cols:
        performances.extend(interim[tek].dropna().tolist())

    avg_score = mean(performances) if performances else np.nan

    # Only add if not already in tek_performance or if higher times_tested
    if (
        tek_code not in tek_performance
        or times_tested > tek_performance[tek_code]["Times_tested"]
    ):
        tek_performance[tek_code] = {
            "Skill": skill,
            "Times_tested": times_tested,
            "Weighted_avg_score": avg_score,
        }

# Now process untested TEKs from frequency data
for _, row in freq.iterrows():
    tek_code = row["TEK"]
    times_tested = row["2024_4_staar"]

    # Skip if times_tested is 0 or if TEK is already in performance data
    if times_tested == 0 or tek_code in tek_performance:
        continue

    # Add to tek_performance with a default score
    all_scores = []
    for col in tca_cols + interim_cols:
        if col in tca_cols:
            all_scores.extend(tca[col].dropna().tolist())
        if col in interim_cols:
            all_scores.extend(interim[col].dropna().tolist())

    # Use the average of all scores as an estimate
    avg_all = mean(all_scores) if all_scores else 0.5

    tek_performance[tek_code] = {
        "Skill": row["Skill"],
        "Times_tested": times_tested,
        "Weighted_avg_score": max(0.1, avg_all - 0.1),  # Slightly lower than average
    }

# Create dataframe and sort by times tested and score
tek_priorities = pd.DataFrame(
    [
        {
            "TEK": tek,
            "Skill": data["Skill"],
            "Times_tested": data["Times_tested"],
            "Weighted_avg_score": data["Weighted_avg_score"],
        }
        for tek, data in tek_performance.items()
        if pd.notna(data["Weighted_avg_score"])  # Remove TEKs with no performance data
    ]
)

tek_priorities = tek_priorities.sort_values(
    by=["Times_tested", "Weighted_avg_score"], ascending=[False, True]
)
tek_priorities["Priority"] = range(1, len(tek_priorities) + 1)
tek_priorities.to_csv(f"{data_dir}/tek_priorities.csv", index=False)

# Calculate group priorities
all_group_priorities = []
expanded_group_priorities = (
    []
)  # This will store our expanded format with one student per row

for (period, group), group_df in student_groups_df.groupby(["period", "group"]):
    student_ids = group_df["student_id"].tolist()

    # Calculate group performance on each TEK
    group_tek_performance = {}

    for tek in tca_cols + interim_cols:
        tek_code = tek.split(":")[1].strip() if ":" in tek else tek

        # Skip if not in tek_priorities
        if tek_code not in tek_priorities["TEK"].values:
            continue

        # Get TEK info
        priority_row = tek_priorities[tek_priorities["TEK"] == tek_code]
        times_tested = priority_row["Times_tested"].values[0]
        skill = priority_row["Skill"].values[0]

        # Calculate group performance
        performances = []
        if tek in tca_cols:
            tca_students = tca[tca["student_id"].isin(student_ids)]
            performances.extend(tca_students[tek].dropna().tolist())
        if tek in interim_cols:
            interim_students = interim[interim["student_id"].isin(student_ids)]
            performances.extend(interim_students[tek].dropna().tolist())

        if performances:
            group_avg = mean(performances)

            group_tek_performance[tek_code] = {
                "TEK": tek_code,
                "Skill": skill,
                "Times_tested": times_tested,
                "Group_avg_score": group_avg,
            }

    # Sort TEKs by times tested and group average score
    sorted_teks = sorted(
        group_tek_performance.items(),
        key=lambda x: (-x[1]["Times_tested"], x[1]["Group_avg_score"]),
    )

    # Assign priorities
    for priority, (tek, data) in enumerate(sorted_teks, 1):
        group_priority = {
            "period": period,
            "group": group,
            "TEK": data["TEK"],
            "Skill": data["Skill"],
            "Times_tested": data["Times_tested"],
            "Group_avg_score": data["Group_avg_score"],
            "Priority": priority,
            "Student_ids": str(student_ids),  # Keep original format for reference
        }
        all_group_priorities.append(group_priority)

        # Create expanded format with one student per row
        for student_id in student_ids:
            expanded_group_priorities.append(
                {
                    "period": period,
                    "group": group,
                    "TEK": data["TEK"],
                    "Skill": data["Skill"],
                    "Times_tested": data["Times_tested"],
                    "Group_avg_score": data["Group_avg_score"],
                    "Priority": priority,
                    "Student_id": student_id,  # Single student ID per row
                }
            )

# Save the expanded format instead of the original format
group_priorities_df = pd.DataFrame(expanded_group_priorities)
group_priorities_df.to_csv(f"{data_dir}/group_priorities.csv", index=False)

print("Analysis complete. Files generated:")
print("- student_groups.csv")
print("- tek_priorities.csv")
print("- group_priorities.csv")
