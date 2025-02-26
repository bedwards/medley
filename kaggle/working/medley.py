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

tca = read_csv("perf_tca_clean")
fall_interim = read_csv("perf_fall_interim_clean")
spring_interim = read_csv("perf_spring_interim_clean")
freq = read_csv("freq_rla_6_202425_clean")

# Process frequency data
freq["TEK"] = freq["TEK"].str.replace("\\", "")  # Remove backslashes from TEK column

# Extract unique TEKs from performance data
tca_cols = list(tca.columns)
tca_cols.remove("student_id")
fall_interim_cols = list(fall_interim.columns)
fall_interim_cols.remove("student_id")
spring_interim_cols = list(spring_interim.columns)
spring_interim_cols.remove("student_id")

# Process student data and calculate weighted averages
# Group TCA columns by test number
tca_by_test = {}
for col in tca_cols:
    # Extract test number from format like "TCA 1: 6.2(B) [R]"
    test_num = int(col.split("TCA")[1].split(":")[0].strip()) if "TCA" in col else 0
    if test_num not in tca_by_test:
        tca_by_test[test_num] = []
    tca_by_test[test_num].append(col)

# Calculate weighted TCA average
tca["tca_avg"] = tca.apply(
    lambda row: (
        sum(
            test_num * row[cols].mean(skipna=True) * pd.notna(row[cols]).any()
            for test_num, cols in tca_by_test.items()
        )
        / sum(
            test_num * pd.notna(row[cols]).any()
            for test_num, cols in tca_by_test.items()
        )
        if sum(pd.notna(row[cols]).any() for _, cols in tca_by_test.items()) > 0
        else np.nan
    ),
    axis=1,
)
fall_interim["fall_interim_avg"] = fall_interim[fall_interim_cols].mean(axis=1)
spring_interim["spring_interim_avg"] = spring_interim[spring_interim_cols].mean(axis=1)

# Combine TCA and interim data with more weight on interim
all_students = pd.merge(
    tca[["student_id", "tca_avg"]],
    fall_interim[["student_id", "fall_interim_avg"]],
    on="student_id",
    how="outer",
)
all_students = pd.merge(
    all_students,
    spring_interim[["student_id", "spring_interim_avg"]],
    on="student_id",
    how="outer",
)
all_students["weighted_avg"] = all_students.apply(
    lambda row: (
        # All three scores available
        (
            1 * row["tca_avg"]
            + 2 * row["fall_interim_avg"]
            + 4 * row["spring_interim_avg"]
        )
        / 7
        if pd.notna(row["tca_avg"])
        and pd.notna(row["fall_interim_avg"])
        and pd.notna(row["spring_interim_avg"])
        # Only TCA and fall available (original formula)
        else (
            (1 * row["tca_avg"] + 2 * row["fall_interim_avg"]) / 3
            if pd.notna(row["tca_avg"]) and pd.notna(row["fall_interim_avg"])
            # Only TCA and spring available
            else (
                (1 * row["tca_avg"] + 4 * row["spring_interim_avg"]) / 5
                if pd.notna(row["tca_avg"]) and pd.notna(row["spring_interim_avg"])
                # Only fall and spring available
                else (
                    (2 * row["fall_interim_avg"] + 4 * row["spring_interim_avg"]) / 6
                    if pd.notna(row["fall_interim_avg"])
                    and pd.notna(row["spring_interim_avg"])
                    # Only one score available
                    else (
                        row["spring_interim_avg"]
                        if pd.notna(row["spring_interim_avg"])
                        else (
                            row["fall_interim_avg"]
                            if pd.notna(row["fall_interim_avg"])
                            else row["tca_avg"]
                        )
                    )
                )
            )
        )
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
for tek in tca_cols + fall_interim_cols:
    tek_code = tek.split(":")[1].strip() if ":" in tek else tek

    # Find corresponding frequency data
    freq_row = freq[freq["TEK"] == tek_code]
    if len(freq_row) > 0:
        # Calculate weighted frequency
        staar_cols = [col for col in freq.columns if col.endswith("_staar")]
        spring_cols = [col for col in freq.columns if col.endswith("_spring")]
        other_cols = [
            col for col in freq.columns if col.endswith(("_fall", "_winter", "_field"))
        ]

        # Calculate weights for different test types
        staar_weight = 0
        for i, col in enumerate(sorted(staar_cols, reverse=True)):
            weight = 200 - (i * 10)  # 200, 190, 180...
            staar_weight += (
                freq_row[col].values[0] * weight
                if not pd.isna(freq_row[col].values[0])
                else 0
            )

        spring_weight = 0
        for i, col in enumerate(sorted(spring_cols, reverse=True)):
            weight = 150 - (i * 10)  # 150, 140, 130...
            spring_weight += (
                freq_row[col].values[0] * weight
                if not pd.isna(freq_row[col].values[0])
                else 0
            )

        other_weight = 0
        for i, col in enumerate(sorted(other_cols, reverse=True)):
            year = int(col.split("_")[0])
            weight = 100 - ((2025 - year) * 5)  # 100, 95, 90...
            other_weight += (
                freq_row[col].values[0] * weight
                if not pd.isna(freq_row[col].values[0])
                else 0
            )

        total_weighted_freq = staar_weight + spring_weight + other_weight

        # Calculate simple sum of all times this TEK was tested across all years
        numeric_cols = [col for col in freq.columns if col not in ["TEK", "Skill"]]
        times_tested = sum(
            freq_row[col].values[0]
            for col in numeric_cols
            if not pd.isna(freq_row[col].values[0])
        )

        skill = freq_row["Skill"].values[0]

    else:
        times_tested = 0
        total_weighted_freq = 0
        skill = ""

    # Calculate performance on this TEK
    performances = []
    if tek in tca_cols:
        performances.extend(tca[tek].dropna().tolist())
    if tek in fall_interim_cols:
        performances.extend(fall_interim[tek].dropna().tolist())
    if tek in spring_interim_cols:
        performances.extend(spring_interim[tek].dropna().tolist())

    avg_score = mean(performances) if performances else np.nan

    # Only add if not already in tek_performance or if higher times_tested
    if (
        tek_code not in tek_performance
        or total_weighted_freq > tek_performance[tek_code]["Total_weighted_freq"]
    ):
        tek_performance[tek_code] = {
            "Skill": skill,
            "Times_tested": times_tested,
            "Total_weighted_freq": total_weighted_freq,
            "Weighted_avg_score": avg_score,
        }

# Now process untested TEKs from frequency data
for _, row in freq.iterrows():
    tek_code = row["TEK"]

    # Calculate simple sum of all times this TEK was tested across all years
    numeric_cols = [col for col in freq.columns if col not in ["TEK", "Skill"]]
    times_tested = sum(row[col] for col in numeric_cols if not pd.isna(row[col]))

    # Calculate weighted frequency for untested TEKs too
    staar_cols = [col for col in freq.columns if col.endswith("_staar")]
    spring_cols = [col for col in freq.columns if col.endswith("_spring")]
    other_cols = [
        col for col in freq.columns if col.endswith(("_fall", "_winter", "_field"))
    ]

    # Calculate weights for different test types
    staar_weight = 0
    for i, col in enumerate(sorted(staar_cols, reverse=True)):
        weight = 200 - (i * 10)  # 200, 190, 180...
        staar_weight += row[col] * weight if not pd.isna(row[col]) else 0

    spring_weight = 0
    for i, col in enumerate(sorted(spring_cols, reverse=True)):
        weight = 150 - (i * 10)  # 150, 140, 130...
        spring_weight += row[col] * weight if not pd.isna(row[col]) else 0

    other_weight = 0
    for i, col in enumerate(sorted(other_cols, reverse=True)):
        year = int(col.split("_")[0])
        weight = 100 - ((2025 - year) * 5)  # 100, 95, 90...
        other_weight += row[col] * weight if not pd.isna(row[col]) else 0

    total_weighted_freq = staar_weight + spring_weight + other_weight

    # Skip if times_tested is 0 or if TEK is already in performance data
    if times_tested == 0 or tek_code in tek_performance:
        continue

    # Add to tek_performance with a default score
    all_scores = []
    for col in tca_cols + fall_interim_cols + spring_interim_cols:
        if col in tca_cols:
            all_scores.extend(tca[col].dropna().tolist())
        if col in fall_interim_cols:
            all_scores.extend(fall_interim[col].dropna().tolist())
        if col in spring_interim_cols:
            all_scores.extend(spring_interim[col].dropna().tolist())

    # Use the average of all scores as an estimate
    avg_all = mean(all_scores) if all_scores else 0.5

    tek_performance[tek_code] = {
        "Skill": row["Skill"],
        "Times_tested": times_tested,
        "Total_weighted_freq": total_weighted_freq,
        "Weighted_avg_score": max(0.1, avg_all - 0.1),  # Slightly lower than average
    }

# Create dataframe and sort by times tested and score
tek_priorities = pd.DataFrame(
    [
        {
            "TEK": tek,
            "Skill": data["Skill"],
            "Times_tested": data["Times_tested"],
            "Total_weighted_freq": data["Total_weighted_freq"],
            "Weighted_avg_score": data["Weighted_avg_score"],
        }
        for tek, data in tek_performance.items()
        if pd.notna(data["Weighted_avg_score"])  # Remove TEKs with no performance data
    ]
)

# Create combined priority score (2:1 ratio of frequency to performance)
tek_priorities["Combined_score"] = (
    2 * tek_priorities["Total_weighted_freq"]
) - tek_priorities["Weighted_avg_score"]
tek_priorities = tek_priorities.sort_values(by=["Combined_score"], ascending=False)
tek_priorities["Priority"] = range(1, len(tek_priorities) + 1)
tek_priorities.to_csv(f"{data_dir}/tek_priorities.csv", index=False)

# Calculate group priorities
all_group_priorities = []

# This will store our expanded format with one student per row
expanded_group_priorities = []

# Dictionary to store period-group to TEKs mapping for tracking top-5
group_top_teks = {}

for (period, group), group_df in student_groups_df.groupby(["period", "group"]):
    student_ids = group_df["student_id"].tolist()
    group_key = f"{period}-{group}"

    # Initialize dictionary for this period-group
    group_top_teks[group_key] = []

    # Calculate group performance on each TEK
    group_tek_performance = {}

    for tek in tca_cols + fall_interim_cols + spring_interim_cols:
        tek_code = tek.split(":")[1].strip() if ":" in tek else tek

        # Skip if not in tek_priorities
        if tek_code not in tek_priorities["TEK"].values:
            continue

        # Get TEK info
        priority_row = tek_priorities[tek_priorities["TEK"] == tek_code]
        times_tested = priority_row["Times_tested"].values[0]
        total_weighted_freq = priority_row["Total_weighted_freq"].values[0]
        skill = priority_row["Skill"].values[0]

        # Calculate group performance
        performances = []
        if tek in tca_cols:
            tca_students = tca[tca["student_id"].isin(student_ids)]
            performances.extend(tca_students[tek].dropna().tolist())
        if tek in fall_interim_cols:
            fall_interim_students = fall_interim[
                fall_interim["student_id"].isin(student_ids)
            ]
            performances.extend(fall_interim_students[tek].dropna().tolist())
        if tek in spring_interim_cols:
            spring_interim_students = spring_interim[
                spring_interim["student_id"].isin(student_ids)
            ]
            performances.extend(spring_interim_students[tek].dropna().tolist())

        if performances:
            group_avg = mean(performances)

            group_tek_performance[tek_code] = {
                "TEK": tek_code,
                "Skill": skill,
                "Times_tested": times_tested,
                "Total_weighted_freq": total_weighted_freq,
                "Group_avg_score": group_avg,
            }

    # Sort TEKs by times tested and group average score
    max_freq = (
        max([item[1]["Total_weighted_freq"] for item in group_tek_performance.items()])
        if group_tek_performance
        else 1
    )  # Avoid division by zero

    if max_freq == 0:  # Avoid division by zero
        max_freq = 1

    for tek, data in group_tek_performance.items():
        data["norm_freq"] = data["Total_weighted_freq"] / max_freq
        data["norm_score"] = data["Group_avg_score"]  # Already in 0-1 range

    sorted_teks = sorted(
        group_tek_performance.items(),
        key=lambda x: 2 * x[1]["norm_freq"] - x[1]["norm_score"],
        reverse=True,
    )

    # Store only the top 5 TEKs for this period-group
    top_5_teks = [tek for tek, _ in sorted_teks[:5]]
    group_top_teks[group_key] = top_5_teks

    # Assign priorities
    for priority, (tek, data) in enumerate(sorted_teks, 1):
        group_priority = {
            "period": period,
            "group": group,
            "TEK": data["TEK"],
            "Skill": data["Skill"],
            "Times_tested": data["Times_tested"],
            "Total_weighted_freq": data[
                "norm_freq"
            ],  # Use normalized frequency instead
            "Group_avg_score": data["Group_avg_score"],
            "Priority": priority,
            "Student_ids": str(student_ids),
        }
        all_group_priorities.append(group_priority)

        # Create expanded format with one student per row
        # Only include top-5 TEKs in the expanded format
        if data["TEK"] in top_5_teks:
            for student_id in student_ids:
                expanded_group_priorities.append(
                    {
                        "period": period,
                        "group": group,
                        "TEK": data["TEK"],
                        "Skill": data["Skill"],
                        "Times_tested": data["Times_tested"],
                        "Total_weighted_freq": data["norm_freq"],
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
