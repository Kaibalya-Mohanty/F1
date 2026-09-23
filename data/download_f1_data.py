import fastf1
import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "laps.csv"


# Enable FastF1 cache
CACHE_DIR = PROJECT_ROOT / "f1cache"
CACHE_DIR.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


# 2024 F1 calendar
RACES = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canada",
    "Spain",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]


all_laps = []


for race in RACES:

    print(f"\nLoading 2024 {race}...")

    try:
        # Load the race session
        session = fastf1.get_session(2024, race, "R")
        session.load()

        laps = session.laps.copy()

        # Keep only the columns we need
        required_columns = [
            "Driver",
            "LapNumber",
            "LapTime",
            "Compound",
            "TyreLife",
            "Position",
        ]

        laps = laps[required_columns]

        # Remove rows without lap time
        laps = laps.dropna(subset=["LapTime"])

        # Convert lap time from timedelta to seconds
        laps["LapTime"] = laps["LapTime"].dt.total_seconds()

        # Add race name
        laps["Race"] = f"{race} 2024"

        # Rename columns for our project
        laps = laps.rename(
            columns={
                "Driver": "driver",
                "LapNumber": "lap",
                "LapTime": "lap_time",
                "Compound": "compound",
                "TyreLife": "tyre_age",
                "Position": "position",
                "Race": "race",
            }
        )

        # Reorder columns
        laps = laps[
            [
                "race",
                "driver",
                "lap",
                "lap_time",
                "compound",
                "tyre_age",
                "position",
            ]
        ]

        all_laps.append(laps)

        print(f"Loaded {len(laps)} laps.")

    except Exception as e:
        print(f"Could not load {race}: {e}")


# Combine all races
if all_laps:

    final_df = pd.concat(all_laps, ignore_index=True)

    # Sort data
    final_df = final_df.sort_values(
        by=["race", "driver", "lap"]
    )

    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n--------------------------------")
    print("F1 DATASET CREATED")
    print("--------------------------------")
    print(f"Rows: {len(final_df)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nColumns:")
    print(final_df.columns.tolist())

else:
    print("No race data was downloaded.")