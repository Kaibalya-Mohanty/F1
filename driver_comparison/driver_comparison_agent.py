import os
import re
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured in the .env file.")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "laps.csv"
)


# ---------------------------------------------------------
# Driver name mapping
# ---------------------------------------------------------

DRIVER_ALIASES = {
    "verstappen": "VER",
    "max verstappen": "VER",
    "ver": "VER",

    "norris": "NOR",
    "lando norris": "NOR",
    "nor": "NOR",

    "leclerc": "LEC",
    "charles leclerc": "LEC",
    "lec": "LEC",

    "hamilton": "HAM",
    "lewis hamilton": "HAM",
    "ham": "HAM",

    "alonso": "ALO",
    "fernando alonso": "ALO",
    "alo": "ALO",

    "piastri": "PIA",
    "oscar piastri": "PIA",
    "pia": "PIA",

    "russell": "RUS",
    "george russell": "RUS",
    "rus": "RUS",

    "sainz": "SAI",
    "carlos sainz": "SAI",
    "sai": "SAI",

    "perez": "PER",
    "sergio perez": "PER",
    "per": "PER",

    "gasly": "GAS",
    "pierre gasly": "GAS",
    "gas": "GAS",

    "ocon": "OCO",
    "esteban ocon": "OCO",
    "oco": "OCO",

    "tsunoda": "TSU",
    "yuki tsunoda": "TSU",
    "tsu": "TSU",

    "albon": "ALB",
    "alex albon": "ALB",
    "alb": "ALB",

    "stroll": "STR",
    "lance stroll": "STR",
    "str": "STR",

    "bottas": "BOT",
    "valtteri bottas": "BOT",
    "bot": "BOT",

    "hulkenberg": "HUL",
    "nico hulkenberg": "HUL",
    "hul": "HUL",

    "magnussen": "MAG",
    "kevin magnussen": "MAG",
    "mag": "MAG",

    "ricciardo": "RIC",
    "daniel ricciardo": "RIC",
    "ric": "RIC",

    "lawson": "LAW",
    "liam lawson": "LAW",
    "law": "LAW",

    "bearman": "BEA",
    "oliver bearman": "BEA",
    "bea": "BEA",

    "colapinto": "COL",
    "franco colapinto": "COL",
    "col": "COL",

    "zhou": "ZHO",
    "guanyu zhou": "ZHO",
    "zho": "ZHO",

    "sargeant": "SAR",
    "logan sargeant": "SAR",
    "sar": "SAR",

    "alpine": "ALP",
}


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "race",
        "driver",
        "lap",
        "lap_time",
        "compound",
        "tyre_age",
        "position",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in laps.csv: {missing}"
        )

    return df


# ---------------------------------------------------------
# Convert lap time to seconds
# ---------------------------------------------------------

def lap_time_to_seconds(value):
    """
    Converts common lap-time formats into seconds.

    Examples:
        78.592 -> 78.592
        "78.592" -> 78.592
        "1:18.592" -> 78.592
    """

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    try:
        return float(value)
    except ValueError:
        pass

    match = re.match(
        r"^(\d+):(\d+(?:\.\d+)?)$",
        value
    )

    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))

        return minutes * 60 + seconds

    return None


# ---------------------------------------------------------
# Normalize driver name
# ---------------------------------------------------------

def normalize_driver(driver_name, available_drivers):
    if not driver_name:
        return None

    cleaned = driver_name.lower().strip()

    # Direct alias lookup
    if cleaned in DRIVER_ALIASES:
        code = DRIVER_ALIASES[cleaned]

        if code in available_drivers:
            return code

    # Direct code
    upper = driver_name.upper().strip()

    if upper in available_drivers:
        return upper

    # Partial matching
    for alias, code in DRIVER_ALIASES.items():

        if (
            alias in cleaned
            or cleaned in alias
        ):
            if code in available_drivers:
                return code

    return None


# ---------------------------------------------------------
# Extract drivers and race from user question
# ---------------------------------------------------------

def extract_entities(question, df):
    question_lower = question.lower()

    available_drivers = set(
        df["driver"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    detected_drivers = []

    for alias, code in DRIVER_ALIASES.items():

        if alias in question_lower:

            if (
                code in available_drivers
                and code not in detected_drivers
            ):
                detected_drivers.append(code)

    # Also detect direct driver codes
    for code in available_drivers:

        if re.search(
            rf"\b{re.escape(code.lower())}\b",
            question_lower
        ):

            if code not in detected_drivers:
                detected_drivers.append(code)

    # Detect race
    races = df["race"].dropna().unique().tolist()

    detected_race = None

    for race in races:

        race_lower = str(race).lower()

        if race_lower in question_lower:
            detected_race = race
            break

        # Allow "Monaco" to match "Monaco 2024"
        race_name_without_year = re.sub(
            r"\s+\d{4}$",
            "",
            race_lower
        )

        if race_name_without_year in question_lower:
            detected_race = race
            break

    return detected_drivers, detected_race


# ---------------------------------------------------------
# Calculate driver statistics
# ---------------------------------------------------------

def calculate_driver_stats(df, driver_code, race):

    driver_df = df[
        (df["driver"].astype(str).str.upper() == driver_code)
        & (df["race"] == race)
    ].copy()

    if driver_df.empty:
        return {
            "driver": driver_code,
            "race": race,
            "status": "no_data"
        }

    # ---------------------------------------------------------
    # Convert lap times to seconds
    # ---------------------------------------------------------

    driver_df["lap_time_seconds"] = (
        driver_df["lap_time"].apply(lap_time_to_seconds)
    )

    # Count records that could not be converted
    invalid_format_count = int(
        driver_df["lap_time_seconds"].isna().sum()
    )

    # ---------------------------------------------------------
    # Remove invalid / anomalous lap times
    #
    # Extremely large values such as 2468 seconds and
    # 2478 seconds are timing artifacts and should not
    # influence the driver statistics.
    # ---------------------------------------------------------

    MIN_VALID_LAP_TIME = 50
    MAX_VALID_LAP_TIME = 150

    valid_laps = driver_df[
        driver_df["lap_time_seconds"].notna()
        & (driver_df["lap_time_seconds"] >= MIN_VALID_LAP_TIME)
        & (driver_df["lap_time_seconds"] <= MAX_VALID_LAP_TIME)
    ].copy()

    excluded_laps_count = len(driver_df) - len(valid_laps)

    if valid_laps.empty:
        return {
            "driver": driver_code,
            "race": race,
            "status": "no_valid_lap_times"
        }

    # ---------------------------------------------------------
    # Basic lap statistics
    # ---------------------------------------------------------

    average_lap_time = valid_laps[
        "lap_time_seconds"
    ].mean()

    fastest_lap = valid_laps[
        "lap_time_seconds"
    ].min()

    # ---------------------------------------------------------
    # Average race position
    # ---------------------------------------------------------

    average_position = None

    if "position" in valid_laps.columns:

        positions = pd.to_numeric(
            valid_laps["position"],
            errors="coerce"
        ).dropna()

        if not positions.empty:
            average_position = positions.mean()

    # ---------------------------------------------------------
    # Average tyre age
    # ---------------------------------------------------------

    average_tyre_age = None

    if "tyre_age" in valid_laps.columns:

        tyre_ages = pd.to_numeric(
            valid_laps["tyre_age"],
            errors="coerce"
        ).dropna()

        if not tyre_ages.empty:
            average_tyre_age = tyre_ages.mean()

    # ---------------------------------------------------------
    # Compound statistics
    # ---------------------------------------------------------

    compound_stats = {}

    if "compound" in valid_laps.columns:

        compound_groups = (
            valid_laps
            .dropna(subset=["compound"])
            .groupby("compound")["lap_time_seconds"]
            .agg(["mean", "min", "count"])
        )

        for compound, row in compound_groups.iterrows():

            compound_stats[str(compound)] = {
                "average_lap_time": round(
                    float(row["mean"]),
                    3
                ),

                "fastest_lap": round(
                    float(row["min"]),
                    3
                ),

                "laps": int(row["count"])
            }

    # ---------------------------------------------------------
    # Return statistics
    # ---------------------------------------------------------

    return {
        "driver": driver_code,
        "race": race,
        "status": "success",

        "laps_analyzed": int(len(valid_laps)),

        "average_lap_time": round(
            float(average_lap_time),
            3
        ),

        "fastest_lap": round(
            float(fastest_lap),
            3
        ),

        "average_position": (
            round(float(average_position), 2)
            if average_position is not None
            else None
        ),

        "average_tyre_age": (
            round(float(average_tyre_age), 2)
            if average_tyre_age is not None
            else None
        ),

        "compound_stats": compound_stats,

        # -----------------------------------------------------
        # Data quality information
        # -----------------------------------------------------

        "data_quality": {
            "valid_laps": int(len(valid_laps)),
            "excluded_laps": int(excluded_laps_count),
            "invalid_format_laps": int(invalid_format_count),
            "valid_lap_time_range_seconds": [
                MIN_VALID_LAP_TIME,
                MAX_VALID_LAP_TIME
            ]
        }
    }


# ---------------------------------------------------------
# Generate comparison
# ---------------------------------------------------------

def generate_comparison(
    question,
    driver_1_stats,
    driver_2_stats
):

    prompt = f"""
You are the Driver Comparison Agent for an F1 Strategy Copilot.

User question:
{question}

The following statistics were calculated directly from the race dataset.

Driver 1:
{json.dumps(driver_1_stats, indent=2)}

Driver 2:
{json.dumps(driver_2_stats, indent=2)}

Create a concise factual comparison.

Rules:

1. Use only the supplied statistics.

2. Do not invent missing data.

3. Clearly identify which driver had the lower average lap time.

4. Compare fastest laps when available.

5. Mention lap counts.

6. Discuss tyre/compound differences only if the data supports it.

7. If the data is insufficient for a conclusion, explicitly say so.

8. Do not make claims about race outcome unless supported by the data.

9. Mention when anomalous or invalid lap-time records were excluded.

10. Treat data-quality information as a limitation,
not as race-performance evidence.

11. Do not claim that a driver was faster solely because
of a tyre compound unless the supplied data supports
that comparison.

12. Distinguish directly observed dataset values from
interpretations or possible explanations.

Return plain text with these sections:

Comparison

Key Differences

Data Limitations
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual Formula 1 "
                    "data comparison assistant."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------

def compare_drivers(question):

    df = load_data()

    drivers, race = extract_entities(
        question,
        df
    )

    # ---------------------------------------------------------
    # Validate driver detection
    # ---------------------------------------------------------

    if len(drivers) < 2:

        return {
            "status": "error",
            "message": (
                "I could not identify two drivers in the question. "
                "Please specify two drivers, for example: "
                "'Compare Verstappen and Norris at Monaco.'"
            )
        }

    # ---------------------------------------------------------
    # Validate race detection
    # ---------------------------------------------------------

    if race is None:

        return {
            "status": "error",
            "message": (
                "I could not identify the race. "
                "Please specify a race, for example: "
                "'Compare Verstappen and Norris at Monaco.'"
            )
        }

    # ---------------------------------------------------------
    # Select two drivers
    # ---------------------------------------------------------

    driver_1 = drivers[0]
    driver_2 = drivers[1]

    # ---------------------------------------------------------
    # Calculate statistics
    # ---------------------------------------------------------

    driver_1_stats = calculate_driver_stats(
        df,
        driver_1,
        race
    )

    driver_2_stats = calculate_driver_stats(
        df,
        driver_2,
        race
    )

    # ---------------------------------------------------------
    # Validate driver 1
    # ---------------------------------------------------------

    if driver_1_stats["status"] != "success":

        return {
            "status": "error",
            "message": (
                f"No valid data was found for "
                f"{driver_1} at {race}."
            )
        }

    # ---------------------------------------------------------
    # Validate driver 2
    # ---------------------------------------------------------

    if driver_2_stats["status"] != "success":

        return {
            "status": "error",
            "message": (
                f"No valid data was found for "
                f"{driver_2} at {race}."
            )
        }

    # ---------------------------------------------------------
    # Generate LLM comparison
    # ---------------------------------------------------------

    comparison = generate_comparison(
        question,
        driver_1_stats,
        driver_2_stats
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "status": "success",
        "race": race,

        "drivers": [
            driver_1_stats,
            driver_2_stats
        ],

        "comparison": comparison
    }


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "Compare Verstappen and Norris at Monaco"
    )

    result = compare_drivers(question)

    print("\n" + "=" * 70)
    print("DRIVER COMPARISON TEST")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )