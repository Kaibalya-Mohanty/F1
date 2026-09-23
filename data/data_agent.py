import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "laps.csv"


# ============================================================
# REFERENCE DATA
# ============================================================

# 2024 races available in our dataset
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


# Driver name aliases
DRIVER_ALIASES = {
    "max verstappen": "VER",
    "verstappen": "VER",
    "max": "VER",

    "lando norris": "NOR",
    "norris": "NOR",
    "lando": "NOR",

    "charles leclerc": "LEC",
    "leclerc": "LEC",
    "charles": "LEC",

    "carlos sainz": "SAI",
    "sainz": "SAI",
    "carlos": "SAI",

    "lewis hamilton": "HAM",
    "hamilton": "HAM",
    "lewis": "HAM",

    "george russell": "RUS",
    "russell": "RUS",
    "george": "RUS",

    "oscar piastri": "PIA",
    "piastri": "PIA",
    "oscar": "PIA",

    "fernando alonso": "ALO",
    "alonso": "ALO",
    "fernando": "ALO",

    "esteban ocon": "OCO",
    "ocon": "OCO",

    "pierre gasly": "GAS",
    "gasly": "GAS",

    "alexander albon": "ALB",
    "albon": "ALB",

    "yuki tsunoda": "TSU",
    "tsunoda": "TSU",

    "lance stroll": "STR",
    "stroll": "STR",

    "valtteri bottas": "BOT",
    "bottas": "BOT",

    "zhou guanyu": "ZHO",
    "zhou": "ZHO",

    "daniel ricciardo": "RIC",
    "ricciardo": "RIC",

    "kevin magnussen": "MAG",
    "magnussen": "MAG",

    "nico hulkenberg": "HUL",
    "hulkenberg": "HUL",

    "logan sargeant": "SAR",
    "sargeant": "SAR",

    "franco colapinto": "COL",
    "colapinto": "COL",

    "liam lawson": "LAW",
    "lawson": "LAW",

    "oliver bearman": "BEA",
    "bearman": "BEA",
}


COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
    "INTERMEDIATE",
    "WET",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_lap_data():
    """
    Load and clean the F1 lap dataset.
    """

    df = pd.read_csv(DATA_FILE)

    # Convert numeric columns
    numeric_columns = [
        "lap",
        "lap_time",
        "tyre_age",
        "position",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Clean strings
    df["race"] = df["race"].astype(str).str.strip()
    df["driver"] = df["driver"].astype(str).str.strip().str.upper()
    df["compound"] = (
        df["compound"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Remove invalid lap times
    df = df[
        df["lap_time"].notna()
        & (df["lap_time"] > 0)
    ]

    return df


# ============================================================
# QUESTION HELPERS
# ============================================================

def detect_race(question):
    """
    Detect the 2024 race mentioned in the question.

    Returns:
        race string such as "Monaco 2024"
        or None if no race was specified.
    """

    question_lower = question.lower()

    for race in RACES:

        race_lower = race.lower()

        if race_lower in question_lower:

            return f"{race} 2024"

    return None


def detect_driver(question):
    """
    Detect driver abbreviation from a natural-language question.
    """

    question_lower = question.lower()

    # Check full names / aliases first
    for name, abbreviation in DRIVER_ALIASES.items():

        if name in question_lower:
            return abbreviation

    # Check abbreviations
    for abbreviation in DRIVER_ALIASES.values():

        if abbreviation.lower() in question_lower.split():
            return abbreviation

    return None


def detect_compound(question):
    """
    Detect tyre compound from the question.
    """

    question_upper = question.upper()

    for compound in COMPOUNDS:

        if compound in question_upper:
            return compound

    return None


def detect_lap_number(question):
    """
    Detect a lap number in questions such as:

    "What was Verstappen's position on lap 30?"
    """

    words = question.lower().split()

    for i, word in enumerate(words):

        if word == "lap" and i + 1 < len(words):

            try:
                return int(words[i + 1].strip("?,."))
            except ValueError:
                pass

    return None


# ============================================================
# FILTERING
# ============================================================

def filter_data(
    df,
    race=None,
    driver=None,
    compound=None,
):
    """
    Apply optional race, driver and compound filters.
    """

    filtered = df.copy()

    if race:
        filtered = filtered[
            filtered["race"].str.lower() == race.lower()
        ]

    if driver:
        filtered = filtered[
            filtered["driver"].str.upper() == driver.upper()
        ]

    if compound:
        filtered = filtered[
            filtered["compound"].str.upper() == compound.upper()
        ]

    return filtered


# ============================================================
# FASTEST LAP
# ============================================================

def analyze_fastest_lap(df):
    """
    Find the fastest valid lap.
    """

    if df.empty:
        return {
            "type": "fastest_lap",
            "status": "no_data",
            "message": "No matching lap data was found."
        }

    idx = df["lap_time"].idxmin()
    row = df.loc[idx]

    return {
        "type": "fastest_lap",
        "status": "success",
        "race": row["race"],
        "driver": row["driver"],
        "lap": int(row["lap"]),
        "lap_time": round(float(row["lap_time"]), 3),
        "compound": row["compound"],
        "tyre_age": (
            int(row["tyre_age"])
            if pd.notna(row["tyre_age"])
            else None
        ),
        "position": (
            int(row["position"])
            if pd.notna(row["position"])
            else None
        ),
    }


# ============================================================
# AVERAGE LAP TIME
# ============================================================

def analyze_average_lap_time(df):
    """
    Calculate average lap time.

    If a driver is specified, return that driver's average.

    Otherwise return averages for all drivers.
    """

    if df.empty:
        return {
            "type": "average_lap_time",
            "status": "no_data",
            "message": "No matching lap data was found."
        }

    result = (
        df.groupby("driver")["lap_time"]
        .mean()
        .sort_values()
        .round(3)
    )

    return {
        "type": "average_lap_time",
        "status": "success",
        "result": result.to_dict()
    }


# ============================================================
# LAP COUNT
# ============================================================

def analyze_lap_count(df):
    """
    Count valid laps per driver.
    """

    if df.empty:
        return {
            "type": "lap_count",
            "status": "no_data",
            "message": "No matching lap data was found."
        }

    result = (
        df.groupby("driver")
        .size()
        .sort_values(ascending=False)
    )

    return {
        "type": "lap_count",
        "status": "success",
        "result": result.astype(int).to_dict()
    }


# ============================================================
# TYRE USAGE
# ============================================================

def analyze_tyre_usage(df):
    """
    Analyze tyre compound usage.
    """

    if df.empty:
        return {
            "type": "tyre_usage",
            "status": "no_data",
            "message": "No matching tyre data was found."
        }

    result = (
        df.groupby("compound")
        .size()
        .sort_values(ascending=False)
    )

    return {
        "type": "tyre_usage",
        "status": "success",
        "laps_by_compound": result.astype(int).to_dict()
    }


# ============================================================
# TYRE PERFORMANCE
# ============================================================

def analyze_tyre_performance(df):
    """
    Compare average lap times by tyre compound.
    """

    if df.empty:
        return {
            "type": "tyre_performance",
            "status": "no_data",
            "message": "No matching tyre data was found."
        }

    result = (
        df.groupby("compound")["lap_time"]
        .mean()
        .sort_values()
        .round(3)
    )

    return {
        "type": "tyre_performance",
        "status": "success",
        "average_lap_time_by_compound": result.to_dict()
    }


# ============================================================
# TYRE AGE
# ============================================================

def analyze_tyre_age(df):
    """
    Analyze tyre age.
    """

    if df.empty:
        return {
            "type": "tyre_age",
            "status": "no_data",
            "message": "No matching tyre data was found."
        }

    average_age = df["tyre_age"].mean()
    maximum_age = df["tyre_age"].max()

    return {
        "type": "tyre_age",
        "status": "success",
        "average_tyre_age": (
            round(float(average_age), 2)
            if pd.notna(average_age)
            else None
        ),
        "maximum_tyre_age": (
            int(maximum_age)
            if pd.notna(maximum_age)
            else None
        ),
    }


# ============================================================
# POSITION ANALYSIS
# ============================================================

def analyze_position(df, lap_number=None):
    """
    Analyze driver positions.

    If lap_number is specified:
        return position at that lap.

    Otherwise:
        return average position by driver.
    """

    if df.empty:
        return {
            "type": "position",
            "status": "no_data",
            "message": "No matching position data was found."
        }

    # Remove missing positions
    position_df = df[df["position"].notna()].copy()

    if position_df.empty:
        return {
            "type": "position",
            "status": "no_data",
            "message": "No position data was found."
        }

    # Specific lap
    if lap_number is not None:

        lap_df = position_df[
            position_df["lap"] == lap_number
        ]

        if lap_df.empty:
            return {
                "type": "position_at_lap",
                "status": "no_data",
                "lap": lap_number,
                "message": "No position data was found for that lap."
            }

        result = {}

        for _, row in lap_df.iterrows():

            result[row["driver"]] = int(row["position"])

        return {
            "type": "position_at_lap",
            "status": "success",
            "lap": lap_number,
            "result": result,
        }

    # Average position
    result = (
        position_df
        .groupby("driver")["position"]
        .mean()
        .sort_values()
        .round(2)
    )

    return {
        "type": "average_position",
        "status": "success",
        "result": result.to_dict()
    }


# ============================================================
# POSITION CHANGE
# ============================================================

def analyze_position_change(df):
    """
    Calculate position change between the first
    and last recorded lap for each driver.

    Positive value = gained positions.
    Negative value = lost positions.
    """

    if df.empty:
        return {
            "type": "position_change",
            "status": "no_data",
            "message": "No matching data was found."
        }

    position_df = df[df["position"].notna()].copy()

    if position_df.empty:
        return {
            "type": "position_change",
            "status": "no_data",
            "message": "No position data was found."
        }

    result = {}

    for driver, driver_df in position_df.groupby("driver"):

        driver_df = driver_df.sort_values("lap")

        first_position = driver_df.iloc[0]["position"]
        last_position = driver_df.iloc[-1]["position"]

        change = first_position - last_position

        result[driver] = {
            "starting_position": int(first_position),
            "ending_position": int(last_position),
            "positions_gained": int(change),
        }

    return {
        "type": "position_change",
        "status": "success",
        "result": result,
    }


# ============================================================
# TYRE AGE VS LAP TIME
# ============================================================

def analyze_degradation(df):
    """
    Analyze how average lap time changes with tyre age.

    This is a simple descriptive analysis, not a
    full tyre degradation model.
    """

    if df.empty:
        return {
            "type": "tyre_degradation",
            "status": "no_data",
            "message": "No matching data was found."
        }

    degradation_df = df[
        df["tyre_age"].notna()
        & df["lap_time"].notna()
    ].copy()

    if degradation_df.empty:
        return {
            "type": "tyre_degradation",
            "status": "no_data",
            "message": "No tyre-age data was found."
        }

    result = (
        degradation_df
        .groupby("tyre_age")["lap_time"]
        .mean()
        .sort_index()
        .round(3)
    )

    return {
        "type": "tyre_degradation",
        "status": "success",
        "average_lap_time_by_tyre_age": result.to_dict()
    }


# ============================================================
# MAIN DATA AGENT
# ============================================================

def analyze_lap_data(question):
    """
    Main entry point for the Data Agent.

    Takes a natural-language F1 data question
    and returns a structured result.
    """

    df = load_lap_data()

    question_lower = question.lower()

    # --------------------------------------------------------
    # Detect entities
    # --------------------------------------------------------

    race = detect_race(question)
    driver = detect_driver(question)
    compound = detect_compound(question)
    lap_number = detect_lap_number(question)

    # --------------------------------------------------------
    # Determine analysis type
    # --------------------------------------------------------

    # Fastest lap
    if (
        "fastest lap" in question_lower
        or "quickest lap" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
            compound=compound,
        )

        result = analyze_fastest_lap(filtered)

        if race:
            result["requested_race"] = race

        if driver:
            result["requested_driver"] = driver

        return result

    # Average lap time
    if (
        "average lap time" in question_lower
        or "average lap" in question_lower
        or "mean lap time" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
            compound=compound,
        )

        result = analyze_average_lap_time(filtered)

        if race:
            result["requested_race"] = race

        if driver:
            result["requested_driver"] = driver

        return result

    # Lap count
    if (
        "number of laps" in question_lower
        or "how many laps" in question_lower
        or "lap count" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
            compound=compound,
        )

        return analyze_lap_count(filtered)

    # Tyre compound usage
    if (
        "tyre usage" in question_lower
        or "tire usage" in question_lower
        or "which tyres" in question_lower
        or "which tires" in question_lower
        or "tyre compound" in question_lower
        or "tire compound" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
        )

        return analyze_tyre_usage(filtered)

    # Tyre performance
    if (
        "tyre performance" in question_lower
        or "tire performance" in question_lower
        or "fastest tyre" in question_lower
        or "fastest tire" in question_lower
        or "tyre lap time" in question_lower
        or "tire lap time" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
        )

        return analyze_tyre_performance(filtered)

    # Tyre age
    if (
        "tyre age" in question_lower
        or "tire age" in question_lower
        or "tyre life" in question_lower
        or "tire life" in question_lower
        or "old tyres" in question_lower
        or "old tires" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
            compound=compound,
        )

        return analyze_tyre_age(filtered)

    # Tyre degradation
    if (
        "degradation" in question_lower
        or "degrade" in question_lower
        or "lap time change" in question_lower
        or "pace drop" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
            compound=compound,
        )

        return analyze_degradation(filtered)

    # Position change
    if (
        "position change" in question_lower
        or "positions gained" in question_lower
        or "positions lost" in question_lower
        or "positions did" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
        )

        return analyze_position_change(filtered)

    # Position at a particular lap
    if (
        "position" in question_lower
        and lap_number is not None
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
        )

        return analyze_position(
            filtered,
            lap_number=lap_number
        )

    # Average position
    if (
        "average position" in question_lower
        or "average race position" in question_lower
    ):

        filtered = filter_data(
            df,
            race=race,
            driver=driver,
        )

        return analyze_position(filtered)

    # --------------------------------------------------------
    # Unknown question
    # --------------------------------------------------------

    return {
        "type": "unknown",
        "status": "unsupported",
        "message": (
            "The Data Agent does not understand this "
            "question yet."
        ),
        "supported_analysis": [
            "fastest lap",
            "average lap time",
            "lap count",
            "tyre usage",
            "tyre performance",
            "tyre age",
            "tyre degradation",
            "position",
            "position change",
        ],
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What is the fastest lap?",
        "What was the fastest lap at Monaco?",
        "What was Max Verstappen's fastest lap at Monaco?",
        "What is the average lap time at Monaco?",
        "What was Lando Norris's average lap time?",
        "Which tyres were used at Monaco?",
        "What was the average lap time on Medium tyres?",
        "What was the maximum tyre age at Monaco?",
        "What was Verstappen's position on lap 30?",
        "Which drivers gained positions?",
        "How did tyre degradation change with tyre age?",
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = analyze_lap_data(question)

        print(result)