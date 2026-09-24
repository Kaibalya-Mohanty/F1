import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "laps.csv"


# ============================================================
# F1 RACES
# ============================================================

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


# ============================================================
# RACE ALIASES
# ============================================================

RACE_ALIASES = {
    "bahrain": "Bahrain 2024",
    "saudi": "Saudi Arabia 2024",
    "saudi arabia": "Saudi Arabia 2024",
    "australia": "Australia 2024",
    "japan": "Japan 2024",
    "china": "China 2024",
    "miami": "Miami 2024",
    "emilia": "Emilia Romagna 2024",
    "emilia romagna": "Emilia Romagna 2024",
    "imola": "Emilia Romagna 2024",
    "monaco": "Monaco 2024",
    "canada": "Canada 2024",
    "spain": "Spain 2024",
    "austria": "Austria 2024",
    "britain": "Great Britain 2024",
    "great britain": "Great Britain 2024",
    "silverstone": "Great Britain 2024",
    "hungary": "Hungary 2024",
    "hungarian": "Hungary 2024",
    "belgium": "Belgium 2024",
    "spa": "Belgium 2024",
    "netherlands": "Netherlands 2024",
    "dutch": "Netherlands 2024",
    "italy": "Italy 2024",
    "monza": "Italy 2024",
    "azerbaijan": "Azerbaijan 2024",
    "baku": "Azerbaijan 2024",
    "singapore": "Singapore 2024",
    "united states": "United States 2024",
    "usa": "United States 2024",
    "us": "United States 2024",
    "america": "United States 2024",
    "mexico": "Mexico 2024",
    "mexican": "Mexico 2024",
    "sao paulo": "São Paulo 2024",
    "são paulo": "São Paulo 2024",
    "brazil": "São Paulo 2024",
    "las vegas": "Las Vegas 2024",
    "vegas": "Las Vegas 2024",
    "qatar": "Qatar 2024",
    "abu dhabi": "Abu Dhabi 2024",
    "yas marina": "Abu Dhabi 2024",
}


# ============================================================
# DRIVER ALIASES
# ============================================================

DRIVER_ALIASES = {
    "max verstappen": "VER",
    "verstappen": "VER",
    "max": "VER",
    "ver": "VER",

    "lando norris": "NOR",
    "norris": "NOR",
    "lando": "NOR",
    "nor": "NOR",

    "charles leclerc": "LEC",
    "leclerc": "LEC",
    "charles": "LEC",
    "lec": "LEC",

    "carlos sainz": "SAI",
    "sainz": "SAI",
    "carlos": "SAI",
    "sai": "SAI",

    "sergio perez": "PER",
    "perez": "PER",
    "checo": "PER",
    "per": "PER",

    "lewis hamilton": "HAM",
    "hamilton": "HAM",
    "lewis": "HAM",
    "ham": "HAM",

    "george russell": "RUS",
    "russell": "RUS",
    "george": "RUS",
    "rus": "RUS",

    "oscar piastri": "PIA",
    "piastri": "PIA",
    "oscar": "PIA",
    "pia": "PIA",

    "fernando alonso": "ALO",
    "alonso": "ALO",
    "fernando": "ALO",
    "alo": "ALO",

    "lance stroll": "STR",
    "stroll": "STR",
    "lance": "STR",
    "str": "STR",

    "esteban ocon": "OCO",
    "ocon": "OCO",
    "esteban": "OCO",
    "oco": "OCO",

    "pierre gasly": "GAS",
    "gasly": "GAS",
    "pierre": "GAS",
    "gas": "GAS",

    "alexander albon": "ALB",
    "albon": "ALB",
    "alex": "ALB",
    "alb": "ALB",

    "yuki tsunoda": "TSU",
    "tsunoda": "TSU",
    "yuki": "TSU",
    "tsu": "TSU",

    "valtteri bottas": "BOT",
    "bottas": "BOT",
    "valtteri": "BOT",
    "bot": "BOT",

    "zhou guanyu": "ZHO",
    "zhou": "ZHO",
    "guanyu": "ZHO",
    "zho": "ZHO",

    "kevin magnussen": "MAG",
    "magnussen": "MAG",
    "kevin": "MAG",
    "mag": "MAG",

    "nico hulkenberg": "HUL",
    "hulkenberg": "HUL",
    "hulken": "HUL",
    "hul": "HUL",

    "daniel ricciardo": "RIC",
    "ricciardo": "RIC",
    "daniel": "RIC",
    "ric": "RIC",

    "logan sargeant": "SAR",
    "sargeant": "SAR",
    "logan": "SAR",
    "sar": "SAR",

    "liam lawson": "LAW",
    "lawson": "LAW",
    "liam": "LAW",
    "law": "LAW",

    "oliver bearman": "BEA",
    "bearman": "BEA",
    "bea": "BEA",

    "franco colapinto": "COL",
    "colapinto": "COL",
    "col": "COL",
}


# ============================================================
# TYRE COMPOUNDS
# ============================================================

COMPOUNDS = {
    "soft": "SOFT",
    "medium": "MEDIUM",
    "hard": "HARD",
    "intermediate": "INTERMEDIATE",
    "inter": "INTERMEDIATE",
    "wet": "WET",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_lap_data():
    """
    Load and clean the F1 lap dataset.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"F1 lap dataset not found at: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "race",
        "driver",
        "lap",
        "lap_time",
        "compound",
        "tyre_age",
        "position",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in laps.csv: {missing_columns}"
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    df["lap"] = pd.to_numeric(
        df["lap"],
        errors="coerce"
    )

    df["lap_time"] = pd.to_numeric(
        df["lap_time"],
        errors="coerce"
    )

    df["tyre_age"] = pd.to_numeric(
        df["tyre_age"],
        errors="coerce"
    )

    df["position"] = pd.to_numeric(
        df["position"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid lap times
    # --------------------------------------------------------

    df = df.dropna(
        subset=["lap_time"]
    )

    df = df[
        df["lap_time"] > 0
    ]

    # --------------------------------------------------------
    # Standardize text
    # --------------------------------------------------------

    df["race"] = (
        df["race"]
        .astype(str)
        .str.strip()
    )

    df["driver"] = (
        df["driver"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["compound"] = (
        df["compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove abnormal lap-time records
    # --------------------------------------------------------
    #
    # F1 lap times differ considerably between circuits, so
    # we do not use one fixed maximum lap time for every race.
    #
    # Instead, calculate the typical lap time for each driver
    # at each race and remove extreme timing records.
    #
    # This prevents records such as 2478 seconds from
    # contaminating:
    #   - average lap time
    #   - tyre performance
    #   - tyre degradation
    #
    # A lap more than 2x the driver's typical lap time at that
    # race is treated as an abnormal timing record.
    # --------------------------------------------------------

    median_lap_time = (
        df.groupby(
            ["race", "driver"]
        )["lap_time"]
        .transform("median")
    )

    df = df[
        df["lap_time"] <= median_lap_time * 2
    ]

    # Remove rows where the reference median was unavailable.
    df = df.dropna(
        subset=["lap_time"]
    )

    return df


# ============================================================
# DETECTION HELPERS
# ============================================================

def detect_race(question):
    """
    Detect a race from the user's question.

    Returns:
        race name such as 'Monaco 2024'
        or None if no race is detected.
    """

    question_lower = question.lower()

    # Sort aliases by length so that more specific phrases
    # are checked before shorter ones.
    sorted_aliases = sorted(
        RACE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, race_name in sorted_aliases:
        if alias in question_lower:
            return race_name

    return None


def detect_driver(question):
    """
    Detect a driver from the user's question.

    Returns:
        3-letter driver code such as VER, HAM, LEC
        or None.
    """

    question_lower = question.lower()

    sorted_aliases = sorted(
        DRIVER_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, driver_code in sorted_aliases:

        # Full names / normal names can be searched as substrings.
        if len(alias) > 3:
            if alias in question_lower:
                return driver_code

        # Driver abbreviations should be matched as words.
        else:
            pattern = rf"\b{re.escape(alias)}\b"

            if re.search(pattern, question_lower):
                return driver_code

    return None


def detect_compound(question):
    """
    Detect tyre compound from the user's question.
    """

    question_lower = question.lower()

    for alias, compound in COMPOUNDS.items():
        if re.search(rf"\b{re.escape(alias)}\b", question_lower):
            return compound

    return None


def detect_lap_number(question):
    """
    Detect a lap number.

    Examples:
        'lap 20' -> 20
        'at lap 35' -> 35
    """

    match = re.search(
        r"\blap\s*(\d+)\b",
        question.lower(),
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# FILTERING
# ============================================================

def filter_data(
    df,
    race=None,
    driver=None,
    compound=None,
    lap_number=None,
):
    """
    Apply optional filters to the dataset.
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

    if lap_number is not None:
        filtered = filtered[
            filtered["lap"] == lap_number
        ]

    return filtered


# ============================================================
# RESPONSE HELPERS
# ============================================================

def success_response(result_type, **kwargs):
    """
    Create a standard successful response.
    """

    response = {
        "type": result_type,
        "status": "success",
    }

    response.update(kwargs)

    return response


def clarification_response(message, **kwargs):
    """
    Create a response when the question needs more information.
    """

    response = {
        "type": "clarification",
        "status": "needs_clarification",
        "message": message,
    }

    response.update(kwargs)

    return response


def no_data_response(message, **kwargs):
    """
    Create a standard no-data response.
    """

    response = {
        "type": "no_data",
        "status": "no_data",
        "message": message,
    }

    response.update(kwargs)

    return response


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_fastest_lap(df, race=None, driver=None):
    """
    Find the fastest lap.

    A race is required because lap times from different
    circuits are not directly comparable.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for the fastest lap. "
            "For example: 'What was the fastest lap at Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    if filtered.empty:
        return no_data_response(
            "No lap data was found for the requested race/driver.",
            requested_race=race,
            requested_driver=driver,
        )

    fastest_index = filtered["lap_time"].idxmin()
    row = filtered.loc[fastest_index]

    result = success_response(
        "fastest_lap",
        race=row["race"],
        driver=row["driver"],
        lap=int(row["lap"]),
        lap_time=round(float(row["lap_time"]), 3),
        compound=row["compound"],
        tyre_age=(
            int(row["tyre_age"])
            if pd.notna(row["tyre_age"])
            else None
        ),
        position=(
            int(row["position"])
            if pd.notna(row["position"])
            else None
        ),
    )

    if race:
        result["requested_race"] = race

    if driver:
        result["requested_driver"] = driver

    return result


def analyze_average_lap_time(df, race=None, driver=None):
    """
    Calculate average lap time.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for the average lap time. "
            "For example: 'What was Verstappen's average lap time "
            "at Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    if filtered.empty:
        return no_data_response(
            "No lap data was found for the requested race/driver.",
            requested_race=race,
            requested_driver=driver,
        )

    average_time = filtered["lap_time"].mean()

    result = success_response(
        "average_lap_time",
        race=race,
        average_lap_time=round(float(average_time), 3),
        laps_analyzed=int(len(filtered)),
    )

    if driver:
        result["driver"] = driver

    return result


def analyze_lap_count(df, race=None, driver=None):
    """
    Count recorded laps.
    """

    if race is None:
        return clarification_response(
            "Please specify the race when asking for a driver's "
            "lap count. For example: 'How many laps did Verstappen "
            "complete at Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    if filtered.empty:
        return no_data_response(
            "No lap data was found for the requested race/driver.",
            requested_race=race,
            requested_driver=driver,
        )

    if driver:
        return success_response(
            "lap_count",
            race=race,
            driver=driver,
            laps=int(len(filtered)),
        )

    counts = (
        filtered.groupby("driver")
        .size()
        .sort_values(ascending=False)
    )

    result = {
        driver_code: int(count)
        for driver_code, count in counts.items()
    }

    return success_response(
        "lap_count",
        race=race,
        result=result,
    )


def analyze_tyre_usage(df, race=None, driver=None):
    """
    Count laps completed on each tyre compound.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for tyre usage analysis. "
            "For example: 'What tyres did Verstappen use at Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    if filtered.empty:
        return no_data_response(
            "No tyre data was found for the requested race/driver.",
            requested_race=race,
            requested_driver=driver,
        )

    usage = (
        filtered.groupby("compound")
        .size()
        .sort_values(ascending=False)
    )

    result = {
        compound: int(count)
        for compound, count in usage.items()
    }

    response = success_response(
        "tyre_usage",
        race=race,
        tyre_laps=result,
    )

    if driver:
        response["driver"] = driver

    return response


def analyze_tyre_performance(df, race=None, driver=None):
    """
    Compare average lap time by tyre compound.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for tyre performance analysis. "
            "For example: 'How did the different tyres perform "
            "for Verstappen at Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    if filtered.empty:
        return no_data_response(
            "No tyre performance data was found.",
            requested_race=race,
            requested_driver=driver,
        )

    performance = (
        filtered.groupby("compound")["lap_time"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )

    result = {}

    for compound, row in performance.iterrows():
        result[compound] = {
            "average_lap_time": round(float(row["mean"]), 3),
            "laps": int(row["count"]),
        }

    response = success_response(
        "tyre_performance",
        race=race,
        result=result,
    )

    if driver:
        response["driver"] = driver

    return response


def analyze_tyre_age(df, race=None, driver=None, lap_number=None):
    """
    Analyze tyre age at a particular lap.
    """

    if race is None:
        return clarification_response(
            "Please specify the race. "
            "For example: 'What was Verstappen's tyre age "
            "at lap 30 in Monaco?'"
        )

    if driver is None:
        return clarification_response(
            "Please specify the driver. "
            "For example: 'What was Verstappen's tyre age "
            "at lap 30 in Monaco?'"
        )

    if lap_number is None:
        return clarification_response(
            "Please specify the lap number. "
            "For example: 'What was Verstappen's tyre age "
            "at lap 30 in Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
        lap_number=lap_number,
    )

    if filtered.empty:
        return no_data_response(
            "No data was found for that driver, race and lap.",
            race=race,
            driver=driver,
            lap=lap_number,
        )

    row = filtered.iloc[0]

    return success_response(
        "tyre_age",
        race=race,
        driver=driver,
        lap=lap_number,
        tyre_age=(
            int(row["tyre_age"])
            if pd.notna(row["tyre_age"])
            else None
        ),
        compound=row["compound"],
    )


def analyze_position(df, race=None, driver=None, lap_number=None):
    """
    Find driver position at a particular lap.
    """

    if race is None:
        return clarification_response(
            "Please specify the race. "
            "For example: 'What was Verstappen's position "
            "at lap 20 in Monaco?'"
        )

    if driver is None:
        return clarification_response(
            "Please specify the driver. "
            "For example: 'What was Verstappen's position "
            "at lap 20 in Monaco?'"
        )

    if lap_number is None:
        return clarification_response(
            "Please specify the lap number. "
            "For example: 'What was Verstappen's position "
            "at lap 20 in Monaco?'"
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
        lap_number=lap_number,
    )

    if filtered.empty:
        return no_data_response(
            "No position data was found for that driver, race "
            "and lap.",
            race=race,
            driver=driver,
            lap=lap_number,
        )

    row = filtered.iloc[0]

    position = (
        int(row["position"])
        if pd.notna(row["position"])
        else None
    )

    return success_response(
        "position_at_lap",
        race=race,
        driver=driver,
        lap=lap_number,
        position=position,
    )


def analyze_position_change(df, race=None, driver=None):
    """
    Compare starting/early position with final recorded position.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for position-change analysis."
        )

    if driver is None:
        return clarification_response(
            "Please specify the driver for position-change analysis."
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    filtered = filtered.dropna(subset=["position"])

    if filtered.empty:
        return no_data_response(
            "No position data was found.",
            race=race,
            driver=driver,
        )

    filtered = filtered.sort_values("lap")

    first_position = int(filtered.iloc[0]["position"])
    last_position = int(filtered.iloc[-1]["position"])

    # Positive means the driver gained positions.
    positions_gained = first_position - last_position

    return success_response(
        "position_change",
        race=race,
        driver=driver,
        starting_position=first_position,
        final_recorded_position=last_position,
        positions_gained=positions_gained,
    )


def analyze_degradation(df, race=None, driver=None):
    """
    Analyze how lap time changes with tyre age.
    """

    if race is None:
        return clarification_response(
            "Please specify the race for tyre degradation analysis."
        )

    if driver is None:
        return clarification_response(
            "Please specify the driver for tyre degradation analysis."
        )

    filtered = filter_data(
        df,
        race=race,
        driver=driver,
    )

    filtered = filtered.dropna(
        subset=["tyre_age", "lap_time"]
    )

    if filtered.empty:
        return no_data_response(
            "No tyre degradation data was found.",
            race=race,
            driver=driver,
        )

    grouped = (
        filtered.groupby("tyre_age")["lap_time"]
        .mean()
        .sort_index()
    )

    degradation = {
        int(age): round(float(avg_time), 3)
        for age, avg_time in grouped.items()
    }

    return success_response(
        "tyre_degradation",
        race=race,
        driver=driver,
        average_lap_time_by_tyre_age=degradation,
    )


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def classify_question(question):
    """
    Determine which analysis should be performed.

    Returns a simple internal analysis type.
    """

    q = question.lower()

    # Position at a specific lap
    if (
        ("position" in q or "place" in q)
        and "lap" in q
    ):
        return "position"

    # Tyre age
    if (
        ("tyre age" in q or "tire age" in q)
        and "lap" in q
    ):
        return "tyre_age"

    # Degradation
    if (
        "degradation" in q
        or "degrade" in q
        or "tyre life" in q
        or "tire life" in q
    ):
        return "degradation"

    # Tyre performance
    if (
        ("tyre" in q or "tire" in q)
        and (
            "performance" in q
            or "perform" in q
            or "average" in q
        )
    ):
        return "tyre_performance"

    # Tyre usage
    if (
        ("tyre" in q or "tire" in q)
        and (
            "used" in q
            or "usage" in q
            or "compound" in q
        )
    ):
        return "tyre_usage"

    # Position change
    if (
        "position" in q
        and (
            "gain" in q
            or "lost" in q
            or "change" in q
        )
    ):
        return "position_change"

    # Fastest lap
    if (
        "fastest lap" in q
        or "quickest lap" in q
        or "best lap" in q
    ):
        return "fastest_lap"

    # Average lap time
    if (
        "average lap time" in q
        or "average lap" in q
        or "mean lap time" in q
    ):
        return "average_lap_time"

    # Number of laps
    if (
        "number of laps" in q
        or "how many laps" in q
        or "lap count" in q
        or "laps did" in q
    ):
        return "lap_count"

    return "unknown"


# ============================================================
# MAIN DATA AGENT
# ============================================================

def analyze_lap_data(question):
    """
    Main entry point for the Data Agent.

    This function:
        1. Loads the dataset
        2. Detects race
        3. Detects driver
        4. Detects compound
        5. Detects lap number
        6. Determines analysis type
        7. Runs deterministic Pandas analysis
    """

    try:
        df = load_lap_data()

    except Exception as error:
        return {
            "type": "error",
            "status": "error",
            "message": str(error),
        }

    race = detect_race(question)
    driver = detect_driver(question)
    compound = detect_compound(question)
    lap_number = detect_lap_number(question)

    analysis_type = classify_question(question)

    # --------------------------------------------------------
    # FASTEST LAP
    # --------------------------------------------------------

    if analysis_type == "fastest_lap":
        return analyze_fastest_lap(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # AVERAGE LAP TIME
    # --------------------------------------------------------

    if analysis_type == "average_lap_time":
        return analyze_average_lap_time(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # LAP COUNT
    # --------------------------------------------------------

    if analysis_type == "lap_count":
        return analyze_lap_count(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # TYRE USAGE
    # --------------------------------------------------------

    if analysis_type == "tyre_usage":
        return analyze_tyre_usage(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # TYRE PERFORMANCE
    # --------------------------------------------------------

    if analysis_type == "tyre_performance":
        return analyze_tyre_performance(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # TYRE AGE
    # --------------------------------------------------------

    if analysis_type == "tyre_age":
        return analyze_tyre_age(
            df,
            race=race,
            driver=driver,
            lap_number=lap_number,
        )

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    if analysis_type == "position":
        return analyze_position(
            df,
            race=race,
            driver=driver,
            lap_number=lap_number,
        )

    # --------------------------------------------------------
    # POSITION CHANGE
    # --------------------------------------------------------

    if analysis_type == "position_change":
        return analyze_position_change(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # DEGRADATION
    # --------------------------------------------------------

    if analysis_type == "degradation":
        return analyze_degradation(
            df,
            race=race,
            driver=driver,
        )

    # --------------------------------------------------------
    # UNKNOWN QUESTION
    # --------------------------------------------------------

    return {
        "type": "unknown",
        "status": "unsupported",
        "message": (
            "I could not determine the requested data analysis. "
            "Try asking about fastest lap, average lap time, "
            "lap count, tyre usage, tyre performance, tyre age, "
            "position, position change, or tyre degradation."
        ),
    }


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What was the fastest lap at Monaco?",
        "What was Verstappen's fastest lap at Monaco?",
        "What was Verstappen's average lap time at Monaco?",
        "How many laps did Verstappen complete at Monaco?",
        "What tyres did Verstappen use at Monaco?",
        "How did the different tyres perform for Verstappen at Monaco?",
        "What was Verstappen's tyre age at lap 30 in Monaco?",
        "What was Verstappen's position at lap 20 in Monaco?",
        "How many positions did Verstappen gain at Monaco?",
        "How did Verstappen's tyre degradation look at Monaco?",

        # Ambiguous question
        "What was Verstappen's position at lap 20?",
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        result = analyze_lap_data(question)

        print(result)