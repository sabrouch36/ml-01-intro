"""app_sabri.py - Sabri's modified regression project.

A supervised regression project using study habits to predict student scores.
This file is a personal copy of the provided example and includes a custom
technical modification.

Author: Sabri Hamdaoui
Date: 2026-07

Process:
    - Load a CSV dataset.
    - Inspect and validate the data.
    - Create a derived study-engagement feature.
    - Train and evaluate a supervised regression model.
    - Predict one new case.
    - Create useful charts.

Data Source:
    data/raw/hours_scores_sabri.csv

Run from the project root folder with:
    uv run python -m mlstudio.app_sabri
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from datafun_toolkit.logger import get_logger, log_header
except ImportError:

    def get_logger(
        name: str = "ML",
        level: str | int = "INFO",
    ) -> logging.Logger:
        """Return a basic logger when datafun_toolkit is unavailable."""
        level_value = (
            getattr(logging, level.upper(), logging.INFO)
            if isinstance(level, str)
            else level
        )

        logger = logging.getLogger(name)
        logger.setLevel(level_value)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log_header(logger: logging.Logger, title: str) -> None:
        """Log a simple section header."""
        logger.info("=" * 40)
        logger.info(title)
        logger.info("=" * 40)


# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

# datafun_toolkit expects the logging level as text, such as "DEBUG".
LOG: logging.Logger = get_logger("ML", level="DEBUG")


# === Section 1c. GLOBAL CONSTANTS AND CONFIGURATION ===

DATASET_NAME: Final[str] = "hours_scores_sabri"
DATASET_PATH: Final[Path] = Path("data/raw") / f"{DATASET_NAME}.csv"

TARGET_COL: Final[str] = "score"

BASE_FEATURE_COLS: Final[list[str]] = [
    "hours_studied",
    "practice_quizzes",
    "attendance_pct",
    "sleep_hours",
    "prior_score",
]

DERIVED_FEATURE_COL: Final[str] = "study_engagement"

FEATURE_COLS: Final[list[str]] = [
    *BASE_FEATURE_COLS,
    DERIVED_FEATURE_COL,
]

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42


# === Section 1d. PANDAS CONFIGURATION FOR DISPLAY ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. LOAD THE DATA ===


def load_data() -> pd.DataFrame:
    """Load the dataset from the data/raw folder."""
    LOG.info("Loading dataset: %s", DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Confirm that hours_scores_sabri.csv is inside data/raw."
        )

    df = pd.read_csv(DATASET_PATH)

    LOG.info("Loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    LOG.debug("\n%s", df.head())

    return df


# === Section 3. INSPECT DATA SHAPE AND STRUCTURE ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug("%s", list(df.columns))

    LOG.info("DataFrame info")
    df.info()

    LOG.info("Dataset shape: %d rows, %d columns", df.shape[0], df.shape[1])


# === Section 4. CHECK DATA QUALITY ===


def check_quality(df: pd.DataFrame) -> None:
    """Check required columns, missing values, and duplicate rows."""
    required_cols = [*BASE_FEATURE_COLS, TARGET_COL]
    missing_cols = [column for column in required_cols if column not in df.columns]

    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_cols)
        )

    LOG.info("Missing values by column")
    LOG.debug("\n%s", df[required_cols].isna().sum())

    duplicate_count = int(df.duplicated().sum())
    LOG.info("Duplicate row count: %d", duplicate_count)


# === Section 5. CREATE A CLEAN VIEW ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view and add a derived engagement feature."""
    LOG.info("Creating clean modeling view")

    # Work on a copy so the original DataFrame remains unchanged.
    df_work = df.copy()

    # Technical modification:
    # Combine study hours and attendance into one engagement metric.
    df_work[DERIVED_FEATURE_COL] = (
        df_work["hours_studied"] * df_work["attendance_pct"] / 100.0
    )

    LOG.info("Added derived feature: %s", DERIVED_FEATURE_COL)
    LOG.debug(
        "\n%s",
        df_work[["hours_studied", "attendance_pct", DERIVED_FEATURE_COL]].head(),
    )

    selected_cols = [*FEATURE_COLS, TARGET_COL]

    # Select only the columns needed for modeling, drop incomplete rows,
    # and create an independent copy.
    df_clean = df_work.loc[:, selected_cols].dropna().copy()

    if df_clean.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    LOG.info(
        "Clean view: %d rows, %d columns",
        df_clean.shape[0],
        df_clean.shape[1],
    )

    return df_clean


# === Section 6. TRAIN SUPERVISED MODEL ===


def train_model(df_clean: pd.DataFrame) -> LinearRegression:
    """Train and evaluate a supervised regression model."""
    LOG.info("Training LinearRegression model")

    if len(df_clean) < 4:
        raise ValueError(
            "At least 4 complete rows are required to create training and test sets."
        )

    x = df_clean.loc[:, FEATURE_COLS]
    y = df_clean.loc[:, TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    LOG.info("Mean absolute error: %.2f", mae)
    LOG.info("R-squared: %.2f", r2)

    return model


# === Section 7. PREDICT ONE NEW CASE ===


def predict_example(model: LinearRegression) -> None:
    """Use the trained model to predict one new student score."""
    LOG.info("Predicting one new case")

    new_case = pd.DataFrame(
        [
            {
                "hours_studied": 6.5,
                "practice_quizzes": 4,
                "attendance_pct": 92.0,
                "sleep_hours": 7.0,
                "prior_score": 72.0,
            }
        ]
    )

    new_case[DERIVED_FEATURE_COL] = (
        new_case["hours_studied"] * new_case["attendance_pct"] / 100.0
    )

    # Enforce the exact feature order used during model training.
    new_case = new_case.loc[:, FEATURE_COLS]

    predicted_score = float(model.predict(new_case)[0])

    LOG.info("New case:\n%s", new_case)
    LOG.info("Predicted score: %.1f", predicted_score)


# === Section 8. CREATE VISUALIZATIONS ===


def make_plots(df_clean: pd.DataFrame, model: LinearRegression) -> None:
    """Create and save charts for the supervised regression case."""
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Hours studied vs score
    LOG.info("Creating chart: hours studied vs score")

    fig1, ax1 = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df_clean,
        x="hours_studied",
        y=TARGET_COL,
        ax=ax1,
    )

    ax1.set_title("Hours Studied vs Student Score")
    ax1.set_xlabel("Hours Studied")
    ax1.set_ylabel("Score")

    fig1.tight_layout()
    fig1.savefig(
        str(output_dir / "Figure_1.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Figure 2: Model coefficients
    LOG.info("Creating chart: model coefficients")

    coefficient_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_,
        }
    ).sort_values("coefficient", ascending=True)

    LOG.info("Model coefficients:\n%s", coefficient_df)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.barh(
        coefficient_df["feature"],
        coefficient_df["coefficient"],
    )

    ax2.axvline(0, linewidth=1)

    ax2.set_title("Linear Regression Model Coefficients")
    ax2.set_xlabel("Coefficient")
    ax2.set_ylabel("Feature")

    fig2.tight_layout()
    fig2.savefig(
        str(output_dir / "Figure_2.png"),
        dpi=300,
        bbox_inches="tight",
    )

    LOG.info("Charts saved in: %s", output_dir)


# === Section 9. SUMMARY AND NEXT STEPS ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief workflow summary."""
    log_header(LOG, "SUMMARY")
    LOG.info("Dataset: %s", DATASET_PATH)
    LOG.info("Original rows: %d", df.shape[0])
    LOG.info("Clean rows: %d", df_clean.shape[0])
    LOG.info("Features: %s", FEATURE_COLS)
    LOG.info("Target: %s", TARGET_COL)


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Run the supervised machine-learning workflow."""
    log_header(LOG, "ML")
    LOG.info("START main()")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model = train_model(df_clean)

    LOG.info("Predict one case...........")
    predict_example(model)

    LOG.info("Create charts..............")
    make_plots(df_clean, model)

    LOG.info("Summarize workflow.........")
    summarize(df, df_clean)

    LOG.info("Close the chart windows to finish the program.")
    plt.show()

    log_header(LOG, "Executed successfully!")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
