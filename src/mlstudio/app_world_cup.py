"""Predict football player market values from tournament performance data.

This application uses one aggregated record per player and trains a supervised
Linear Regression model to predict market value in euros.

Author: Sabri Hamdaoui
Date: 2026-07
"""

import logging
from pathlib import Path
from typing import Final, cast

from datafun_toolkit.logger import get_logger, log_header
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG: logging.Logger = get_logger("WORLD_CUP_ML", level="DEBUG")

INPUT_PATH: Final[Path] = Path("data/processed/world_cup_players_model.csv")

OUTPUT_PATH: Final[Path] = Path("data/output/world_cup_player_value_predictions.csv")

CHART_PATH: Final[Path] = Path("docs/images/world_cup_actual_vs_predicted.png")

TARGET_COL: Final[str] = "market_value_eur"

NUMERIC_FEATURES: Final[list[str]] = [
    "age",
    "height_cm",
    "weight_kg",
    "avg_pass_accuracy",
    "max_top_speed_kmh",
    "avg_stamina_score",
    "goals_per_90",
    "assists_per_90",
    "shots_on_target_per_90",
    "xg_per_90",
    "xa_per_90",
    "defensive_actions_per_90",
    "distance_km_per_90",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "position",
]

TEST_SIZE: Final[float] = 0.20
RANDOM_STATE: Final[int] = 42


def load_data() -> pd.DataFrame:
    """Load the aggregated player dataset."""
    LOG.info("Loading dataset: %s", INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}. Run prepare_world_cup_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    LOG.info("Loaded %s players and %s columns", len(df), len(df.columns))
    return df


def add_per_90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized performance features per 90 minutes."""
    LOG.info("Creating performance features per 90 minutes")

    df_work = df.copy()
    minutes = df_work["total_minutes"].replace(0, np.nan)

    df_work["goals_per_90"] = df_work["total_goals"] / minutes * 90
    df_work["assists_per_90"] = df_work["total_assists"] / minutes * 90
    df_work["shots_on_target_per_90"] = df_work["total_shots_on_target"] / minutes * 90
    df_work["xg_per_90"] = df_work["total_xg"] / minutes * 90
    df_work["xa_per_90"] = df_work["total_xa"] / minutes * 90
    df_work["defensive_actions_per_90"] = (
        df_work["total_defensive_actions"] / minutes * 90
    )
    df_work["distance_km_per_90"] = df_work["total_distance_km"] / minutes * 90

    return df_work


def create_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Select the required columns and remove unusable rows."""
    required_columns = (
        ["player_id", "player_name"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COL]
    )

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    selected_data = pd.DataFrame(df.loc[:, required_columns])
    df_clean: pd.DataFrame = selected_data.dropna().copy()
    target_values = pd.Series(
        df_clean.loc[:, TARGET_COL],
        dtype="float64",
        name=TARGET_COL,
    )

    LOG.info("Clean modeling rows: %s", len(df_clean))
    LOG.info(
        "Market value median: €%s",
        f"{float(target_values.median()):,.0f}",
    )

    return df_clean


def build_model() -> TransformedTargetRegressor:
    """Build the preprocessing and Linear Regression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    return TransformedTargetRegressor(
        regressor=regression_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def save_actual_vs_predicted_plot(
    actual_values: pd.Series,
    predicted_values: np.ndarray,
) -> None:
    """Create and save a chart of actual versus predicted market values."""
    LOG.info("Creating actual-versus-predicted chart")

    actual_array = np.asarray(actual_values, dtype=float)
    predicted_array = np.asarray(predicted_values, dtype=float)

    minimum_value = float(min(actual_array.min(), predicted_array.min()))
    maximum_value = float(max(actual_array.max(), predicted_array.max()))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        actual_array,
        predicted_array,
        alpha=0.7,
    )

    ax.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linewidth=2,
        linestyle="--",
        label="Perfect prediction",
    )

    ax.set_title("Actual vs Predicted Player Market Value")
    ax.set_xlabel("Actual Market Value (EUR)")
    ax.set_ylabel("Predicted Market Value (EUR)")
    ax.ticklabel_format(style="plain", axis="both")
    ax.legend()

    fig.tight_layout()

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(CHART_PATH),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    LOG.info("Chart saved to: %s", CHART_PATH)


def train_and_evaluate(
    df_clean: pd.DataFrame,
) -> TransformedTargetRegressor:
    """Train the model, compare it with a baseline, and save predictions."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    x = pd.DataFrame(df_clean.loc[:, feature_columns]).copy()
    y = pd.Series(
        df_clean.loc[:, TARGET_COL],
        dtype="float64",
        name=TARGET_COL,
    )
    position_values = pd.Series(
        df_clean.loc[:, "position"],
        dtype="string",
        name="position",
    )

    split_data = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=position_values,
    )
    x_train = cast(pd.DataFrame, split_data[0])
    x_test = cast(pd.DataFrame, split_data[1])
    y_train = cast(pd.Series, split_data[2])
    y_test = cast(pd.Series, split_data[3])

    model = build_model()

    LOG.info("Training model with %s players", len(x_train))
    model.fit(x_train, y_train)

    predictions = np.maximum(model.predict(x_test), 0)

    # Baseline: predict the training median for every test player.
    baseline_value = float(y_train.median())
    baseline_predictions = np.full(
        shape=len(y_test),
        fill_value=baseline_value,
    )

    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    if baseline_mae > 0:
        improvement_pct = (baseline_mae - mae) / baseline_mae * 100
    else:
        improvement_pct = 0.0

    LOG.info("Test players: %s", len(x_test))
    LOG.info(
        "Baseline Median Prediction: €%s",
        f"{baseline_value:,.0f}",
    )
    LOG.info(
        "Baseline Mean Absolute Error: €%s",
        f"{baseline_mae:,.0f}",
    )
    LOG.info("Mean Absolute Error: €%s", f"{mae:,.0f}")
    LOG.info(
        "Root Mean Squared Error: €%s",
        f"{rmse:,.0f}",
    )
    LOG.info("R-squared: %.3f", r2)
    LOG.info(
        "MAE improvement over baseline: %.2f%%",
        improvement_pct,
    )

    save_actual_vs_predicted_plot(y_test, predictions)

    result_columns = ["player_id", "player_name", "position"]
    results = pd.DataFrame(df_clean.loc[x_test.index, result_columns]).copy()

    results["actual_market_value_eur"] = y_test
    results["predicted_market_value_eur"] = predictions
    results["absolute_error_eur"] = np.abs(
        results["actual_market_value_eur"] - results["predicted_market_value_eur"]
    )

    results = results.sort_values(
        "absolute_error_eur",
        ascending=False,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    LOG.info("Predictions saved to: %s", OUTPUT_PATH)

    return model


def main() -> None:
    """Run the World Cup player-value regression workflow."""
    log_header(LOG, "WORLD CUP PLAYER VALUE MODEL")

    df = load_data()
    df_features = add_per_90_features(df)
    df_clean = create_clean_view(df_features)

    train_and_evaluate(df_clean)

    LOG.info("World Cup regression workflow completed successfully.")


if __name__ == "__main__":
    main()
