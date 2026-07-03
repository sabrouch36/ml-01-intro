"""Prepare one modeling record per World Cup player."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/raw/world_cup_2026_players.csv")
OUTPUT_PATH = Path("data/processed/world_cup_players_model.csv")


def main() -> None:
    """Aggregate match-level records into one row per player."""

    df = pd.read_csv(INPUT_PATH)

    player_df = df.groupby("player_id", as_index=False).agg(
        player_name=("player_name", "first"),
        age=("age", "first"),
        team=("team", "first"),
        position=("position", "first"),
        height_cm=("height_cm", "first"),
        weight_kg=("weight_kg", "first"),
        market_value_eur=("market_value_eur", "first"),
        matches_played=("match_id", "nunique"),
        total_minutes=("minutes_played", "sum"),
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_shots_on_target=("shots_on_target", "sum"),
        total_xg=("expected_goals_xg", "sum"),
        total_xa=("expected_assists_xa", "sum"),
        avg_pass_accuracy=("pass_accuracy", "mean"),
        total_defensive_actions=("defensive_actions", "sum"),
        total_distance_km=("distance_covered_km", "sum"),
        max_top_speed_kmh=("top_speed_kmh", "max"),
        avg_stamina_score=("stamina_score", "mean"),
        avg_player_rating=("player_rating", "mean"),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    player_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Raw rows: {len(df):,}")
    print(f"Unique players: {df['player_id'].nunique():,}")
    print(f"Processed shape: {player_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print(player_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
