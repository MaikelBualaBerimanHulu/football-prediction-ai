"""
Feature Engineering Module for Football Prediction System.

This module handles the transformation of raw match data into analytical features
suitable for machine learning models. It calculates rolling form scores and
head-to-head statistics based on historical match data.

Author: [Your Name]
Date: 2026-06-15
Version: 1.1.0 (Fixed index alignment issue)
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_rolling_form(
    df: pd.DataFrame,
    team_name: str,
    window_size: int = 5
) -> pd.Series:
    """
    Calculate rolling form points for a specific team over a defined window.

    Points are assigned as follows:
        - Win: 3 points
        - Draw: 1 point
        - Loss: 0 points

    Args:
        df: DataFrame containing match history sorted by date.
        team_name: The name of the team to calculate form for.
        window_size: Number of recent matches to consider for rolling sum.

    Returns:
        A pandas Series containing the rolling form score aligned with the
        original DataFrame index.
    """
    # Filter matches involving the target team
    team_mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    team_matches = df.loc[team_mask].copy()

    if team_matches.empty:
        return pd.Series(np.nan, index=df.index)

    # Determine match outcome from the perspective of the target team
    def get_match_points(row: pd.Series) -> int:
        is_home = row["home_team"] == team_name
        result = row["result"]

        if result == "Draw":
            return 1
        elif (is_home and result == "Home Win") or (not is_home and result == "Away Win"):
            return 3
        else:
            return 0

    team_matches["points"] = team_matches.apply(get_match_points, axis=1)

    # Calculate rolling sum
    rolling_scores = team_matches["points"].rolling(
        window=window_size, min_periods=1
    ).sum()

    # CRITICAL FIX: Return series with explicit index from team_matches
    # This ensures alignment when we later use .reindex() or merge
    return rolling_scores


def compute_head_to_head_stats(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    window_size: int = 5
) -> Tuple[int, int, int]:
    """
    Compute head-to-head statistics between two teams within a lookback window.

    Args:
        df: DataFrame containing all historical matches sorted by date descending.
        home_team: Name of the home team in the current fixture.
        away_team: Name of the away team in the current fixture.
        window_size: Maximum number of past encounters to analyze.

    Returns:
        A tuple of (home_wins, draws, away_wins).
    """
    h2h_mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )

    h2h_matches = df.loc[h2h_mask].sort_values("date", ascending=False).head(window_size)

    home_wins = 0
    draws = 0
    away_wins = 0

    for _, match in h2h_matches.iterrows():
        if match["result"] == "Draw":
            draws += 1
        elif (match["home_team"] == home_team and match["result"] == "Home Win") or \
             (match["away_team"] == home_team and match["result"] == "Away Win"):
            home_wins += 1
        else:
            away_wins += 1

    return home_wins, draws, away_wins


def enrich_match_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main pipeline function to enrich raw match data with engineered features.

    Args:
        input_path: Path to the raw CSV file containing match data.
        output_path: Path where the enriched CSV will be saved.

    Returns:
        The enriched DataFrame with additional feature columns.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure date column is datetime and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_columns = {"home_team", "away_team", "result", "date"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[INFO] Dataset loaded: {len(df)} matches, {len(df.columns)} columns")

    # --- FIX: Use reindex instead of map to handle index alignment ---
    print("[INFO] Computing rolling form scores...")
    unique_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    
    # Initialize form columns with NaN
    df["home_form_score"] = np.nan
    df["away_form_score"] = np.nan

    for team in unique_teams:
        form_series = calculate_rolling_form(df, team, window_size=5)
        
        # Find indices where this team plays home or away
        home_indices = df.index[df["home_team"] == team]
        away_indices = df.index[df["away_team"] == team]
        
        # Assign form scores using reindex to ensure correct alignment
        # We take the form value that corresponds to the same date/index
        aligned_form = form_series.reindex(df.index)
        
        df.loc[home_indices, "home_form_score"] = aligned_form.loc[home_indices]
        df.loc[away_indices, "away_form_score"] = aligned_form.loc[away_indices]

    # Fill any remaining NaN (e.g., first match of season) with 0
    df["home_form_score"] = df["home_form_score"].fillna(0)
    df["away_form_score"] = df["away_form_score"].fillna(0)

    # Calculate H2H statistics
    print("[INFO] Computing head-to-head statistics...")
    h2h_records: List[Dict[str, int]] = []

    for _, row in df.iterrows():
        hw, d, aw = compute_head_to_head_stats(
            df, row["home_team"], row["away_team"], window_size=5
        )
        h2h_records.append({
            "h2h_home_wins": hw,
            "h2h_draws": d,
            "h2h_away_wins": aw
        })

    h2h_df = pd.DataFrame(h2h_records, index=df.index)
    df = pd.concat([df, h2h_df], axis=1)

    # Save enriched dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Enriched dataset saved to {output_path}")
    print(f"[INFO] Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "matches_real.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "matches_enriched.csv")

    try:
        enriched_df = enrich_match_dataset(INPUT_FILE, OUTPUT_FILE)
        print("\n[SUCCESS] Feature engineering completed successfully.")
        print(enriched_df[
            ["home_team", "away_team", "home_form_score", "away_form_score",
             "h2h_home_wins", "h2h_draws", "h2h_away_wins"]
        ].head(10).to_string(index=False))
    except Exception as e:
        print(f"\n[ERROR] Feature engineering failed: {e}")
        raise