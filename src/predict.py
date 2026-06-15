"""
Prediction Inference Module for Football Prediction System.
Handles match outcome predictions using trained Random Forest model v2.
"""

import os
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# CRITICAL FIX: Ensure project root is in Python path for relative imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now safe to import from src
from src.database import db_manager  # noqa: E402

EXPECTED_FEATURES: List[str] = [
    "home_team_encoded", "away_team_encoded", "home_goals", "away_goals",
    "home_form_score", "away_form_score", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
]


class MatchPredictor:
    """Wrapper class for loading model artifacts and performing predictions."""

    def __init__(self, models_dir: str) -> None:
        self.model_path = os.path.join(models_dir, "football_model_v2.pkl")
        self.encoder_path = os.path.join(models_dir, "encoders_v2.pkl")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {self.encoder_path}")

        print(f"[INFO] Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.encoders: Dict[str, object] = joblib.load(self.encoder_path)
        print("[INFO] Model and encoders loaded successfully.")

    def predict(
        self,
        home_team: str, away_team: str,
        home_goals_avg: float = 1.5, away_goals_avg: float = 1.2,
        home_form_score: float = 0.0, away_form_score: float = 0.0,
        h2h_home_wins: int = 0, h2h_draws: int = 0, h2h_away_wins: int = 0,
    ) -> Dict[str, float]:
        """Generate prediction for a single match fixture."""
        le_home = self.encoders["home"]
        le_away = self.encoders["away"]

        if home_team not in le_home.classes_:
            raise ValueError(f"Home team '{home_team}' not found in training data.")
        if away_team not in le_away.classes_:
            raise ValueError(f"Away team '{away_team}' not found in training data.")

        home_code = int(le_home.transform([home_team])[0])
        away_code = int(le_away.transform([away_team])[0])

        feature_vector = np.array([[
            home_code, away_code, home_goals_avg, away_goals_avg,
            home_form_score, away_form_score,
            h2h_home_wins, h2h_draws, h2h_away_wins,
        ]])

        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        classes = self.model.classes_

        result: Dict[str, float] = {}
        for cls, prob in zip(classes, probabilities):
            result[cls] = round(float(prob), 4)

        print(f"\n[PREDICTION] {home_team} vs {away_team}")
        print(f"[RESULT]     Predicted: {prediction}")
        for cls, prob in sorted(result.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(prob * 30)
            print(f"             {cls:>10s}: {prob*100:5.1f}% {'#' * bar_len}")

        return result


def create_predictor(models_dir: Optional[str] = None) -> MatchPredictor:
    """Factory function to create a MatchPredictor instance."""
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    return MatchPredictor(models_dir)


if __name__ == "__main__":
    try:
        predictor = create_predictor()
        
        HOME_TEAM = "Liverpool FC"
        AWAY_TEAM = "Manchester United FC"

        print(f"\n[INFO] Predicting: {HOME_TEAM} vs {AWAY_TEAM}")
        
        result = predictor.predict(
            home_team=HOME_TEAM, away_team=AWAY_TEAM,
            home_goals_avg=2.1, away_goals_avg=1.3,
            home_form_score=12.0, away_form_score=7.0,
            h2h_home_wins=3, h2h_draws=1, h2h_away_wins=1,
        )

        max_outcome = max(result, key=result.get)
        max_confidence = result[max_outcome]
        
        log_data = {
            "home_team": HOME_TEAM, "away_team": AWAY_TEAM,
            "home_form_score": 12.0, "away_form_score": 7.0,
            "h2h_home_wins": 3, "h2h_draws": 1, "h2h_away_wins": 1,
            "predicted_outcome": max_outcome, "confidence_score": max_confidence,
        }
        
        print("\n[INFO] Logging prediction to database...")
        record_id = db_manager.log_prediction(log_data)
        
        if record_id:
            print(f"[SUCCESS] Prediction persisted with ID: {record_id}")
        else:
            print("[WARNING] Database logging failed. Check [DB ERROR] logs above.")

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()