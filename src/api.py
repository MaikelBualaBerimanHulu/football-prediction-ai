"""
FastAPI REST Interface for Football Prediction System.

Exposes match prediction functionality via HTTP endpoints with automatic
validation, documentation, database persistence, and automated background
data pipeline scheduling.

Author: [Your Name]
Date: 2026-06-15
Version: 3.1.0 (Integrated APScheduler)
"""

import os
import sys
from typing import Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is in path for src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import create_predictor  # noqa: E402
from src.database import db_manager       # noqa: E402
from src.pipeline import run_data_pipeline # noqa: E402

app = FastAPI(
    title="Football Prediction AI API",
    description="REST API for ML-based football match outcome prediction with automated data pipeline",
    version="3.1.0",
)

predictor = None
scheduler = BackgroundScheduler(daemon=True)


@app.on_event("startup")
async def startup_events():
    """Load ML model and initialize background scheduler on server startup."""
    global predictor
    
    # Load model
    try:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        predictor = create_predictor(models_dir)
        print("[API] Model loaded successfully at startup.")
    except Exception as e:
        print(f"[API CRITICAL] Failed to load model: {e}")
        raise

    # Initialize scheduler
    try:
        scheduler.add_job(
            func=run_data_pipeline,
            trigger="cron",
            hour=2,
            minute=0,
            id="daily_data_pipeline",
            replace_existing=True,
            misfire_grace_time=3600,  # Allow job to run if server was down during scheduled time
        )
        scheduler.start()
        print("[API] Background scheduler started. Next pipeline: Daily at 02:00")
    except Exception as e:
        print(f"[API WARNING] Scheduler initialization failed: {e}")
        # Do not crash server if scheduler fails; predictions still work


@app.on_event("shutdown")
async def shutdown_events():
    """Gracefully shut down background scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[API] Background scheduler stopped.")


class PredictionRequest(BaseModel):
    """Schema for incoming prediction requests with strict validation."""
    home_team: str = Field(..., min_length=1, max_length=100, example="Liverpool FC")
    away_team: str = Field(..., min_length=1, max_length=100, example="Manchester United FC")
    home_goals_avg: float = Field(default=1.5, ge=0, le=10)
    away_goals_avg: float = Field(default=1.2, ge=0, le=10)
    home_form_score: float = Field(default=0.0, ge=0, le=15)
    away_form_score: float = Field(default=0.0, ge=0, le=15)
    h2h_home_wins: int = Field(default=0, ge=0, le=5)
    h2h_draws: int = Field(default=0, ge=0, le=5)
    h2h_away_wins: int = Field(default=0, ge=0, le=5)


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    probabilities: Dict[str, float]
    db_record_id: Optional[int] = None


@app.post("/api/v1/predict", response_model=PredictionResponse, status_code=201)
async def predict_match(request: PredictionRequest):
    """
    Predict match outcome and persist result to database.

    Validates input via Pydantic schema, runs inference through trained
    Random Forest model, and logs the prediction for audit/analytics.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        result = predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            home_goals_avg=request.home_goals_avg,
            away_goals_avg=request.away_goals_avg,
            home_form_score=request.home_form_score,
            away_form_score=request.away_form_score,
            h2h_home_wins=request.h2h_home_wins,
            h2h_draws=request.h2h_draws,
            h2h_away_wins=request.h2h_away_wins,
        )

        max_outcome = max(result, key=result.get)
        max_confidence = result[max_outcome]

        log_data = {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "home_form_score": request.home_form_score,
            "away_form_score": request.away_form_score,
            "h2h_home_wins": request.h2h_home_wins,
            "h2h_draws": request.h2h_draws,
            "h2h_away_wins": request.h2h_away_wins,
            "predicted_outcome": max_outcome,
            "confidence_score": max_confidence,
        }
        record_id = db_manager.log_prediction(log_data)

        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            predicted_outcome=max_outcome,
            confidence=max_confidence,
            probabilities=result,
            db_record_id=record_id,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "scheduler_running": scheduler.running,
    }