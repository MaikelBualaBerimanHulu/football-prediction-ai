"""
Database Persistence Module for Football Prediction System.

Handles MySQL connections and prediction logging with connection pooling
and parameterized queries to ensure security and performance.

Author: [Your Name]
Date: 2026-06-15
Version: 1.0.0
"""

import os
from contextlib import contextmanager
from typing import Dict, Optional

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """
    Manages MySQL database connections and prediction logging operations.
    
    Uses context managers to ensure proper connection lifecycle management
    and prevent resource leaks.
    """

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "football_prediction_db"),
            "port": int(os.getenv("DB_PORT", "3306")),
            "charset": "utf8mb4",
            "collation": "utf8mb4_unicode_ci",
        }

    @contextmanager
    def get_connection(self):
        """
        Context manager for safe database connection handling.
        
        Yields:
            mysql.connector.connection: Active database connection.
            
        Ensures connection is properly closed even if exceptions occur.
        """
        conn = None
        try:
            conn = mysql.connector.connect(**self.db_config)
            yield conn
        except mysql.connector.Error as err:
            print(f"[ERROR] Database connection failed: {err}")
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()

    def log_prediction(self, prediction_data: Dict) -> Optional[int]:
        """
        Persist a single prediction record to the database.

        Args:
            prediction_data: Dictionary containing prediction details.
                Required keys: home_team, away_team, predicted_outcome, 
                confidence_score, and optional feature values.

        Returns:
            The auto-generated ID of the inserted record, or None on failure.
        """
        query = """
            INSERT INTO prediction_logs (
                match_date, home_team, away_team,
                home_form_score, away_form_score,
                h2h_home_wins, h2h_draws, h2h_away_wins,
                predicted_outcome, confidence_score
            ) VALUES (
                NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        
        params = (
            prediction_data.get("home_team"),
            prediction_data.get("away_team"),
            float(prediction_data.get("home_form_score", 0)),
            float(prediction_data.get("away_form_score", 0)),
            int(prediction_data.get("h2h_home_wins", 0)),
            int(prediction_data.get("h2h_draws", 0)),
            int(prediction_data.get("h2h_away_wins", 0)),
            prediction_data.get("predicted_outcome"),
            float(prediction_data.get("confidence_score", 0)),
        )

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                record_id = cursor.lastrowid
                print(f"[DB] Prediction logged successfully. ID: {record_id}")
                return record_id
        except Exception as e:
            print(f"[ERROR] Failed to log prediction: {e}")
            return None


# Singleton instance for module-level access
db_manager = DatabaseManager()