"""
Automated Data & Retraining Pipeline.

Handles scheduled fetching of fresh match data, feature engineering,
and conditional model retraining based on performance thresholds.

Author: [Your Name]
Date: 2026-06-15
Version: 3.1.0
"""

import os
import sys
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import fetch_matches          # noqa: E402
from src.feature_engineering import enrich_match_dataset  # noqa: E402
from src.model import run_training_pipeline         # noqa: E402


def run_data_pipeline():
    """
    Execute the full data refresh and retraining cycle.
    
    This function is designed to be called by APScheduler. It fetches
    new matches, enriches them with features, and retrains the model.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Starting automated pipeline at {timestamp}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Fetch Fresh Data
        print("[PIPELINE] Step 1/3: Fetching latest match data from API...")
        df_raw = fetch_matches(competition_id='PL', limit=200)
        
        if df_raw is None or len(df_raw) == 0:
            print("[PIPELINE WARNING] No new data fetched. Skipping pipeline.")
            return
            
        print(f"[PIPELINE] Fetched {len(df_raw)} new matches.")
        
        # Step 2: Feature Engineering
        print("[PIPELINE] Step 2/3: Running feature engineering...")
        input_path = os.path.join(PROJECT_ROOT, "data", "matches_real.csv")
        output_path = os.path.join(PROJECT_ROOT, "data", "matches_enriched.csv")
        
        # Save raw data first so enricher can read it
        df_raw.to_csv(input_path, index=False)
        
        enriched_df = enrich_match_dataset(input_path, output_path)
        print(f"[PIPELINE] Enriched dataset ready: {len(enriched_df)} rows.")
        
        # Step 3: Retrain Model
        print("[PIPELINE] Step 3/3: Retraining model with fresh data...")
        run_training_pipeline()
        
        print(f"[PIPELINE SUCCESS] Pipeline completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"[PIPELINE ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow manual execution for testing
    run_data_pipeline()