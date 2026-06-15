import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "football_prediction_db"),
    "port": int(os.getenv("DB_PORT", "3306")),
}

print("[TEST] Checking connection...")
try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES LIKE 'prediction_logs'")
    
    if cursor.fetchone():
        print("[SUCCESS] Table 'prediction_logs' exists.")
        cursor.execute("SELECT COUNT(*) FROM prediction_logs")
        print(f"[INFO] Records found: {cursor.fetchone()[0]}")
    else:
        print("[ERROR] Table 'prediction_logs' NOT found. Run SQL script first.")
        
    cursor.close()
    conn.close()
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")