import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model():
    print("🚀 Memulai proses training model dengan DATA REAL...")
    
    # 1. Load Data Real
    data_path = os.path.join('data', 'matches_real.csv')
    if not os.path.exists(data_path):
        print("❌ Error: File matches_real.csv tidak ditemukan.")
        print("   Jalankan src/data_loader.py dulu untuk mengambil data dari API.")
        return

    df = pd.read_csv(data_path)
    print(f"📊 Data loaded: {df.shape[0]} pertandingan real")
    
    if len(df) < 20:
        print("️ Peringatan: Data terlalu sedikit untuk training yang baik.")
        print("   Coba ambil lebih banyak data di data_loader.py (ubah parameter limit)")

    # 2. Preprocessing
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    
    df['home_team_encoded'] = le_home.fit_transform(df['home_team'])
    df['away_team_encoded'] = le_away.fit_transform(df['away_team'])
    
    y = df['result']
    
    # Fitur untuk data real: hanya encoded team + goals
    # Nanti bisa ditambah fitur lain seperti H2H, home/away stats, dll
    features = ['home_team_encoded', 'away_team_encoded', 'home_goals', 'away_goals']
    X = df[features]

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Training Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("⚙️ Sedang melatih model dengan data real...")
    model.fit(X_train, y_train)
    
    # 5. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training Selesai!")
    print(f"🎯 Akurasi Model: {accuracy * 100:.2f}%")
    print("\n📋 Laporan Detail:")
    print(classification_report(y_test, y_pred))

    # 6. Simpan Model & Encoder
    model_path = os.path.join('models', 'football_model_real.pkl')
    encoder_path = os.path.join('models', 'encoders_real.pkl')
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump({'home': le_home, 'away': le_away}, encoder_path)
    
    print(f"\n💾 Model disimpan di {model_path}")
    print(f"💾 Encoder disimpan di {encoder_path}")

if __name__ == "__main__":
    train_model()