import pandas as pd
import joblib
import os

def predict_match(home_team, away_team, home_goals_avg=1.5, away_goals_avg=1.2):
    """
    Memprediksi hasil pertandingan menggunakan MODEL REAL.
    
    Parameter:
    - home_team: Nama tim tuan rumah (harus sama persis dengan di API)
    - away_team: Nama tim tamu
    - home_goals_avg: Rata-rata gol tim tuan rumah (opsional, default 1.5)
    - away_goals_avg: Rata-rata gol tim tamu (opsional, default 1.2)
    """
    print(f"\n⚽ Menganalisis Pertandingan: {home_team} vs {away_team}")
    
    # 1. Load Model & Encoder REAL
    model_path = os.path.join('models', 'football_model_real.pkl')
    encoder_path = os.path.join('models', 'encoders_real.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("❌ Error: Model real belum di-training.")
        print("   Jalankan 'python src/model.py' dulu!")
        return None

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    le_home = encoders['home']
    le_away = encoders['away']

    # 2. Encode nama tim
    try:
        home_code = le_home.transform([home_team])[0]
        away_code = le_away.transform([away_team])[0]
    except ValueError:
        print(f"⚠️ Tim '{home_team}' atau '{away_team}' tidak ada di data training.")
        print("   Pastikan nama tim sama persis dengan di Premier League.")
        return None

    # 3. Siapkan input (urutan harus sama saat training!)
    # Fitur: ['home_team_encoded', 'away_team_encoded', 'home_goals', 'away_goals']
    input_data = pd.DataFrame({
        'home_team_encoded': [home_code],
        'away_team_encoded': [away_code],
        'home_goals': [home_goals_avg],
        'away_goals': [away_goals_avg]
    })

    # 4. Prediksi
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    # 5. Tampilkan Hasil
    print("-" * 40)
    print(f"🏆 PREDIKSI HASIL: {prediction}")
    print("-" * 40)
    print(" Probabilitas:")
    for cls, prob in sorted(zip(classes, probabilities), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {cls:10s}: {prob*100:5.1f}% {bar}")
    print("-" * 40)
    
    return prediction

if __name__ == "__main__":
    # Contoh prediksi dengan tim Premier League
    # ⚠️ Gunakan nama tim PERSIS seperti di API (case-sensitive)
    predict_match(
        home_team="Everton FC",  # Contoh tim tuan rumah
        away_team="Liverpool FC",
        home_goals_avg=1.8,  # Sesuaikan dengan statistik tim
        away_goals_avg=2.1
    )