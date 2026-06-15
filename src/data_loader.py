import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load variabel dari file .env
load_dotenv()

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
BASE_URL = "https://api.football-data.org/v4"

def fetch_matches(competition_id='PL', limit=100):
    """
    Mengambil data pertandingan dari Football-Data.org
    competition_id: 'PL' (Premier League), 'PD' (La Liga), dll.
    """
    if not API_KEY:
        print("❌ Error: API Key belum diset di file .env")
        return None

    headers = {'X-Auth-Token': API_KEY}
    url = f"{BASE_URL}/competitions/{competition_id}/matches?limit={limit}"
    
    print(f"📡 Mengambil data dari {url}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Cek error HTTP
        data = response.json()
        
        matches = []
        for match in data['matches']:
            # Hanya ambil pertandingan yang sudah selesai (FINISHED)
            if match['status'] == 'FINISHED':
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                home_score = match['score']['fullTime']['home']
                away_score = match['score']['fullTime']['away']
                
                # Tentukan hasil
                if home_score > away_score:
                    result = 'Home Win'
                elif away_score > home_score:
                    result = 'Away Win'
                else:
                    result = 'Draw'
                
                matches.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_score,
                    'away_goals': away_score,
                    'result': result,
                    'date': match['utcDate']
                })
                
        df = pd.DataFrame(matches)
        
        # Simpan ke CSV
        output_path = os.path.join('data', 'matches_real.csv')
        df.to_csv(output_path, index=False)
        print(f"✅ Berhasil mengambil {len(matches)} pertandingan!")
        print(f"💾 Data disimpan di {output_path}")
        return df
        
    except Exception as e:
        print(f"❌ Gagal mengambil data: {e}")
        return None

if __name__ == "__main__":
    # Ambil 100 pertandingan terakhir Premier League
    fetch_matches(competition_id='PL', limit=100)