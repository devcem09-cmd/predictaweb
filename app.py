import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import logging
from datetime import datetime
from io import StringIO
import time

# --- AYARLAR ---
app = Flask(__name__)
CORS(app) # Senin HTML dosyanın bağlanabilmesi için izinler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictaBrain")

# Nesine Headerları (Tokenlar değişirse güncellemen gerekir)
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
CSV_URL = "https://raw.githubusercontent.com/devcem09-cmd/predicta-api/main/data/merged_all.csv" # Senin veri setin

# --- GLOBAL VERİ HAFIZASI ---
# Her istekte CSV indirmemek için RAM'de tutacağız
historical_data = None
team_stats = {}

def load_historical_data():
    """CSV dosyasını sunucu hafızasına yükler ve analiz eder."""
    global historical_data, team_stats
    try:
        logger.info("📥 Geçmiş veriler GitHub'dan çekiliyor...")
        response = requests.get(CSV_URL)
        if response.status_code == 200:
            # CSV'yi Pandas ile oku (JavaScript'ten 100 kat hızlıdır)
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Basit bir istatistik tablosu oluştur
            # Not: Gerçek bir projede burası SQL veritabanı olmalı!
            stats = {}
            
            for index, row in df.iterrows():
                home = str(row.get('home_team', '')).strip()
                away = str(row.get('away_team', '')).strip()
                
                if home not in stats: stats[home] = {'matches': [], 'goals_for': 0, 'goals_against': 0}
                if away not in stats: stats[away] = {'matches': [], 'goals_for': 0, 'goals_against': 0}
                
                # Basit istatistikleri ekle
                try:
                    h_score = int(row.get('home_score', 0))
                    a_score = int(row.get('away_score', 0))
                    
                    stats[home]['matches'].append('W' if h_score > a_score else ('D' if h_score == a_score else 'L'))
                    stats[away]['matches'].append('W' if a_score > h_score else ('D' if a_score == h_score else 'L'))
                    
                    stats[home]['goals_for'] += h_score
                    stats[away]['goals_for'] += a_score
                except:
                    continue
            
            team_stats = stats
            logger.info(f"✅ Veri yüklendi. {len(team_stats)} takım analiz edildi.")
        else:
            logger.error("❌ CSV indirilemedi!")
    except Exception as e:
        logger.error(f"❌ Veri yükleme hatası: {e}")

# İlk açılışta veriyi yükle
load_historical_data()

def get_team_form(team_name):
    """Takımın son 5 maçlık formunu döndürür."""
    if team_name in team_stats:
        form = team_stats[team_name]['matches'][-5:] # Son 5 maç
        return "".join(form)
    return None

# --- NESİNE ENTEGRASYONU ---

def fetch_nesine_live():
    try:
        resp = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=10)
        data = resp.json()
        matches = []
        
        # Futbol maçlarını al (EA kodu)
        if "sg" in data and "EA" in data["sg"]:
            raw_matches = data["sg"]["EA"]
            
            for m in raw_matches:
                if m.get("GT") != 1: continue # Sadece bülten
                
                # Oranları ayrıştır
                odds = {}
                for market in m.get("MA", []):
                    # Maç Sonucu (1, X, 2)
                    if market.get("MTID") == 1:
                        for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["1"] = o.get("O")
                            if o.get("N") == 2: odds["X"] = o.get("O")
                            if o.get("N") == 3: odds["2"] = o.get("O")
                    
                    # Alt/Üst 2.5
                    if market.get("MTID") == 450:
                        odds["Over/Under +2.5"] = {}
                        for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["Over/Under +2.5"]["Over +2.5"] = o.get("O")
                            if o.get("N") == 2: odds["Over/Under +2.5"]["Under +2.5"] = o.get("O")

                    # KG Var/Yok
                    if market.get("MTID") == 38:
                         odds["Both Teams To Score"] = {}
                         for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["Both Teams To Score"]["Yes"] = o.get("O")
                            if o.get("N") == 2: odds["Both Teams To Score"]["No"] = o.get("O")

                # Sadece oranları tam olanları al
                if "1" in odds:
                    match_info = {
                        "match_id": str(m.get("C")),
                        "home_team": m.get("HN"),
                        "away_team": m.get("AN"),
                        "date": f"{m.get('D')}T{m.get('T')}:00",
                        "league_code": m.get("LC"),
                        "league_name": m.get("LN"),
                        "odds": odds,
                        # Backend tarafında hesaplanmış form bilgisini ekle
                        "stats": {
                            "home_form": get_team_form(m.get("HN")),
                            "away_form": get_team_form(m.get("AN"))
                        }
                    }
                    matches.append(match_info)
                    
        return matches
    except Exception as e:
        logger.error(f"Nesine API hatası: {e}")
        return []

# --- API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat()})

@app.route('/api/matches/upcoming', methods=['GET'])
def get_matches():
    """Frontend'in beklediği JSON formatında maçları döndürür."""
    matches = fetch_nesine_live()
    
    return jsonify({
        "success": True,
        "count": len(matches),
        "matches": matches,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Render.com genelde PORT environment variable'ını kullanır
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
