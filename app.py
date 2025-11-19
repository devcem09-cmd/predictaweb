import requests
import json
import logging
import time
import os
from datetime import datetime
from flask import Flask, jsonify, render_template
from flask_cors import CORS
# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Flask uygulamasını başlat (template_folder varsayılan olarak 'templates'tir)
app = Flask(__name__)
CORS(app)
# Nesine API Ayarları
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}
# Önbellek (Cache) Mekanizması
cache = {
    "data": None,
    "timestamp": 0
}
CACHE_DURATION = 60  # 60 saniye cache
def get_nesine_data():
    """Nesine.com'dan bülten verilerini çeker."""
    global cache
    current_time = time.time()
    # Cache geçerliyse onu döndür
    if cache["data"] and (current_time - cache["timestamp"] < CACHE_DURATION):
        logger.info("📦 Cache'den veri kullanılıyor.")
        return cache["data"]
    try:
        logger.info("🌐 Nesine API'ye istek gönderiliyor...")
        response = requests.get(NESINE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Cache güncelle
        cache["data"] = data
        cache["timestamp"] = current_time
        logger.info("✅ Veri başarıyla çekildi ve cachelendi.")
        return data
    except Exception as e:
        logger.error(f"❌ Veri çekme hatası: {e}")
        return None
def parse_matches(data):
    """Ham veriyi işleyip bizim formatımıza çevirir."""
    matches = []
    
    if not data or "sg" not in data:
        return matches
    football_matches = data.get("sg", {}).get("EA", [])
    
    for m in football_matches:
        if m.get("GT") != 1:  # Sadece Futbol (GT=1)
            continue
        match_id = str(m.get("C"))
        home_team = m.get("HN")
        away_team = m.get("AN")
        date = m.get("D")
        time_str = m.get("T")
        league_name = m.get("LN")
        try:
            day, month, year = date.split('.')
            iso_date = f"{year}-{month}-{day}T{time_str}:00"
        except:
            iso_date = datetime.now().isoformat()
        odds = {
            "1": None, "X": None, "2": None,
            "Over 2.5": None, "Under 2.5": None,
            "Yes": None, "No": None
        }
        for market in m.get("MA", []):
            mtid = market.get("MTID")
            outcomes = market.get("OCA", [])
            if mtid == 1 and len(outcomes) >= 3:
                odds["1"] = outcomes[0].get("O")
                odds["X"] = outcomes[1].get("O")
                odds["2"] = outcomes[2].get("O")
            if "Alt" in str(outcomes[0].get("N")) and "Üst" in str(outcomes[1].get("N")):
                 if mtid == 5 or mtid == 450:
                     odds["Under 2.5"] = outcomes[0].get("O")
                     odds["Over 2.5"] = outcomes[1].get("O")
            if "Var" in str(outcomes[0].get("N")) and "Yok" in str(outcomes[1].get("N")):
                odds["Yes"] = outcomes[0].get("O")
                odds["No"] = outcomes[1].get("O")
        matches.append({
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "league_name": league_name,
            "date": iso_date,
            "odds": odds,
            "raw_odds": m.get("MA")
        })
    return matches
# --- ROUTES ---
@app.route('/')
def index():
    """Ana sayfayı (HTML) render eder."""
    return render_template('index.html')
@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Maç verilerini JSON olarak döndürür."""
    data = get_nesine_data()
    if not data:
        return jsonify({"success": False, "message": "Veri çekilemedi"}), 500
    
    matches = parse_matches(data)
    return jsonify({
        "success": True,
        "count": len(matches),
        "matches": matches
    })
if __name__ == '__main__':
    # Render'ın PORT environment variable'ını kullanması için
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
