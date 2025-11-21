import os
import logging
import time
from functools import lru_cache # <--- YENİ: Performans için
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import requests
from scipy.stats import poisson
from rapidfuzz import process, fuzz

# --- YAPILANDIRMA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
# APK ve Web'den gelen isteklere izin ver
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

class MatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.team_list = [] # Fuzzy search için liste
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı başlatılıyor...")
        
        if not os.path.exists(CSV_PATH):
            logger.warning(f"⚠️ UYARI: CSV Bulunamadı ({CSV_PATH}). Tahminler çalışmayacak.")
            return

        try:
            # Sadece gerekli sütunları oku
            df = pd.read_csv(CSV_PATH, usecols=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], encoding='utf-8', on_bad_lines='skip')
            
            # İsim standardizasyonu
            df.columns = ['home_team', 'away_team', 'home_score', 'away_score']
            
            # Veri tiplerini küçült (RAM Optimizasyonu)
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            
            # İstatistikleri hesapla
            self._calculate_stats(df)
            
            # Listeyi fuzzy search için hazırla
            self.team_list = list(self.team_stats.keys())
            
            # DataFrame'i bellekten sil
            del df
            logger.info(f"✅ Veritabanı hazır. {len(self.team_stats)} takım yüklendi.")
            
        except Exception as e:
            logger.error(f"❌ DB Kritik Hata: {e}")

    def _calculate_stats(self, df):
        if df.empty: return
        
        self.avg_home_goals = df['home_score'].mean() or 1.5
        self.avg_away_goals = df['away_score'].mean() or 1.2
        
        # Takımları grupla ve ortalamaları al (Pandas Vectorization - Çok daha hızlı)
        home_stats = df.groupby('home_team')['home_score'].agg(['mean', 'count'])
        home_conceded = df.groupby('home_team')['away_score'].mean()
        
        away_stats = df.groupby('away_team')['away_score'].agg(['mean', 'count'])
        away_conceded = df.groupby('away_team')['home_score'].mean()
        
        # Tüm takımları birleştir
        all_teams = set(home_stats.index) | set(away_stats.index)
        
        for team in all_teams:
            # En az 3 maç verisi gerekli
            h_count = home_stats.loc[team, 'count'] if team in home_stats.index else 0
            a_count = away_stats.loc[team, 'count'] if team in away_stats.index else 0
            
            if h_count < 3 or a_count < 3: continue

            att_h = home_stats.loc[team, 'mean'] / self.avg_home_goals
            def_h = home_conceded.loc[team] / self.avg_away_goals
            
            att_a = away_stats.loc[team, 'mean'] / self.avg_away_goals
            def_a = away_conceded.loc[team] / self.avg_home_goals
            
            self.team_stats[team] = {
                'att_h': att_h, 'def_h': def_h,
                'att_a': att_a, 'def_a': def_a
            }

    @lru_cache(maxsize=2048) # <--- CACHE: Aynı ismi tekrar aramaz
    def find_team_cached(self, name):
        if not name or not self.team_stats: return None
        
        # Basit temizlik
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        
        # Rapidfuzz ile en iyi eşleşme
        match = process.extractOne(
            clean_name, 
            self.team_list, 
            scorer=fuzz.token_set_ratio, 
            score_cutoff=65
        )
        
        return match[0] if match else None

    def predict(self, home, away):
        # Cache'lenmiş fonksiyonu çağır
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        if not home_db or not away_db: return None
            
        hs = self.team_stats[home_db]
        as_ = self.team_stats[away_db]
        
        # xG Hesaplama
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        # Poisson Olasılıkları
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        prob_1, prob_x, prob_2 = 0, 0, 0
        prob_over = 0
        prob_btts_yes = 0
        
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                if h > a: prob_1 += p
                elif h == a: prob_x += p
                else: prob_2 += p
                if (h + a) > 2.5: prob_over += p
                if h > 0 and a > 0: prob_btts_yes += p # Daha hassas BTTS hesabı

        return {
            "stats": {"home_xg": round(h_xg, 2), "away_xg": round(a_xg, 2)},
            "probs": {
                "1": round(prob_1 * 100, 1),
                "X": round(prob_x * 100, 1),
                "2": round(prob_2 * 100, 1),
                "over": round(prob_over * 100, 1),
                "under": round((1 - prob_over) * 100, 1),
                "btts_yes": round(prob_btts_yes * 100, 1),
                "btts_no": round((1 - prob_btts_yes) * 100, 1)
            }
        }

predictor = MatchPredictor()

@app.route('/')
def index():
    return "Predicta PRO API Online. Use /api/matches/live"

@app.route('/health')
def health():
    """Uptime Robot gibi servisler buraya ping atıp sunucuyu uyanık tutar."""
    return jsonify({"status": "ok", "timestamp": time.time()})

@app.route('/api/matches/live')
def live():
    try:
        r = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=10)
        d = r.json()
        matches = []
        
        if "sg" in d and "EA" in d["sg"]:
            for m in d["sg"]["EA"]:
                # Sadece Futbol (GT=1) ve henüz başlamamış veya canlı olmayan maçlar
                if m.get("GT") != 1: continue 
                
                # Oranları Çek
                odds = {}
                markets = m.get("MA", [])
                
                # Hızlı erişim için map oluştur
                # Bu kısım kodun okunabilirliğini artırır
                for market in markets:
                    mtid = market.get("MTID")
                    oca = market.get("OCA", [])
                    
                    if mtid == 1: # MS
                        for o in oca:
                            if o["N"] == 1: odds["1"] = o["O"]
                            elif o["N"] == 2: odds["X"] = o["O"]
                            elif o["N"] == 3: odds["2"] = o["O"]
                    
                    elif mtid == 450: # 2.5 Alt/Üst (ID değişebilir, kontrol et)
                         if "Over/Under +2.5" not in odds: odds["Over/Under +2.5"] = {}
                         for o in oca:
                             if o["N"] == 1: odds["Over/Under +2.5"]["Over +2.5"] = o["O"]
                             if o["N"] == 2: odds["Over/Under +2.5"]["Under +2.5"] = o["O"]

                # Eğer MS oranları yoksa (bazen sadece özel etkinlikler olur) maçı geç
                if "1" not in odds: continue

                # Maç Verisi Oluştur
                match_data = {
                    "home": m.get("HN"),
                    "away": m.get("AN"),
                    "date": f"{m.get('D')} {m.get('T')}",
                    "league": m.get("LN") or "Lig",
                    "odds": odds,
                    "prediction": predictor.predict(m.get("HN"), m.get("AN"))
                }
                matches.append(match_data)
        
        return jsonify({"success": True, "count": len(matches), "matches": matches})

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"success": False, "error": "Veri alınamadı"}), 500

if __name__ == '__main__':
    # Render/Heroku için PORT ayarı
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
