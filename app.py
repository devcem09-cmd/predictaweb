import os
import logging
import time
from functools import lru_cache
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
        self.team_list = []
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı başlatılıyor...")
        
        if not os.path.exists(CSV_PATH):
            logger.warning(f"⚠️ UYARI: CSV Bulunamadı ({CSV_PATH}). Tahminler çalışmayacak.")
            return

        try:
            df = pd.read_csv(CSV_PATH, usecols=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], encoding='utf-8', on_bad_lines='skip')
            df.columns = ['home_team', 'away_team', 'home_score', 'away_score']
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            
            self._calculate_stats(df)
            self.team_list = list(self.team_stats.keys())
            del df
            logger.info(f"✅ Veritabanı hazır. {len(self.team_stats)} takım yüklendi.")
            
        except Exception as e:
            logger.error(f"❌ DB Kritik Hata: {e}")

    def _calculate_stats(self, df):
        if df.empty: return
        
        self.avg_home_goals = df['home_score'].mean() or 1.5
        self.avg_away_goals = df['away_score'].mean() or 1.2
        
        home_stats = df.groupby('home_team')['home_score'].agg(['mean', 'count'])
        home_conceded = df.groupby('home_team')['away_score'].mean()
        away_stats = df.groupby('away_team')['away_score'].agg(['mean', 'count'])
        away_conceded = df.groupby('away_team')['home_score'].mean()
        all_teams = set(home_stats.index) | set(away_stats.index)
        
        for team in all_teams:
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

    @lru_cache(maxsize=2048)
    def find_team_cached(self, name):
        if not name or not self.team_stats: return None
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        match = process.extractOne(clean_name, self.team_list, scorer=fuzz.token_set_ratio, score_cutoff=65)
        return match[0] if match else None

    def predict_all(self, home, away):
        """Tüm bahis türleri için tahmin hesapla"""
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        if not home_db or not away_db: 
            return None
            
        hs = self.team_stats[home_db]
        as_ = self.team_stats[away_db]
        
        # xG Hesaplama
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        # İlk yarı xG (yaklaşık %45-50 oranında)
        h_xg_ht = h_xg * 0.47
        a_xg_ht = a_xg * 0.47
        
        # Poisson Olasılıkları (0-7 gol arası)
        h_probs = [poisson.pmf(i, h_xg) for i in range(8)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(8)]
        h_probs_ht = [poisson.pmf(i, h_xg_ht) for i in range(5)]
        a_probs_ht = [poisson.pmf(i, a_xg_ht) for i in range(5)]
        
        result = {
            "home": home,
            "away": away,
            "xg": {"home": round(h_xg, 2), "away": round(a_xg, 2)},
            "predictions": {}
        }
        
        # --- MAÇ SONUCU (MS) ---
        prob_1, prob_x, prob_2 = 0, 0, 0
        for h in range(8):
            for a in range(8):
                p = h_probs[h] * a_probs[a]
                if h > a: prob_1 += p
                elif h == a: prob_x += p
                else: prob_2 += p
        
        result["predictions"]["ms"] = {
            "1": round(prob_1 * 100, 1),
            "0": round(prob_x * 100, 1),
            "2": round(prob_2 * 100, 1)
        }
        
        # --- İLK YARI SONUCU (IY) ---
        iy_1, iy_0, iy_2 = 0, 0, 0
        for h in range(5):
            for a in range(5):
                p = h_probs_ht[h] * a_probs_ht[a]
                if h > a: iy_1 += p
                elif h == a: iy_0 += p
                else: iy_2 += p
        
        result["predictions"]["iy"] = {
            "İY 1": round(iy_1 * 100, 1),
            "İY 0": round(iy_0 * 100, 1),
            "İY 2": round(iy_2 * 100, 1)
        }
        
        # --- İY/MS (9 kombinasyon) ---
        iyms_probs = {}
        ht_results = {"1": iy_1, "0": iy_0, "2": iy_2}
        ft_results = {"1": prob_1, "0": prob_x, "2": prob_2}
        
        for ht in ["0", "1", "2"]:
            for ft in ["0", "1", "2"]:
                key = f"{ht}/{ft}"
                iyms_probs[key] = round(ht_results[ht] * ft_results[ft] * 100, 1)
        
        result["predictions"]["iyms"] = iyms_probs
        
        # --- HANDİKAP ---
        # Favori takımı belirle (xG'ye göre)
        is_home_fav = h_xg > a_xg
        handicap = {}
        
        for hnd in [1, 2, 3]:
            if is_home_fav:
                # Ev sahibi favori
                prob_win = sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if h - a > hnd)
                handicap[f"Ev -{hnd}"] = round(prob_win * 100, 1)
            else:
                # Deplasman favori
                prob_win = sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if a - h > hnd)
                handicap[f"Dep -{hnd}"] = round(prob_win * 100, 1)
        
        result["predictions"]["handikap"] = handicap
        
        # --- KARŞILIKLI GOL (KG) ---
        btts = sum(h_probs[h] * a_probs[a] for h in range(1, 8) for a in range(1, 8))
        btts_ht = sum(h_probs_ht[h] * a_probs_ht[a] for h in range(1, 5) for a in range(1, 5))
        
        result["predictions"]["kg"] = {
            "KGV": round(btts * 100, 1),
            "İY KGV": round(btts_ht * 100, 1),
            "2Y KGV": round((btts - btts_ht) * 100, 1) if btts > btts_ht else 0
        }
        
        # --- ALT/ÜST (AU) ---
        au = {}
        for threshold in [1.5, 2.5, 3.5, 4.5]:
            over = sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if h + a > threshold)
            au[f"{threshold} Üst"] = round(over * 100, 1)
        
        # İlk yarı alt/üst
        for threshold in [1.5, 2.5]:
            over_ht = sum(h_probs_ht[h] * a_probs_ht[a] for h in range(5) for a in range(5) if h + a > threshold)
            au[f"İY {threshold} Üst"] = round(over_ht * 100, 1)
        
        result["predictions"]["au"] = au
        
        # --- TOPLAM GOL ---
        tg = {}
        tg["0-1"] = round(sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if h + a <= 1) * 100, 1)
        tg["2-3"] = round(sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if 2 <= h + a <= 3) * 100, 1)
        tg["4-5"] = round(sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if 4 <= h + a <= 5) * 100, 1)
        tg["+6"] = round(sum(h_probs[h] * a_probs[a] for h in range(8) for a in range(8) if h + a >= 6) * 100, 1)
        
        result["predictions"]["toplamgol"] = tg
        
        return result

predictor = MatchPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prematch')
def prematch():
    return render_template('prematch.html')

@app.route('/prematch_bahisler')
def prematch_bahisler():
    return render_template('prematch_bahisler.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})

@app.route('/prematch/<bet_type>')
def get_prematch(bet_type):
    """Belirli bahis türü için top 10 maçı döndür"""
    try:
        r = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=10)
        d = r.json()
        matches = []
        
        if "sg" in d and "EA" in d["sg"]:
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue
                
                home = m.get("HN")
                away = m.get("AN")
                
                prediction = predictor.predict_all(home, away)
                if not prediction: continue
                
                matches.append({
                    "home": home,
                    "away": away,
                    "date": f"{m.get('D')} {m.get('T')}",
                    "league": m.get("LN") or "Lig",
                    "prediction": prediction["predictions"].get(bet_type, {})
                })
        
        # İlgili bahis türüne göre sırala
        if bet_type in matches[0]["prediction"]:
            # Tüm seçeneklerin maksimum olasılığına göre sırala
            for match in matches:
                match["max_prob"] = max(match["prediction"].values())
            
            matches.sort(key=lambda x: x["max_prob"], reverse=True)
            matches = matches[:10]
        
        return jsonify({"success": True, "count": len(matches), "matches": matches})
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
