import os
import logging
import sqlite3
import json
import time
from datetime import datetime
from functools import lru_cache
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import requests
from rapidfuzz import process, fuzz

# --- YAPILANDIRMA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

# --- GÜVENLİK ---
NESINE_AUTH = os.environ.get("NESINE_AUTH")
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": NESINE_AUTH,
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

# --- VERİTABANI ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER UNIQUE, 
                date TEXT,
                home TEXT,
                away TEXT,
                prob_1 REAL,
                prob_x REAL,
                prob_2 REAL,
                prob_over REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def save_prediction_to_db(match_data):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            probs = match_data['prediction']['probs']
            cursor.execute('''
                INSERT OR IGNORE INTO history 
                (match_id, date, home, away, prob_1, prob_x, prob_2, prob_over)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['id'], match_data['date'], match_data['home'], match_data['away'],
                probs['1'], probs['X'], probs['2'], probs['over']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"DB Save Error: {e}")

init_db()

# --- GELİŞMİŞ TAHMİN MOTORU (MONTE CARLO + TIME DECAY) ---
class AdvancedMatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.league_stats = {} # Lig bazlı istatistikler
        self.team_list = []
        self.load_database()

    def load_database(self):
        if not os.path.exists(CSV_PATH):
            logger.warning("⚠️ CSV Dosyası Bulunamadı!")
            return
        try:
            # 'Div' (Lig) ve 'Date' sütunlarını da okuyoruz
            use_cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            # CSV'de bu sütunlar yoksa hata vermemesi için kontrol
            df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
            
            # Gerekli sütun eşleştirmesi (CSV başlıkların farklıysa burayı düzenle)
            # Örn: senin CSV'de Lig sütunu 'League' ise burayı değiştir.
            # Standart football-data.co.uk formatı varsayılmıştır: Div, Date, HomeTeam...
            
            # Standartlaştırma
            df.rename(columns={
                'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 
                'FTHG': 'home_score', 'FTAG': 'away_score',
                'Div': 'league', 'Date': 'date'
            }, inplace=True)

            # Tarih formatı düzeltme ve sıralama (Zaman ağırlığı için şart)
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            df.sort_values('date', inplace=True)
            
            # Sayısal dönüşüm
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0)
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0)

            # Lig bazlı ortalamalar
            self._calculate_league_stats(df)
            
            # Takım bazlı güç puanları (Time Decay ile)
            self._calculate_team_ratings(df)
            
            self.team_list = list(self.team_stats.keys())
            del df
            logger.info(f"✅ Gelişmiş Veritabanı Yüklendi: {len(self.team_list)} takım, {len(self.league_stats)} lig.")
            
        except Exception as e:
            logger.error(f"❌ DB Yükleme Hatası: {e}")

    def _calculate_league_stats(self, df):
        """Her ligin gol karakteristiğini çıkarır (Örn: Eredivisie bol gollü, Ligue 2 az gollü)"""
        if 'league' not in df.columns:
            # Lig sütunu yoksa global ortalama kullan
            self.league_stats['global'] = {
                'avg_h': df['home_score'].mean(),
                'avg_a': df['away_score'].mean()
            }
            return

        stats = df.groupby('league')[['home_score', 'away_score']].mean()
        for league, row in stats.iterrows():
            self.league_stats[league] = {
                'avg_h': row['home_score'],
                'avg_a': row['away_score']
            }

    def _calculate_team_ratings(self, df):
        """
        Time Decay (Zaman Aşımı) ve Form Hesaplama.
        Pandas ewm (Exponential Weighted Mean) kullanarak yakın tarihli maçlara daha çok ağırlık verir.
        """
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in teams:
            # Takımın ev ve deplasman maçlarını al
            home_matches = df[df['home_team'] == team].copy()
            away_matches = df[df['away_team'] == team].copy()
            
            if len(home_matches) < 5 or len(away_matches) < 5: continue
            
            league = home_matches['league'].iloc[0] if 'league' in home_matches else 'global'
            lg_avg_h = self.league_stats.get(league, {}).get('avg_h', 1.5)
            lg_avg_a = self.league_stats.get(league, {}).get('avg_a', 1.2)

            # --- TIME DECAY (ZAMAN AĞIRLIĞI) ---
            # span=20 -> Son 20 maçın etkisi baskın olur (Yaklaşık 1 sezonun yarısı)
            att_h_series = home_matches['home_score'].ewm(span=20).mean()
            def_h_series = home_matches['away_score'].ewm(span=20).mean()
            att_a_series = away_matches['away_score'].ewm(span=20).mean()
            def_a_series = away_matches['home_score'].ewm(span=20).mean()

            # Son değerleri al (En güncel güç)
            curr_att_h = att_h_series.iloc[-1] / lg_avg_h
            curr_def_h = def_h_series.iloc[-1] / lg_avg_a
            curr_att_a = att_a_series.iloc[-1] / lg_avg_a
            curr_def_a = def_a_series.iloc[-1] / lg_avg_h

            # --- FORM (SON 5 MAÇ) ---
            # Son 5 maçın gol ortalaması / Sezon ortalaması
            last5_h = home_matches['home_score'].tail(5).mean()
            last5_a = away_matches['away_score'].tail(5).mean()
            
            form_factor_h = (last5_h / (home_matches['home_score'].mean() + 0.1)) 
            form_factor_a = (last5_a / (away_matches['away_score'].mean() + 0.1))
            
            # Form faktörünü yumuşat (Çok uçuk değerler çıkmasın diye max 1.3, min 0.7)
            form_factor_h = np.clip(form_factor_h, 0.8, 1.2)
            form_factor_a = np.clip(form_factor_a, 0.8, 1.2)

            self.team_stats[team] = {
                'league': league,
                'att_h': curr_att_h, 'def_h': curr_def_h,
                'att_a': curr_att_a, 'def_a': curr_def_a,
                'form_h': form_factor_h,
                'form_a': form_factor_a
            }

    @lru_cache(maxsize=4096)
    def find_team_cached(self, name):
        if not name or not self.team_stats: return None
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        match = process.extractOne(clean_name, self.team_list, scorer=fuzz.token_sort_ratio, score_cutoff=70)
        return match[0] if match else None

    def monte_carlo_simulation(self, h_xg, a_xg, simulations=10000):
        """
        Poisson yerine Monte Carlo Simülasyonu.
        Maçı 10.000 kez oynatır ve olasılıkları sayar.
        """
        # Numpy ile vektörel işlem (Çok hızlıdır)
        h_goals = np.random.poisson(h_xg, simulations)
        a_goals = np.random.poisson(a_xg, simulations)

        wins = np.sum(h_goals > a_goals)
        draws = np.sum(h_goals == a_goals)
        losses = np.sum(h_goals < a_goals)
        
        over_25 = np.sum((h_goals + a_goals) > 2.5)
        btts = np.sum((h_goals > 0) & (a_goals > 0))

        return {
            "1": (wins / simulations) * 100,
            "X": (draws / simulations) * 100,
            "2": (losses / simulations) * 100,
            "over": (over_25 / simulations) * 100,
            "btts_yes": (btts / simulations) * 100
        }

    def predict(self, home, away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        if not home_db or not away_db: return None
            
        hs = self.team_stats[home_db]
        as_ = self.team_stats[away_db]
        
        # Lig ortalamalarını çek
        league = hs.get('league', 'global')
        lg_stats = self.league_stats.get(league, self.league_stats.get('global', {'avg_h': 1.5, 'avg_a': 1.2}))
        
        # xG Hesaplama (Form ve Lig Ortalamaları Dahil)
        # Formül: Lig Ort * Ev Hücum * Dep Defans * Ev Sahibi Formu
        h_xg = lg_stats['avg_h'] * hs['att_h'] * as_['def_a'] * hs['form_h']
        a_xg = lg_stats['avg_a'] * as_['att_a'] * hs['def_h'] * as_['form_a']

        # Monte Carlo Simülasyonu Başlat
        probs = self.monte_carlo_simulation(h_xg, a_xg)
        
        return {
            "stats": {"home_xg": round(h_xg, 2), "away_xg": round(a_xg, 2)},
            "probs": {
                "1": round(probs['1'], 1),
                "X": round(probs['X'], 1),
                "2": round(probs['2'], 1),
                "over": round(probs['over'], 1),
                "under": round(100 - probs['over'], 1),
                "btts_yes": round(probs['btts_yes'], 1),
                "btts_no": round(100 - probs['btts_yes'], 1)
            }
        }

predictor = AdvancedMatchPredictor()

# --- ROUTE'LAR (DEĞİŞMEDİ) ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/history')
def history_page():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
    return render_template('history.html', matches=rows)

@app.route('/api/matches/live')
def live():
    if not NESINE_AUTH: return jsonify({"error": "NESINE_AUTH missing"}), 500

    try:
        r = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=15)
        if r.status_code != 200: return jsonify({"error": "Nesine API Error"}), 502
        
        d = r.json()
        matches = []
        
        if "sg" in d and "EA" in d["sg"]:
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue 
                
                odds = {}
                markets = m.get("MA", [])
                for market in markets:
                    mtid = market.get("MTID")
                    oca = market.get("OCA", [])
                    if mtid == 1:
                        for o in oca:
                            if o["N"] == 1: odds["1"] = o["O"]
                            elif o["N"] == 2: odds["X"] = o["O"]
                            elif o["N"] == 3: odds["2"] = o["O"]
                    elif mtid == 450:
                         if "Over/Under +2.5" not in odds: odds["Over/Under +2.5"] = {}
                         for o in oca:
                             if o["N"] == 1: odds["Over/Under +2.5"]["Over +2.5"] = o["O"]
                             if o["N"] == 2: odds["Over/Under +2.5"]["Under +2.5"] = o["O"]

                if "1" not in odds: continue

                prediction = predictor.predict(m.get("HN"), m.get("AN"))
                if prediction is None: continue 
                
                match_data = {
                    "id": m.get("C"),
                    "home": m.get("HN"),
                    "away": m.get("AN"),
                    "date": f"{m.get('D')} {m.get('T')}",
                    "league": m.get("LN") or "Lig",
                    "odds": odds,
                    "prediction": prediction
                }
                save_prediction_to_db(match_data)
                matches.append(match_data)
        
        return jsonify({"success": True, "count": len(matches), "matches": matches})

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"success": False, "error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
