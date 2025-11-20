import os
import logging
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import requests
from scipy.stats import poisson
from rapidfuzz import process, fuzz

# --- AYARLAR VE KURULUM ---

# Dosya yollarını dinamik olarak belirle (Render uyumlu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Flask uygulamasını başlat
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# Loglama (Render Loglarında görmek için önemli)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger("PredictaPRO")

# Nesine Headerları
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# --- TAHMİN MOTORU (THE BRAIN) ---

class MatchPredictor:
    def __init__(self):
        self.df = None
        self.team_stats = {}
        self.team_names_map = {}
        self.avg_home_goals = 1.5 # Varsayılan
        self.avg_away_goals = 1.2 # Varsayılan
        
        # Başlarken veritabanını yükle
        self.load_database()

    def load_database(self):
        """CSV Veri tabanını yerel diskten yükler ve işler."""
        logger.info(f"📂 Veritabanı yükleniyor... Yol: {CSV_PATH}")
        
        if not os.path.exists(CSV_PATH):
            logger.error(f"❌ KRİTİK HATA: CSV dosyası bulunamadı! {CSV_PATH}")
            logger.info(f"Mevcut Klasör Yapısı: {os.listdir(BASE_DIR)}")
            if os.path.exists(os.path.join(BASE_DIR, 'data')):
                logger.info(f"Data Klasörü: {os.listdir(os.path.join(BASE_DIR, 'data'))}")
            return

        try:
            # CSV okuma
            self.df = pd.read_csv(CSV_PATH, encoding='utf-8', on_bad_lines='skip')
            
            # Sütun isimlerini temizle (küçük harf, boşluksuz)
            self.df.columns = [c.lower().strip().replace(' ', '_') for c in self.df.columns]
            
            # Veri tiplerini zorla (Sayısal)
            cols_to_numeric = ['home_score', 'away_score']
            for col in cols_to_numeric:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
            
            # İstatistikleri Hesapla
            self._calculate_stats()
            logger.info(f"✅ Veritabanı başarıyla yüklendi. {len(self.df)} maç, {len(self.team_stats)} takım.")
            
        except Exception as e:
            logger.error(f"❌ Veritabanı işleme hatası: {e}")

    def _calculate_stats(self):
        """Takımların güçlerini hesaplar."""
        if self.df is None or self.df.empty: return
        
        # Lig Ortalamaları
        if 'home_score' in self.df.columns:
            self.avg_home_goals = self.df['home_score'].mean() or 1.5
            self.avg_away_goals = self.df['away_score'].mean() or 1.2
        
        stats = {}
        # Tüm takımları bul
        home_teams = self.df['home_team'].unique() if 'home_team' in self.df.columns else []
        away_teams = self.df['away_team'].unique() if 'away_team' in self.df.columns else []
        teams = set(list(home_teams) + list(away_teams))
        
        for team in teams:
            if pd.isna(team) or str(team).strip() == '': continue
            
            # Ev ve Deplasman maçlarını filtrele
            h_matches = self.df[self.df['home_team'] == team]
            a_matches = self.df[self.df['away_team'] == team]
            
            h_games = len(h_matches)
            a_games = len(a_matches)
            
            # Basit Atak/Defans Gücü Hesaplama
            # (Veri azsa varsayılan 1.0 ata)
            att_home = (h_matches['home_score'].sum() / h_games / self.avg_home_goals) if h_games > 5 else 1.0
            def_home = (h_matches['away_score'].sum() / h_games / self.avg_away_goals) if h_games > 5 else 1.0
            
            att_away = (a_matches['away_score'].sum() / a_games / self.avg_away_goals) if a_games > 5 else 1.0
            def_away = (a_matches['home_score'].sum() / a_games / self.avg_home_goals) if a_games > 5 else 1.0
            
            # Form (Son 5 Maç)
            # Tarih sütunu sorunluysa diye index sırasına güveniyoruz (CSV genelde tarihe göre sıralıdır)
            last_5_form = []
            recent_matches = pd.concat([h_matches, a_matches]).sort_index().tail(5)
            
            for _, row in recent_matches.iterrows():
                try:
                    h_s, a_s = row['home_score'], row['away_score']
                    is_home = row['home_team'] == team
                    
                    if h_s > a_s: res = 'W' if is_home else 'L'
                    elif h_s < a_s: res = 'L' if is_home else 'W'
                    else: res = 'D'
                    last_5_form.append(res)
                except:
                    continue
            
            stats[team] = {
                'att_h': att_home, 'def_h': def_home,
                'att_a': att_away, 'def_a': def_away,
                'form': "".join(last_5_form)
            }
            
        self.team_stats = stats

    def find_team(self, name):
        """Fuzzy logic ile takım ismi eşleştirir."""
        if not name: return None
        
        # Önbellekte varsa direkt döndür
        if name in self.team_names_map:
            return self.team_names_map[name]
            
        if not self.team_stats:
            return None

        # Eşleştirme yap
        match = process.extractOne(name, self.team_stats.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=70)
        
        if match:
            found_name = match[0]
            self.team_names_map[name] = found_name # Cache'e at
            return found_name
        
        return None

    def predict(self, home, away):
        """xG ve Poisson ile tahmin yapar."""
        home_db = self.find_team(home)
        away_db = self.find_team(away)
        
        if not home_db or not away_db:
            return None
            
        hs = self.team_stats.get(home_db)
        as_ = self.team_stats.get(away_db)
        
        # xG Hesaplama
        home_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        away_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        # Poisson Olasılıkları
        h_probs = [poisson.pmf(i, home_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, away_xg) for i in range(6)]
        
        prob_1, prob_x, prob_2, prob_over = 0, 0, 0, 0
        
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                if h > a: prob_1 += p
                elif h == a: prob_x += p
                else: prob_2 += p
                if (h+a) > 2.5: prob_over += p
                
        return {
            "home_team_db": home_db,
            "away_team_db": away_db,
            "stats": {
                "home_xg": round(home_xg, 2),
                "away_xg": round(away_xg, 2)
            },
            "probs": {
                "1": round(prob_1 * 100, 1),
                "X": round(prob_x * 100, 1),
                "2": round(prob_2 * 100, 1),
                "over": round(prob_over * 100, 1)
            }
        }

# --- UYGULAMA BAŞLATMA ---

predictor = MatchPredictor()

@app.route('/')
def index():
    """HTML Arayüzünü Sunar"""
    return render_template('index.html')

@app.route('/api/matches/live')
def live_matches():
    """Nesine'den veri çeker, analiz eder ve JSON döndürür"""
    try:
        logger.info("🔄 Nesine API isteği gönderiliyor...")
        response = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=15)
        
        if response.status_code != 200:
            raise Exception(f"Nesine Hata Kodu: {response.status_code}")
            
        data = response.json()
        matches = []
        
        if "sg" in data and "EA" in data["sg"]:
            for m in data["sg"]["EA"]:
                if m.get("GT") != 1: continue # Sadece bülten
                
                # Oranları Çek
                odds = {}
                for market in m.get("MA", []):
                    # MS
                    if market.get("MTID") == 1:
                        for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["1"] = o.get("O")
                            if o.get("N") == 2: odds["X"] = o.get("O")
                            if o.get("N") == 3: odds["2"] = o.get("O")
                    # Alt/Üst
                    if market.get("MTID") == 450:
                        odds["Over/Under +2.5"] = {}
                        for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["Over/Under +2.5"]["Over +2.5"] = o.get("O")
                            if o.get("N") == 2: odds["Over/Under +2.5"]["Under +2.5"] = o.get("O")
                
                # Sadece oranları olan maçları işle
                if "1" in odds:
                    home = m.get("HN")
                    away = m.get("AN")
                    
                    # Tahmin Yap
                    pred = predictor.predict(home, away)
                    
                    matches.append({
                        "id": str(m.get("C")),
                        "home": home,
                        "away": away,
                        "date": f"{m.get('D')} {m.get('T')}",
                        "league": m.get("LN"),
                        "odds": odds,
                        "prediction": pred
                    })
        
        return jsonify({
            "success": True,
            "count": len(matches),
            "matches": matches
        })
        
    except Exception as e:
        logger.error(f"❌ API Hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "db_loaded": predictor.df is not None})

if __name__ == '__main__':
    # Render PORT environment variable'ını kullanır
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
