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

# --- AYARLAR ---
app = Flask(__name__)
CORS(app)

# Loglama Yapısı
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger("PredictaEngine")

# Nesine Ayarları
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# Veri Yolu
DATA_PATH = os.path.join("data", "database.csv")

# --- ANALİZ MOTORU (ENGINE) ---

class MatchPredictor:
    def __init__(self):
        self.df = None
        self.team_stats = {}
        self.team_names_map = {} # Nesine İsmi -> CSV İsmi eşleşmesi
        self.load_database()

    def load_database(self):
        """CSV Veri tabanını yükler ve temizler."""
        if not os.path.exists(DATA_PATH):
            logger.warning(f"⚠️ Veri dosyası bulunamadı: {DATA_PATH}. Analizler boş dönecek.")
            return

        try:
            # CSV okuma (Encoding hatası olursa 'latin1' dene)
            self.df = pd.read_csv(DATA_PATH, encoding='utf-8', on_bad_lines='skip')
            
            # Sütun isimlerini standartlaştır (Küçük harf, boşlukları sil)
            self.df.columns = [c.lower().strip().replace(' ', '_') for c in self.df.columns]
            
            # Kritik Sütun Kontrolü
            required = ['home_team', 'away_team', 'home_score', 'away_score'] # result opsiyonel
            missing = [c for c in required if c not in self.df.columns]
            
            if missing:
                # Senin bozuk CSV formatına (odds_2, date, home_team...) uyum sağlamaya çalış
                logger.warning(f"Standart sütunlar eksik: {missing}. Alternatif haritalama deneniyor...")
                # Burada senin CSV yapına göre rename yapılabilir gerekirse
                # self.df.rename(columns={'home': 'home_team'}, inplace=True) vb.
            
            # Veri Tiplerini Düzelt
            self.df['home_score'] = pd.to_numeric(self.df['home_score'], errors='coerce').fillna(0).astype(int)
            self.df['away_score'] = pd.to_numeric(self.df['away_score'], errors='coerce').fillna(0).astype(int)
            
            # Takım İstatistiklerini Hesapla (Cache)
            self._calculate_stats()
            logger.info(f"✅ Veritabanı yüklendi: {len(self.df)} maç, {len(self.team_stats)} takım.")
            
        except Exception as e:
            logger.error(f"❌ Veritabanı yükleme hatası: {e}")

    def _calculate_stats(self):
        """Takımların saldırı ve savunma güçlerini hesaplar."""
        stats = {}
        
        # Lig Ortalamaları
        avg_home_goals = self.df['home_score'].mean()
        avg_away_goals = self.df['away_score'].mean()
        
        teams = pd.concat([self.df['home_team'], self.df['away_team']]).unique()
        
        for team in teams:
            if pd.isna(team): continue
            
            # Ev Sahibi Performansı
            home_matches = self.df[self.df['home_team'] == team]
            home_goals_for = home_matches['home_score'].sum()
            home_goals_ag = home_matches['away_score'].sum()
            home_count = len(home_matches)
            
            # Deplasman Performansı
            away_matches = self.df[self.df['away_team'] == team]
            away_goals_for = away_matches['away_score'].sum()
            away_goals_ag = away_matches['home_score'].sum()
            away_count = len(away_matches)
            
            # Saldırı Gücü (Attack Strength)
            # (Takımın Attığı Ort. Gol) / (Ligin Ort. Golü)
            att_home = (home_goals_for / home_count / avg_home_goals) if home_count > 0 else 1.0
            att_away = (away_goals_for / away_count / avg_away_goals) if away_count > 0 else 1.0
            
            # Savunma Gücü (Defense Strength) - Düşük olması iyidir (Az yiyor demektir) ama hesapta ters kullanacağız
            # (Takımın Yediği Ort. Gol) / (Ligin Ort. Golü)
            def_home = (home_goals_ag / home_count / avg_away_goals) if home_count > 0 else 1.0
            def_away = (away_goals_ag / away_count / avg_home_goals) if away_count > 0 else 1.0
            
            # Form (Son 5 Maç)
            all_matches = pd.concat([home_matches, away_matches]).sort_index(ascending=True) # Tarih olmadığı için index'e güvendik
            last_5 = []
            for idx, row in all_matches.tail(5).iterrows():
                is_home = row['home_team'] == team
                h_s, a_s = row['home_score'], row['away_score']
                if h_s > a_s: res = 'W' if is_home else 'L'
                elif h_s < a_s: res = 'L' if is_home else 'W'
                else: res = 'D'
                last_5.append(res)
            
            stats[team] = {
                'attack_home': att_home,
                'defense_home': def_home,
                'attack_away': att_away,
                'defense_away': def_away,
                'form': "".join(last_5),
                'matches_played': home_count + away_count
            }
            
        self.team_stats = stats
        self.avg_home_goals = avg_home_goals
        self.avg_away_goals = avg_away_goals

    def find_team_name(self, nesine_name):
        """Nesine'den gelen ismi CSV'deki isimle eşleştirir (Fuzzy Logic)."""
        # Cache kontrolü
        if nesine_name in self.team_names_map:
            return self.team_names_map[nesine_name]
            
        if not self.team_stats:
            return None

        # En iyi eşleşmeyi bul
        # Score cutoff 80: %80 benzerlik yoksa eşleşme yok say
        match = process.extractOne(nesine_name, self.team_stats.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=75)
        
        if match:
            csv_name = match[0]
            score = match[1]
            logger.info(f"🔗 Eşleşti: {nesine_name} -> {csv_name} (Skor: {score})")
            self.team_names_map[nesine_name] = csv_name
            return csv_name
        else:
            # logger.warning(f"❌ Eşleşme bulunamadı: {nesine_name}")
            return None

    def predict(self, home_nesine, away_nesine):
        """İki takım için Poisson Tahmini üretir."""
        home_csv = self.find_team_name(home_nesine)
        away_csv = self.find_team_name(away_nesine)
        
        if not home_csv or not away_csv:
            return None # Veri yoksa tahmin yok
            
        h_stats = self.team_stats.get(home_csv)
        a_stats = self.team_stats.get(away_csv)
        
        if not h_stats or not a_stats:
            return None
            
        # Gol Beklentisi (Expected Goals - xG) Hesabı
        # Ev xG = Ev Saldırı * Dep Savunma * Lig Ortalaması
        home_xg = h_stats['attack_home'] * a_stats['defense_away'] * self.avg_home_goals
        away_xg = a_stats['attack_away'] * h_stats['defense_home'] * self.avg_away_goals
        
        # Poisson Dağılımı ile Olasılıklar
        # 0'dan 5 gole kadar olasılıkları hesapla
        h_probs = [poisson.pmf(i, home_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, away_xg) for i in range(6)]
        
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        over_25_prob = 0
        
        for h in range(6):
            for a in range(6):
                prob = h_probs[h] * a_probs[a]
                if h > a: home_win_prob += prob
                elif h == a: draw_prob += prob
                else: away_win_prob += prob
                
                if (h + a) > 2.5: over_25_prob += prob
                
        return {
            "home_team_db": home_csv,
            "away_team_db": away_csv,
            "stats": {
                "home_form": h_stats['form'],
                "away_form": a_stats['form'],
                "home_xg": round(home_xg, 2),
                "away_xg": round(away_xg, 2)
            },
            "probs": {
                "1": round(home_win_prob * 100, 1),
                "X": round(draw_prob * 100, 1),
                "2": round(away_win_prob * 100, 1),
                "over_25": round(over_25_prob * 100, 1)
            }
        }

# Motoru Başlat
predictor = MatchPredictor()

# --- ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/matches/live', methods=['GET'])
def get_live_matches():
    try:
        logger.info("Nesine'den veri çekiliyor...")
        response = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=10)
        data = response.json()
        
        processed_matches = []
        
        # Sadece Futbol (EA)
        if "sg" in data and "EA" in data["sg"]:
            for m in data["sg"]["EA"]:
                if m.get("GT") != 1: continue # Bülten maçı değilse geç
                
                match_id = str(m.get("C"))
                home = m.get("HN")
                away = m.get("AN")
                
                # Oranları Al
                odds = {}
                for market in m.get("MA", []):
                    if market.get("MTID") == 1: # MS
                        for o in market.get("OCA", []):
                            if o.get("N") == 1: odds["1"] = o.get("O")
                            if o.get("N") == 2: odds["X"] = o.get("O")
                            if o.get("N") == 3: odds["2"] = o.get("O")
                
                # Tahmin Motorunu Çalıştır
                prediction = predictor.predict(home, away)
                
                match_data = {
                    "id": match_id,
                    "home": home,
                    "away": away,
                    "date": m.get("D") + " " + m.get("T"),
                    "league": m.get("LN"),
                    "odds": odds,
                    "prediction": prediction # Tahmin sonucu buraya eklenir
                }
                processed_matches.append(match_data)
                
        return jsonify({
            "success": True,
            "count": len(processed_matches),
            "matches": processed_matches
        })
        
    except Exception as e:
        logger.error(f"API Hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# app.py'nin EN ALTI böyle olmalı:
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000)) # Render genelde 10000 verir, yoksa 5000
    app.run(host='0.0.0.0', port=port)
