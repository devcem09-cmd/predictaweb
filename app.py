import os
import json
import logging
import requests
import atexit
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# --- YAPILANDIRMA ---
load_dotenv() # .env dosyasını okur

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# Veritabanı (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictapro.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- VERİTABANI MODELİ ---
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True)
    league = db.Column(db.String(50))
    home_team = db.Column(db.String(50))
    away_team = db.Column(db.String(50))
    date = db.Column(db.DateTime)
    odds = db.Column(db.Text) 
    
    # Tahmin Verileri
    prob_home = db.Column(db.Float, default=0.0)
    prob_draw = db.Column(db.Float, default=0.0)
    prob_away = db.Column(db.Float, default=0.0)
    prob_over_25 = db.Column(db.Float, default=0.0)
    prob_btts = db.Column(db.Float, default=0.0)
    
    status = db.Column(db.String(20), default="Pending")

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "league": self.league,
            "home": self.home_team,
            "away": self.away_team,
            "date": self.date.strftime("%Y-%m-%d %H:%M"),
            "odds": json.loads(self.odds) if self.odds else {},
            "probs": {
                "1": round(self.prob_home * 100, 1),
                "X": round(self.prob_draw * 100, 1),
                "2": round(self.prob_away * 100, 1),
                "over": round(self.prob_over_25 * 100, 1),
                "btts": round(self.prob_btts * 100, 1)
            }
        }

# --- TAHMİN MOTORU ---
class MatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.team_list = []
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı başlatılıyor... Yol: {CSV_PATH}")
        
        if not os.path.exists(CSV_PATH):
            logger.warning(f"⚠️ UYARI: CSV Bulunamadı ({CSV_PATH}).")
            return

        try:
            # CSV Okuma (Senin formatına uygun)
            required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
            df = pd.read_csv(CSV_PATH, usecols=required_cols, encoding='utf-8', on_bad_lines='skip')
            
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            
            self._calculate_stats(df)
            self.team_list = list(self.team_stats.keys())
            
            del df
            logger.info(f"✅ Veritabanı Hazır. {len(self.team_stats)} takım yüklendi.")
            
        except Exception as e:
            logger.error(f"❌ DB Hata: {e}")

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
            if team not in home_stats.index or team not in away_stats.index: continue
            
            if home_stats.loc[team, 'count'] < 3 or away_stats.loc[team, 'count'] < 3: continue

            self.team_stats[team] = {
                'att_h': home_stats.loc[team, 'mean'] / self.avg_home_goals,
                'def_h': home_conceded.loc[team] / self.avg_away_goals,
                'att_a': away_stats.loc[team, 'mean'] / self.avg_away_goals,
                'def_a': away_conceded.loc[team] / self.avg_home_goals
            }

    @lru_cache(maxsize=2048)
    def find_team_cached(self, name):
        if not name or not self.team_list: return None
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        match = process.extractOne(clean_name, self.team_list, scorer=fuzz.token_set_ratio, score_cutoff=60)
        return match[0] if match else None

    def predict(self, home, away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        if not home_db or not away_db: return 0, 0, 0, 0, 0
            
        hs = self.team_stats[home_db]
        as_ = self.team_stats[away_db]
        
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        p_1, p_x, p_2 = 0, 0, 0
        p_over, p_btts = 0, 0
        
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                if h > a: p_1 += p
                elif h == a: p_x += p
                else: p_2 += p
                if (h + a) > 2.5: p_over += p
                if h > 0 and a > 0: p_btts += p

        return p_1, p_x, p_2, p_over, p_btts

predictor = MatchPredictor()

# --- NESİNE VERİ ÇEKME (DÜZELTİLDİ: ID 14 ve 450) ---
def fetch_live_data():
    with app.app_context():
        # .env kontrolü
        auth_token = os.getenv("NESINE_AUTH")
        if not auth_token:
            logger.error("⚠️ NESINE_AUTH token bulunamadı!")
            return

        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Authorization": auth_token,
            "Origin": "https://www.nesine.com"
        }

        try:
            logger.info("🔄 Nesine'den veri çekiliyor...")
            r = requests.get(url, headers=headers, timeout=15)
            d = r.json()
            
            if "sg" not in d or "EA" not in d["sg"]: return

            count = 0
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue # Sadece Futbol

                match_code = str(m.get("C"))
                
                # Oranları Varsayılan Olarak Boş Ata
                odds = {"ms1": "-", "msx": "-", "ms2": "-", "alt": "-", "ust": "-", "kgvar": "-", "kgyok": "-"}
                
                # Marketleri Gez
                markets = m.get("MA", [])
                for market in markets:
                    mtid = market.get("MTID")
                    oca = market.get("OCA", [])
                    
                    # MTID 1: Maç Sonucu (MS)
                    if mtid == 1:
                        for o in oca:
                            if o["N"] == 1: odds["ms1"] = o["O"]
                            elif o["N"] == 2: odds["msx"] = o["O"]
                            elif o["N"] == 3: odds["ms2"] = o["O"]
                    
                    # MTID 14: Karşılıklı Gol (KG Var/Yok) - SENİN VERİNE GÖRE
                    elif mtid == 14:
                        for o in oca:
                            if o["N"] == 1: odds["kgvar"] = o["O"] # N:1 -> KG VAR
                            elif o["N"] == 2: odds["kgyok"] = o["O"] # N:2 -> KG YOK
                            
                    # MTID 450: 2.5 Gol Alt/Üst - SENİN VERİNE GÖRE
                    elif mtid == 450:
                         for o in oca:
                             if o["N"] == 1: odds["ust"] = o["O"] # N:1 -> ÜST
                             elif o["N"] == 2: odds["alt"] = o["O"] # N:2 -> ALT

                if odds["ms1"] == "-": continue

                # Tahmin Yap
                p1, px, p2, pover, pbtts = predictor.predict(m.get("HN"), m.get("AN"))

                # Veritabanına Yaz
                existing = Match.query.filter_by(code=match_code).first()
                
                if not existing:
                    new_match = Match(
                        code=match_code,
                        league=m.get("LN"),
                        home_team=m.get("HN"),
                        away_team=m.get("AN"),
                        date=datetime.strptime(f"{m.get('D')} {m.get('T')}", "%d.%m.%Y %H:%M"),
                        odds=json.dumps(odds),
                        prob_home=p1, prob_draw=px, prob_away=p2,
                        prob_over_25=pover, prob_btts=pbtts
                    )
                    db.session.add(new_match)
                    count += 1
                else:
                    existing.odds = json.dumps(odds)
            
            db.session.commit()
            logger.info(f"✅ Başarılı: {count} yeni maç eklendi, oranlar güncellendi.")

        except Exception as e:
            logger.error(f"❌ API Hatası: {e}")

# --- ZAMANLAYICI ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_live_data, trigger="interval", minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- ROTALAR ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/matches')
def get_matches():
    sort_by = request.args.get('sort_by', 'default')
    cutoff = datetime.now() - timedelta(hours=2)
    matches = Match.query.filter(Match.date >= cutoff).all()
    data = [m.to_dict() for m in matches]
    
    if sort_by == 'prob_high':
        data.sort(key=lambda x: max(x['probs']['1'], x['probs']['X'], x['probs']['2']), reverse=True)
    elif sort_by == 'prob_over':
        data.sort(key=lambda x: x['probs']['over'], reverse=True)
    else:
        data.sort(key=lambda x: x['date'])

    return jsonify(data)

@app.route('/health')
def health(): return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        try: fetch_live_data()
        except: pass
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
