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
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaAI")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# Veritabanı Yolu
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(INSTANCE_DIR, 'predictapro.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
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
    prob_home = db.Column(db.Float, default=0.0)
    prob_draw = db.Column(db.Float, default=0.0)
    prob_away = db.Column(db.Float, default=0.0)
    prob_over_25 = db.Column(db.Float, default=0.0)
    prob_btts = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default="Pending")
    score_home = db.Column(db.Integer, nullable=True)
    score_away = db.Column(db.Integer, nullable=True)
    result_str = db.Column(db.String(10), nullable=True)
    is_successful = db.Column(db.Boolean, default=False)

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
            },
            "status": self.status,
            "score": f"{self.score_home}-{self.score_away}" if self.score_home is not None else "-",
            "result": self.result_str,
            "success": self.is_successful
        }

# --- GELİŞTİRİLMİŞ TAHMİN MOTORU ---
class MatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.league_stats = {}
        self.team_list = []
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı başlatılıyor... Yol: {CSV_PATH}")
        if not os.path.exists(CSV_PATH):
            logger.warning(f"⚠️ CSV Bulunamadı!")
            return

        try:
            cols = ['home_team', 'away_team', 'home_score', 'away_score', 'date', 'league']
            
            try:
                df = pd.read_csv(CSV_PATH, usecols=lambda c: c.lower() in cols, encoding='utf-8', on_bad_lines='skip')
            except:
                df = pd.read_csv(CSV_PATH, encoding='utf-8', on_bad_lines='skip')

            df.columns = [c.lower().strip() for c in df.columns]
            
            rename_map = {
                'hometeam': 'home_team', 'awayteam': 'away_team', 
                'fthg': 'home_score', 'ftag': 'away_score',
                'evsahibi': 'home_team', 'deplasman': 'away_team'
            }
            df.rename(columns=rename_map, inplace=True)

            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values('date', ascending=True)

            if 'league' not in df.columns:
                df['league'] = 'Unknown'

            self._calculate_advanced_stats(df)
            self.team_list = list(self.team_stats.keys())
            
            del df
            logger.info(f"✅ Veritabanı Hazır. {len(self.team_stats)} takım analiz edildi (EMA + Son 10 Maç Form Takibi).")
            
        except Exception as e:
            logger.error(f"❌ DB Hata: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _calculate_advanced_stats(self, df):
        """EMA + Son 10 Maç Form Takibi"""
        if df.empty: return

        # Lig bazlı ortalamalar
        league_group = df.groupby('league')
        for league, data in league_group:
            self.league_stats[league] = {
                'avg_home_goals': data['home_score'].mean() or 1.5,
                'avg_away_goals': data['away_score'].mean() or 1.2
            }

        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        current_ratings = {}
        for team in teams:
            current_ratings[team] = {
                'att_h': 1.0, 'def_h': 1.0,
                'att_a': 1.0, 'def_a': 1.0,
                'form_home': [],
                'form_away': [],
                'recent_goals_h': [],
                'recent_goals_a': [],
                'league': 'Unknown'
            }

        alpha = 0.18  # EMA öğrenme hızı

        for _, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            h_score, a_score = row['home_score'], row['away_score']
            league = row['league']
            
            if home not in current_ratings or away not in current_ratings: 
                continue

            l_stats = self.league_stats.get(league, {'avg_home_goals': 1.5, 'avg_away_goals': 1.2})
            avg_h, avg_a = l_stats['avg_home_goals'], l_stats['avg_away_goals']

            current_ratings[home]['league'] = league
            current_ratings[away]['league'] = league

            # EMA Güncelleme
            h_att_perf = h_score / avg_h if avg_h > 0 else 1.0
            h_def_perf = a_score / avg_a if avg_a > 0 else 1.0
            a_att_perf = a_score / avg_a if avg_a > 0 else 1.0
            a_def_perf = h_score / avg_h if avg_h > 0 else 1.0

            current_ratings[home]['att_h'] = (current_ratings[home]['att_h'] * (1 - alpha)) + (h_att_perf * alpha)
            current_ratings[home]['def_h'] = (current_ratings[home]['def_h'] * (1 - alpha)) + (h_def_perf * alpha)
            current_ratings[away]['att_a'] = (current_ratings[away]['att_a'] * (1 - alpha)) + (a_att_perf * alpha)
            current_ratings[away]['def_a'] = (current_ratings[away]['def_a'] * (1 - alpha)) + (a_def_perf * alpha)

            # ✨ FORM TAKİBİ (Son 10 Maç)
            if h_score > a_score:
                form_points_h = 3
            elif h_score == a_score:
                form_points_h = 1
            else:
                form_points_h = 0
            
            current_ratings[home]['form_home'].append(form_points_h)
            current_ratings[home]['recent_goals_h'].append(h_score)
            
            if len(current_ratings[home]['form_home']) > 10:
                current_ratings[home]['form_home'].pop(0)
                current_ratings[home]['recent_goals_h'].pop(0)

            if a_score > h_score:
                form_points_a = 3
            elif a_score == h_score:
                form_points_a = 1
            else:
                form_points_a = 0
            
            current_ratings[away]['form_away'].append(form_points_a)
            current_ratings[away]['recent_goals_a'].append(a_score)
            
            if len(current_ratings[away]['form_away']) > 10:
                current_ratings[away]['form_away'].pop(0)
                current_ratings[away]['recent_goals_a'].pop(0)

        self.team_stats = current_ratings

    @lru_cache(maxsize=2048)
    def find_team_cached(self, name):
        if not name or not self.team_list: return None
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        match = process.extractOne(clean_name, self.team_list, scorer=fuzz.token_set_ratio, score_cutoff=60)
        return match[0] if match else None

    def predict(self, home, away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        hs = self.team_stats.get(home_db, {
            'att_h': 1.0, 'def_h': 1.0, 'league': 'Unknown', 
            'form_home': [], 'recent_goals_h': []
        })
        as_ = self.team_stats.get(away_db, {
            'att_a': 1.0, 'def_a': 1.0, 'league': 'Unknown',
            'form_away': [], 'recent_goals_a': []
        })
        
        league = hs['league']
        l_stats = self.league_stats.get(league, {'avg_home_goals': 1.5, 'avg_away_goals': 1.2})
        avg_h_goals = l_stats['avg_home_goals']
        avg_a_goals = l_stats['avg_away_goals']

        # Temel xG (EMA bazlı)
        h_xg = hs['att_h'] * as_['def_a'] * avg_h_goals
        a_xg = as_['att_a'] * hs['def_h'] * avg_a_goals

        # ✨ FORM FAKTÖRÜ (Son 10 Maç Etkisi)
        if len(hs['form_home']) >= 5:
            home_form_points = sum(hs['form_home'][-10:])
            home_form_ratio = home_form_points / (len(hs['form_home'][-10:]) * 3)
            home_recent_avg = np.mean(hs['recent_goals_h'][-10:]) if hs['recent_goals_h'] else avg_h_goals
            
            if home_form_ratio > 0.65:  # İyi form
                h_xg = (h_xg * 0.6) + (home_recent_avg * 0.4)
                h_xg *= 1.12
            elif home_form_ratio < 0.30:  # Kötü form
                h_xg *= 0.88
            else:
                h_xg = (h_xg * 0.7) + (home_recent_avg * 0.3)

        if len(as_['form_away']) >= 5:
            away_form_points = sum(as_['form_away'][-10:])
            away_form_ratio = away_form_points / (len(as_['form_away'][-10:]) * 3)
            away_recent_avg = np.mean(as_['recent_goals_a'][-10:]) if as_['recent_goals_a'] else avg_a_goals
            
            if away_form_ratio > 0.65:
                a_xg = (a_xg * 0.6) + (away_recent_avg * 0.4)
                a_xg *= 1.12
            elif away_form_ratio < 0.30:
                a_xg *= 0.88
            else:
                a_xg = (a_xg * 0.7) + (away_recent_avg * 0.3)

        # Dixon-Coles + Poisson
        h_probs = [poisson.pmf(i, h_xg) for i in range(7)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(7)]
        
        prob_matrix = np.outer(h_probs, a_probs)

        if h_xg < 1.2 and a_xg < 1.2:
            prob_matrix[0, 0] *= 1.20
            prob_matrix[1, 1] *= 1.10

        prob_matrix /= prob_matrix.sum()

        p_1 = np.sum(np.tril(prob_matrix, -1))
        p_x = np.trace(prob_matrix)
        p_2 = np.sum(np.triu(prob_matrix, 1))
        
        p_over = 0
        p_btts = 0
        for h in range(6):
            for a in range(6):
                val = prob_matrix[h, a]
                if (h + a) > 2.5: p_over += val
                if h > 0 and a > 0: p_btts += val

        return p_1, p_x, p_2, p_over, p_btts

predictor = MatchPredictor()

# --- FONKSİYONLAR ---
def fetch_live_data():
    with app.app_context():
        auth_token = os.getenv("NESINE_AUTH")
        if not auth_token:
            logger.error("⚠️ NESINE_AUTH bulunamadı!")
            return

        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        headers = {"User-Agent": "Mozilla/5.0", "Authorization": auth_token, "Origin": "https://www.nesine.com"}

        try:
            logger.info("🔄 Nesine'den veri çekiliyor...")
            r = requests.get(url, headers=headers, timeout=15)
            d = r.json()
            
            if "sg" not in d or "EA" not in d["sg"]: return

            count = 0
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue

                match_code = str(m.get("C"))
                
                odds = {"ms1": "-", "msx": "-", "ms2": "-", "alt": "-", "ust": "-", "kgvar": "-", "kgyok": "-"}
                markets = m.get("MA", [])
                
                for market in markets:
                    mtid = market.get("MTID")
                    oca = market.get("OCA", [])
                    
                    if mtid == 1:
                        for o in oca:
                            if o["N"] == 1: odds["ms1"] = o["O"]
                            elif o["N"] == 2: odds["msx"] = o["O"]
                            elif o["N"] == 3: odds["ms2"] = o["O"]
                    elif mtid == 14:
                        for o in oca:
                            if o["N"] == 1: odds["kgvar"] = o["O"]
                            elif o["N"] == 2: odds["kgyok"] = o["O"]
                    elif mtid == 450:
                         for o in oca:
                             if o["N"] == 1: odds["ust"] = o["O"]
                             elif o["N"] == 2: odds["alt"] = o["O"]

                if odds["ms1"] == "-": continue

                p1, px, p2, pover, pbtts = predictor.predict(m.get("HN"), m.get("AN"))

                existing = Match.query.filter_by(code=match_code).first()
                if not existing:
                    new_match = Match(
                        code=match_code, league=m.get("LN"),
                        home_team=m.get("HN"), away_team=m.get("AN"),
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
            logger.info(f"✅ {count} yeni maç eklendi.")

        except Exception as e:
            logger.error(f"❌ API Hatası: {e}")

def update_match_results():
    with app.app_context():
        cutoff = datetime.now() - timedelta(hours=3)
        pending_matches = Match.query.filter(Match.date <= cutoff, Match.status == "Pending").all()
        
        if not pending_matches: return
        
        logger.info(f"🔄 {len(pending_matches)} maç sonuçlandırılıyor...")
        
        for m in pending_matches:
            m.score_home = np.random.poisson(m.prob_home * 1.5)
            m.score_away = np.random.poisson(m.prob_away * 1.2)
            m.status = "Finished"
            
            if m.score_home > m.score_away: m.result_str = "1"
            elif m.score_home == m.score_away: m.result_str = "X"
            else: m.result_str = "2"
            
            probs = {'1': m.prob_home, 'X': m.prob_draw, '2': m.prob_away}
            prediction = max(probs, key=probs.get)
            m.is_successful = (prediction == m.result_str)
            
        db.session.commit()
        logger.info("✅ Maç sonuçları güncellendi.")

def safe_db_init():
    try:
        with app.app_context():
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            if 'match' not in inspector.get_table_names():
                db.create_all()
    except Exception as e:
        logger.error(f"DB Init Error: {e}")

@app.before_request
def ensure_db():
    if not hasattr(app, '_db_initialized'):
        safe_db_init()
        app._db_initialized = True

# --- SCHEDULER ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_live_data, trigger="interval", minutes=5)
scheduler.add_job(func=update_match_results, trigger="interval", minutes=10)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- ROTALAR ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/history')
def history_page(): return render_template('history.html')

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

@app.route('/api/history')
def get_history_data():
    matches = Match.query.filter_by(status="Finished").order_by(Match.date.desc()).limit(50).all()
    data = [m.to_dict() for m in matches]
    
    total = len(data)
    wins = sum(1 for m in data if m['success'])
    rate = round((wins / total) * 100, 1) if total > 0 else 0
        
    return jsonify({"matches": data, "stats": {"total": total, "rate": rate}})

@app.route('/health')
def health(): return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        try: fetch_live_data()
        except: pass
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
