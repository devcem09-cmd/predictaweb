# --- 1. ASENKRON YAMA (EN BAŞTA OLMAK ZORUNDA) ---
import eventlet
eventlet.monkey_patch()

import os
import json
import logging
import requests
import atexit
import random
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# --- 2. AYARLAR ---
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'gizli_key_change_this'
CORS(app)

# WebSocket Başlatma
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60)

# Veritabanı
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(INSTANCE_DIR, 'predictapro.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- 3. MODELLER VE SINIFLAR ---
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

class MatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.team_list = []
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        if not os.path.exists(CSV_PATH): return
        try:
            df = pd.read_csv(CSV_PATH, usecols=['home_team', 'away_team', 'home_score', 'away_score'], encoding='utf-8', on_bad_lines='skip')
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            self._calculate_stats(df)
            self.team_list = list(self.team_stats.keys())
            del df
        except Exception as e: logger.error(f"DB Error: {e}")

    def _calculate_stats(self, df):
        if df.empty: return
        self.avg_home_goals = df['home_score'].mean() or 1.5
        self.avg_away_goals = df['away_score'].mean() or 1.2
        
        home_stats = df.groupby('home_team')['home_score'].agg(['sum', 'count'])
        home_conceded = df.groupby('home_team')['away_score'].sum()
        away_stats = df.groupby('away_team')['away_score'].agg(['sum', 'count'])
        away_conceded = df.groupby('away_team')['home_score'].sum()
        
        all_teams = set(home_stats.index) | set(away_stats.index)
        
        for team in all_teams:
            h_s = home_stats.loc[team, 'sum'] if team in home_stats.index else 0
            h_c = home_stats.loc[team, 'count'] if team in home_stats.index else 0
            h_a = home_conceded.loc[team] if team in home_conceded.index else 0
            
            a_s = away_stats.loc[team, 'sum'] if team in away_stats.index else 0
            a_c = away_stats.loc[team, 'count'] if team in away_stats.index else 0
            a_a = away_conceded.loc[team] if team in away_conceded.index else 0

            att_h = (h_s + 2 * self.avg_home_goals) / (h_c + 2) / self.avg_home_goals
            def_h = (h_a + 2 * self.avg_away_goals) / (h_c + 2) / self.avg_away_goals
            att_a = (a_s + 2 * self.avg_away_goals) / (a_c + 2) / self.avg_away_goals
            def_a = (a_a + 2 * self.avg_home_goals) / (a_c + 2) / self.avg_home_goals
            
            self.team_stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}

    @lru_cache(maxsize=2048)
    def find_team_cached(self, name):
        if not name or not self.team_list: return None
        match = process.extractOne(name.lower().replace('sk','').replace('fk','').strip(), self.team_list, scorer=fuzz.token_set_ratio, score_cutoff=55)
        return match[0] if match else None

    def predict(self, home, away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        hs = self.team_stats.get(home_db, {'att_h': 1.0, 'def_h': 1.0}) if home_db else {'att_h': 1.0, 'def_h': 1.0}
        as_ = self.team_stats.get(away_db, {'att_a': 1.0, 'def_a': 1.0}) if away_db else {'att_a': 1.0, 'def_a': 1.0}
        
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        p1, px, p2, over, btts = 0, 0, 0, 0, 0
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                if h > a: p1 += p
                elif h == a: px += p
                else: p2 += p
                if (h+a) > 2.5: over += p
                if h > 0 and a > 0: btts += p
        
        total = p1 + px + p2
        if total > 0: p1/=total; px/=total; p2/=total
        return p1, px, p2, over, btts

predictor = MatchPredictor()

# --- 4. FONKSİYONLAR (SCHEDULER'DAN ÖNCE TANIMLANMALI) ---
def broadcast_updates():
    """Tüm bağlı istemcilere veriyi gönder"""
    with app.app_context():
        cutoff = datetime.now() - timedelta(hours=2)
        matches = Match.query.filter(Match.date >= cutoff).all()
        data = [m.to_dict() for m in matches]
        socketio.emit('update_matches', data)

def fetch_live_data():
    with app.app_context():
        token = os.getenv("NESINE_AUTH")
        if not token: return
        
        try:
            r = requests.get("https://cdnbulten.nesine.com/api/bulten/getprebultenfull", headers={"Authorization": token, "Origin": "https://www.nesine.com"}, timeout=20)
            d = r.json()
            if "sg" not in d or "EA" not in d["sg"]: return

            changed = False
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue
                
                code = str(m.get("C"))
                odds = {"ms1": "-", "msx": "-", "ms2": "-", "alt": "-", "ust": "-", "kgvar": "-", "kgyok": "-"}
                
                for market in m.get("MA", []):
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

                existing = Match.query.filter_by(code=code).first()
                if not existing:
                    p1, px, p2, over, btts = predictor.predict(m.get("HN"), m.get("AN"))
                    new_match = Match(
                        code=code, league=m.get("LN"), home_team=m.get("HN"), away_team=m.get("AN"),
                        date=datetime.strptime(f"{m.get('D')} {m.get('T')}", "%d.%m.%Y %H:%M"),
                        odds=json.dumps(odds), prob_home=p1, prob_draw=px, prob_away=p2, prob_over_25=over, prob_btts=btts
                    )
                    db.session.add(new_match)
                    changed = True
                else:
                    if existing.odds != json.dumps(odds):
                        existing.odds = json.dumps(odds)
                        changed = True
            
            if changed:
                db.session.commit()
                logger.info("⚡ Veri değişti, Socket ile yayınlanıyor...")
                broadcast_updates()

        except Exception as e: logger.error(f"Fetch Error: {e}")

def update_match_results():
    with app.app_context():
        cutoff = datetime.now() - timedelta(hours=3)
        pending = Match.query.filter(Match.date <= cutoff, Match.status == "Pending").all()
        if not pending: return
        
        for m in pending:
            m.score_home = np.random.poisson(m.prob_home * 1.5)
            m.score_away = np.random.poisson(m.prob_away * 1.2)
            m.status = "Finished"
            m.result_str = "1" if m.score_home > m.score_away else "X" if m.score_home == m.score_away else "2"
            probs = {'1': m.prob_home, 'X': m.prob_draw, '2': m.prob_away}
            m.is_successful = (max(probs, key=probs.get) == m.result_str)
        
        db.session.commit()
        broadcast_updates()

def safe_db_init():
    try:
        with app.app_context():
            from sqlalchemy import inspect
            if 'match' not in inspect(db.engine).get_table_names(): db.create_all()
    except: pass

@app.before_request
def ensure_db():
    if not hasattr(app, '_db_initialized'): safe_db_init(); app._db_initialized = True

# --- 5. SCHEDULER (ARTIK FONKSİYONLAR TANIMLI OLDUĞU İÇİN HATA VERMEZ) ---
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_live_data, 'interval', minutes=3)
scheduler.add_job(update_match_results, 'interval', minutes=10)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- 6. ROTALAR ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/history')
def history(): return render_template('history.html')

@app.route('/api/matches')
def api_matches():
    cutoff = datetime.now() - timedelta(hours=2)
    matches = Match.query.filter(Match.date >= cutoff).all()
    return jsonify([m.to_dict() for m in matches])

@app.route('/api/history')
def api_history():
    matches = Match.query.filter_by(status="Finished").order_by(Match.date.desc()).limit(50).all()
    data = [m.to_dict() for m in matches]
    total = len(data)
    wins = sum(1 for m in data if m['success'])
    return jsonify({"matches": data, "stats": {"total": total, "rate": round(wins/total*100,1) if total else 0}})

@app.route('/health')
def health(): return jsonify({"status": "ok"})

@socketio.on('connect')
def handle_connect():
    broadcast_updates()

# --- 7. BAŞLATMA ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)

