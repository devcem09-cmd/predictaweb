import os
import json
import logging
import requests
import atexit
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import random # Skor simülasyonu için (Gerçek API bağlayana kadar)

# --- YAPILANDIRMA ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# Veritabanı Ayarları
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(INSTANCE_DIR, 'predictapro.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
logger.info(f"📍 Veritabanı yolu: {DB_PATH}")

# --- GÜNCELLENMİŞ VERİTABANI MODELİ ---
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True)
    league = db.Column(db.String(50))
    home_team = db.Column(db.String(50))
    away_team = db.Column(db.String(50))
    date = db.Column(db.DateTime)
    odds = db.Column(db.Text) 
    
    # Tahminler
    prob_home = db.Column(db.Float, default=0.0)
    prob_draw = db.Column(db.Float, default=0.0)
    prob_away = db.Column(db.Float, default=0.0)
    prob_over_25 = db.Column(db.Float, default=0.0)
    prob_btts = db.Column(db.Float, default=0.0)
    
    # --- YENİ EKLENEN SÜTUNLAR (HISTORY İÇİN) ---
    status = db.Column(db.String(20), default="Pending") # Pending, Finished
    score_home = db.Column(db.Integer, nullable=True)
    score_away = db.Column(db.Integer, nullable=True)
    result_str = db.Column(db.String(10), nullable=True) # "1", "X", "2"
    is_successful = db.Column(db.Boolean, default=False) # Tahmin tuttu mu?

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "league": self.league,
            "home": self.home_team,
            "away": self.away_team,
            "date": self.date.strftime("%d.%m %H:%M"),
            "odds": json.loads(self.odds) if self.odds else {},
            "probs": {
                "1": round(self.prob_home * 100, 1),
                "X": round(self.prob_draw * 100, 1),
                "2": round(self.prob_away * 100, 1),
                "over": round(self.prob_over_25 * 100, 1),
                "btts": round(self.prob_btts * 100, 1)
            },
            # History için ek veriler
            "status": self.status,
            "score": f"{self.score_home} - {self.score_away}" if self.score_home is not None else "-",
            "result": self.result_str,
            "success": self.is_successful
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
            logger.warning("⚠️ CSV Bulunamadı!")
            return
        try:
            required_cols = ['home_team','away_team','home_score','away_score']
            df = pd.read_csv(CSV_PATH,usecols=required_cols,encoding='utf-8',on_bad_lines='skip')
            df['home_score'] = pd.to_numeric(df['home_score'],errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'],errors='coerce').fillna(0).astype('int32')
            self._calculate_stats(df)
            self.team_list = list(self.team_stats.keys())
            del df
            logger.info(f"✅ Veritabanı Hazır. {len(self.team_stats)} takım analiz edildi.")
        except Exception as e:
            logger.error(f"❌ DB Hata: {e}")

    def _calculate_stats(self, df):
        if df.empty: return
        self.avg_home_goals = df['home_score'].mean() or 1.5
        self.avg_away_goals = df['away_score'].mean() or 1.2

        home_stats = df.groupby('home_team')['home_score'].agg(['sum','count'])
        home_conceded = df.groupby('home_team')['away_score'].sum()
        away_stats = df.groupby('away_team')['away_score'].agg(['sum','count'])
        away_conceded = df.groupby('away_team')['home_score'].sum()
        all_teams = set(home_stats.index) | set(away_stats.index)

        for team in all_teams:
            h_scored = home_stats.loc[team,'sum'] if team in home_stats.index else 0
            h_games = home_stats.loc[team,'count'] if team in home_stats.index else 0
            h_allowed = home_conceded.loc[team] if team in home_conceded.index else 0
            a_scored = away_stats.loc[team,'sum'] if team in away_stats.index else 0
            a_games = away_stats.loc[team,'count'] if team in away_stats.index else 0
            a_allowed = away_conceded.loc[team] if team in away_conceded.index else 0

            att_h = (h_scored+2*self.avg_home_goals)/(h_games+2)/self.avg_home_goals
            def_h = (h_allowed+2*self.avg_away_goals)/(h_games+2)/self.avg_away_goals
            att_a = (a_scored+2*self.avg_away_goals)/(a_games+2)/self.avg_away_goals
            def_a = (a_allowed+2*self.avg_home_goals)/(a_games+2)/self.avg_home_goals
            self.team_stats[team] = {'att_h':att_h,'def_h':def_h,'att_a':att_a,'def_a':def_a}

    @lru_cache(maxsize=2048)
    def find_team_cached(self,name):
        if not name or not self.team_list: return None
        clean_name = name.lower().replace('sk','').replace('fk','').replace('fc','').strip()
        match = process.extractOne(clean_name,self.team_list,scorer=fuzz.token_set_ratio,score_cutoff=55)
        return match[0] if match else None

    def predict(self,home,away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        hs = self.team_stats.get(home_db,{'att_h':1.0,'def_h':1.0}) if home_db else {'att_h':1.0,'def_h':1.0}
        as_ = self.team_stats.get(away_db,{'att_a':1.0,'def_a':1.0}) if away_db else {'att_a':1.0,'def_a':1.0}

        h_xg = hs['att_h']*as_['def_a']*self.avg_home_goals
        a_xg = as_['att_a']*hs['def_h']*self.avg_away_goals

        h_probs = [poisson.pmf(i,h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i,a_xg) for i in range(6)]

        p_1=p_x=p_2=p_over=p_btts=0
        for h in range(6):
            for a in range(6):
                p = h_probs[h]*a_probs[a]
                if h>a: p_1+=p
                elif h==a: p_x+=p
                else: p_2+=p
                if h+a>2.5: p_over+=p
                if h>0 and a>0: p_btts+=p

        total_prob = p_1+p_x+p_2
        if total_prob>0:
            p_1/=total_prob
            p_x/=total_prob
            p_2/=total_prob
        return p_1,p_x,p_2,p_over,p_btts

predictor = MatchPredictor()

# --- YENİ: SONUÇ GÜNCELLEME SERVİSİ ---
def update_match_results():
    """
    Maç saati geçen maçları kontrol eder.
    NOT: Gerçek bir 'Canlı Skor API'si olmadığı için, burada 
    maç saati 3 saati geçmiş maçları otomatik 'Finished' yapıyoruz.
    Skorları şimdilik Rastgele atıyorum ki History sayfası boş kalmasın.
    Gerçekte buraya bir skor API'si bağlanmalı.
    """
    with app.app_context():
        # 3 saat önce başlamış ve hala 'Pending' olan maçlar
        cutoff = datetime.now() - timedelta(hours=3)
        pending_matches = Match.query.filter(Match.date <= cutoff, Match.status == "Pending").all()
        
        if not pending_matches: return

        logger.info(f"🔄 {len(pending_matches)} maç sonuçlandırılıyor...")
        
        for m in pending_matches:
            # --- SİMÜLASYON (GERÇEK API YOKSA) ---
            # Burada veritabanındaki olasılıklara göre sanal skor üretiyorum
            # Böylece sistemin "Başarısını" test edebilirsin.
            # Gerçek hayatta burası: m.score_home = api.get_score(m.code)['home']
            
            # Simüle Skor (Bunu production'da silmelisin!)
            m.score_home = np.random.poisson(m.prob_home * 1.5) # XG'ye yakın skor üret
            m.score_away = np.random.poisson(m.prob_away * 1.2)
            m.status = "Finished"
            
            # Sonucu Belirle (1, X, 2)
            if m.score_home > m.score_away: m.result_str = "1"
            elif m.score_home == m.score_away: m.result_str = "X"
            else: m.result_str = "2"
            
            # Başarı Kontrolü (Sistem ne demişti?)
            # En yüksek ihtimali bul
            probs = {'1': m.prob_home, 'X': m.prob_draw, '2': m.prob_away}
            prediction = max(probs, key=probs.get) # Sistemin tahmini (örn: '1')
            
            # Tahmin == Sonuç mu?
            m.is_successful = (prediction == m.result_str)
            
        db.session.commit()
        logger.info("✅ Maç sonuçları güncellendi.")

# --- SCHEDULER AYARI ---
scheduler = BackgroundScheduler()
# Mevcut veri çekme (5 dk)
scheduler.add_job(func=fetch_live_data, trigger="interval", minutes=5)
# YENİ: Sonuç kontrol (10 dk)
scheduler.add_job(func=update_match_results, trigger="interval", minutes=10)
scheduler.start()

# --- YENİ ROTALAR ---

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history')
def get_history_data():
    # Sadece bitmiş maçları getir, tarihe göre tersten sırala
    matches = Match.query.filter_by(status="Finished").order_by(Match.date.desc()).limit(50).all()
    data = [m.to_dict() for m in matches]
    
    # İstatistik Özeti
    total = len(data)
    if total > 0:
        wins = sum(1 for m in data if m['success'])
        success_rate = round((wins / total) * 100, 1)
    else:
        success_rate = 0
        
    return jsonify({
        "matches": data,
        "stats": {
            "total": total,
            "rate": success_rate,
            "status": "Active"
        }
    })

# --- NESINE CANLI VERİ ---
def fetch_live_data():
    with app.app_context():
        auth_token = os.getenv("NESINE_AUTH")
        if not auth_token:
            logger.error("⚠️ NESINE_AUTH bulunamadı!")
            return
        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        headers = {"User-Agent":"Mozilla/5.0","Authorization":auth_token,"Origin":"https://www.nesine.com"}
        try:
            logger.info("🔄 Nesine verisi çekiliyor...")
            r = requests.get(url,headers=headers,timeout=15)
            d = r.json()
            if "sg" not in d or "EA" not in d["sg"]: return
            count=0
            for m in d["sg"]["EA"]:
                if m.get("GT")!=1: continue
                match_code = str(m.get("C"))
                odds={"ms1":"-","msx":"-","ms2":"-","alt":"-","ust":"-","kgvar":"-","kgyok":"-"}
                markets = m.get("MA",[])
                for market in markets:
                    mtid = market.get("MTID")
                    oca = market.get("OCA",[])
                    if mtid==1:
                        for o in oca:
                            if o["N"]==1: odds["ms1"]=o["O"]
                            elif o["N"]==2: odds["msx"]=o["O"]
                            elif o["N"]==3: odds["ms2"]=o["O"]
                    elif mtid==14:
                        for o in oca:
                            if o["N"]==1: odds["kgvar"]=o["O"]
                            elif o["N"]==2: odds["kgyok"]=o["O"]
                    elif mtid==450:
                        for o in oca:
                            if o["N"]==1: odds["ust"]=o["O"]
                            elif o["N"]==2: odds["alt"]=o["O"]
                if odds["ms1"]=="-": continue
                p1,px,p2,pover,pbtts = predictor.predict(m.get("HN"),m.get("AN"))
                existing = Match.query.filter_by(code=match_code).first()
                if not existing:
                    new_match = Match(
                        code=match_code,
                        league=m.get("LN"),
                        home_team=m.get("HN"),
                        away_team=m.get("AN"),
                        date=datetime.strptime(f"{m.get('D')} {m.get('T')}","%d.%m.%Y %H:%M"),
                        odds=odds,
                        prob_home=p1, prob_draw=px, prob_away=p2,
                        prob_over_25=pover, prob_btts=pbtts
                    )
                    db.session.add(new_match)
                    count+=1
                else:
                    existing.odds=odds
            db.session.commit()
            logger.info(f"✅ {count} yeni maç işlendi.")
        except Exception as e:
            logger.error(f"❌ Fetch Hatası: {e}")

# --- Maç Sonucu Güncelle ---
def update_match_result(match_code,home_goals,away_goals):
    with app.app_context():
        match = Match.query.filter_by(code=match_code).first()
        if not match: return
        match.result_home = home_goals
        match.result_away = away_goals
        match.status = "Finished"
        if home_goals>away_goals:
            match.correct_ms = match.prob_home>match.prob_draw and match.prob_home>match.prob_away
        elif home_goals==away_goals:
            match.correct_ms = match.prob_draw>match.prob_home and match.prob_draw>match.prob_away
        else:
            match.correct_ms = match.prob_away>match.prob_home and match.prob_away>match.prob_draw
        match.correct_over = (home_goals+away_goals>2.5 if match.prob_over_25>=0.5 else home_goals+away_goals<=2.5)
        match.correct_btts = (home_goals>0 and away_goals>0 if match.prob_btts>=0.5 else home_goals==0 or away_goals==0)
        db.session.commit()

# --- DB Başlat ---
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
    if not hasattr(app,'_db_initialized'):
        safe_db_init()
        app._db_initialized=True

# --- Scheduler ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_live_data, trigger="interval", minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/history')
def history_page(): return render_template('history.html')

@app.route('/api/matches')
def get_matches():
    sort_by = request.args.get('sort_by','default')
    cutoff = datetime.now() - timedelta(hours=2)
    matches = Match.query.filter(Match.date>=cutoff).all()
    data = [m.to_dict() for m in matches]
    if sort_by=='prob_high': data.sort(key=lambda x:max(x['probs']['1'],x['probs']['X'],x['probs']['2']),reverse=True)
    elif sort_by=='prob_over': data.sort(key=lambda x:x['probs']['over'],reverse=True)
    else: data.sort(key=lambda x:x['date'])
    return jsonify(data)

@app.route('/api/history')
def get_history():
    matches = Match.query.filter(Match.status=="Finished").order_by(Match.date.desc()).all()
    return jsonify([m.to_dict() for m in matches])

@app.route('/api/stats')
def get_stats():
    matches = Match.query.filter(Match.status=="Finished").all()
    total = len(matches)
    correct_ms = sum(1 for m in matches if m.correct_ms)
    correct_over = sum(1 for m in matches if m.correct_over)
    correct_btts = sum(1 for m in matches if m.correct_btts)

    weekly_data={}
    for m in matches:
        week=m.date.strftime("%Y-%W")
        if week not in weekly_data: weekly_data[week]={"ms":0,"total":0}
        weekly_data[week]["ms"] +=1 if m.correct_ms else 0
        weekly_data[week]["total"] +=1
    chart_labels=sorted(weekly_data.keys())
    chart_data=[round(weekly_data[w]["ms"]/weekly_data[w]["total"]*100,2) for w in chart_labels]

    return jsonify({
        "total":total,
        "ms_accuracy": round(correct_ms/total*100,2) if total else 0,
        "over_accuracy": round(correct_over/total*100,2) if total else 0,
        "btts_accuracy": round(correct_btts/total*100,2) if total else 0,
        "weekly_labels": chart_labels,
        "weekly_data": chart_data
    })

@app.route('/health')
def health(): return jsonify({"status":"ok"}),200

# --- APP RUN ---
if __name__=='__main__':
    with app.app_context():
        db.create_all()
        try: fetch_live_data()
        except: pass
    port=int(os.environ.get("PORT",10000))
    app.run(host='0.0.0.0',port=port)

