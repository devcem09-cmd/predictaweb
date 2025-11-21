import os
import json
import logging
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
from rapidfuzz import process, fuzz
import pandas as pd
import atexit

# --- AYARLAR & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictapro.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- VERİTABANI MODELLERİ ---
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True) # Maç Kodu (Benzersizlik için)
    league = db.Column(db.String(50))
    home_team = db.Column(db.String(50))
    away_team = db.Column(db.String(50))
    date = db.Column(db.DateTime)
    
    # Oranlar (JSON olarak saklayacağız)
    odds = db.Column(db.Text) 
    
    # İstatistiksel Tahminler
    prob_home = db.Column(db.Float)
    prob_draw = db.Column(db.Float)
    prob_away = db.Column(db.Float)
    prob_over_25 = db.Column(db.Float)
    prob_btts = db.Column(db.Float)
    
    # Sonuç Takibi
    status = db.Column(db.String(20), default="Pending") # Pending, Finished
    score_home = db.Column(db.Integer, nullable=True)
    score_away = db.Column(db.Integer, nullable=True)
    is_successful = db.Column(db.Boolean, nullable=True) # Ana tahmin tuttu mu?

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "league": self.league,
            "home": self.home_team,
            "away": self.away_team,
            "date": self.date.strftime("%Y-%m-%d %H:%M"),
            "odds": json.loads(self.odds),
            "probs": {
                "1": round(self.prob_home * 100, 1),
                "X": round(self.prob_draw * 100, 1),
                "2": round(self.prob_away * 100, 1),
                "over": round(self.prob_over_25 * 100, 1),
                "btts": round(self.prob_btts * 100, 1)
            },
            "status": self.status,
            "score": f"{self.score_home}-{self.score_away}" if self.score_home is not None else "-"
        }

# --- TAHMİN MOTORU (PREDICTOR ENGINE) ---
class PredictorEngine:
    def __init__(self):
        self.stats = {}
        self.avg_goals = {'home': 1.5, 'away': 1.2}
        # CSV Yükleme simülasyonu - Gerçek projede burası veritabanından okunmalı
        self.load_mock_data()

    def load_mock_data(self):
        # NOT: Burayı senin CSV okuma kodunla değiştirebilirsin.
        # Şimdilik hata vermemesi için boş geçiyorum.
        pass

    def predict(self, home, away):
        # Basit Poisson Modeli (Geliştirilebilir)
        # Örnekleme amacıyla rastgelelik yerine sabit mantık kullanalım
        # Gerçek veride burası takımın gücüne göre hesaplanmalı.
        
        # Simülasyon: İsim uzunluğuna göre güç belirle (Sadece kod çalışsın diye)
        # SENİN CSV KODUNU BURAYA ENTEGRE EDECEKSİN.
        h_str = len(home) * 0.15
        a_str = len(away) * 0.12
        
        h_xg = max(0.8, h_str)
        a_xg = max(0.5, a_str)

        # Poisson
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        p_1, p_x, p_2, p_over, p_btts = 0, 0, 0, 0, 0
        
        for h in range(6):
            for a in range(6):
                prob = h_probs[h] * a_probs[a]
                if h > a: p_1 += prob
                elif h == a: p_x += prob
                else: p_2 += prob
                if (h+a) > 2.5: p_over += prob
                if h > 0 and a > 0: p_btts += prob

        return p_1, p_x, p_2, p_over, p_btts

predictor = PredictorEngine()

# --- NESİNE ENTEGRASYONU (WORKER) ---
def fetch_and_update_data():
    """Nesine'den veri çeker, tahmin yapar ve veritabanına yazar."""
    with app.app_context():
        logger.info("🔄 Veri güncelleme başladı...")
        try:
            # NOT: Token'ı .env dosyasından almalısın!
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
                "Origin": "https://www.nesine.com"
            }
            url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()
            
            count = 0
            if "sg" in data and "EA" in data["sg"]:
                for m in data["sg"]["EA"]:
                    if m.get("GT") != 1: continue # Sadece Futbol
                    
                    match_code = str(m.get("C")) # Maç Kodu
                    
                    # Zaten kayıtlı ve bitmişse geç
                    existing = Match.query.filter_by(code=match_code).first()
                    if existing and existing.status == "Finished": continue

                    # Oranları Parse Et
                    odds = {"ms1": "-", "msx": "-", "ms2": "-", "kgvar": "-", "kgyok": "-", "alt": "-", "ust": "-"}
                    markets = m.get("MA", [])
                    
                    for market in markets:
                        mtid = market.get("MTID")
                        oca = market.get("OCA", [])
                        if mtid == 1: # MS
                            for o in oca:
                                if o["N"] == 1: odds["ms1"] = o["O"]
                                elif o["N"] == 2: odds["msx"] = o["O"]
                                elif o["N"] == 3: odds["ms2"] = o["O"]
                        elif mtid == 450: # 2.5 Alt/Üst
                             for o in oca:
                                 if o["N"] == 1: odds["ust"] = o["O"]
                                 if o["N"] == 2: odds["alt"] = o["O"]
                        elif mtid == 17: # KG Var/Yok (ID değişebilir, kontrol et)
                             for o in oca:
                                 if o["N"] == 1: odds["kgvar"] = o["O"]
                                 if o["N"] == 2: odds["kgyok"] = o["O"]

                    if odds["ms1"] == "-": continue # Oranı olmayan maçı alma

                    # Tahmin Yap
                    p1, px, p2, pover, pbtts = predictor.predict(m.get("HN"), m.get("AN"))

                    # Veritabanı Kaydı
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
                        # Sadece oranları güncelle (oranlar değişebilir)
                        existing.odds = json.dumps(odds)
            
            db.session.commit()
            logger.info(f"✅ Güncelleme tamamlandı. {count} yeni maç eklendi.")

        except Exception as e:
            logger.error(f"❌ Hata: {e}")

# --- SCHEDULER ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_and_update_data, trigger="interval", minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/matches')
def api_matches():
    # Filtre Parametreleri
    min_prob = float(request.args.get('min_prob', 0))
    sort_by = request.args.get('sort_by', 'date')
    
    # Henüz oynanmamış veya sonucu girilmemiş maçlar
    query = Match.query.filter(Match.date >= datetime.now() - timedelta(hours=2))
    
    matches = query.all()
    data = [m.to_dict() for m in matches]
    
    # Python tarafında filtreleme (SQLAlchemy ile de yapılabilir ama hızlı çözüm)
    filtered = []
    for m in data:
        # En yüksek ihtimali bul
        max_p = max(m['probs'].values())
        if max_p >= min_prob:
            filtered.append(m)
            
    # Sıralama
    if sort_by == 'prob_home':
        filtered.sort(key=lambda x: x['probs']['1'], reverse=True)
    elif sort_by == 'prob_over':
        filtered.sort(key=lambda x: x['probs']['over'], reverse=True)
    else: # Date
        filtered.sort(key=lambda x: x['date'])

    return jsonify(filtered)

@app.route('/api/stats')
def api_stats():
    # Basit bir istatistik endpointi
    total = Match.query.count()
    finished = Match.query.filter_by(status="Finished").count()
    successful = Match.query.filter_by(is_successful=True).count()
    
    return jsonify({
        "total_tracked": total,
        "finished": finished,
        "successful": successful,
        "success_rate": round((successful / finished * 100), 2) if finished > 0 else 0
    })

# İlk kurulum için DB oluştur
with app.app_context():
    db.create_all()
    # İlk veriyi çek (Sunucu başlarken)
    fetch_and_update_data()

# ... (kodun geri kalanı aynı) ...

if __name__ == '__main__':
    # Koyeb dinamik port atar, onu yakalamalıyız
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
