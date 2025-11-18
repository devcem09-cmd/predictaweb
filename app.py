from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import logging
from datetime import datetime
import os
import time
from functools import wraps

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nesine headers
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json",
}

NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# Cache için global değişken
cached_matches = []
cache_timestamp = None
CACHE_DURATION = 300  # 5 dakika

# Rate limiting için basit tracker
request_tracker = {}

def rate_limit(max_requests=30, window=60):
    """Rate limiting decorator - dakikada 30 istek"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            
            if client_ip not in request_tracker:
                request_tracker[client_ip] = []
            
            # Eski istekleri temizle
            request_tracker[client_ip] = [
                req_time for req_time in request_tracker[client_ip]
                if now - req_time < window
            ]
            
            if len(request_tracker[client_ip]) >= max_requests:
                logger.warning(f"⚠️ Rate limit aşıldı: {client_ip}")
                return jsonify({
                    'success': False,
                    'error': 'Çok fazla istek. Lütfen biraz bekleyin.',
                    'retry_after': window
                }), 429
            
            request_tracker[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def fetch_nesine_matches(force_refresh=False):
    """Nesine'den maçları çek - GELİŞTİRİLMİŞ VERSİYON"""
    global cached_matches, cache_timestamp
    
    # Cache kontrolü
    if not force_refresh and cached_matches and cache_timestamp:
        age = (datetime.now() - cache_timestamp).total_seconds()
        if age < CACHE_DURATION:
            logger.info(f"📦 Cache kullanılıyor (yaş: {age:.0f}s)")
            return {
                'matches': cached_matches,
                'from_cache': True,
                'cache_age': age
            }
    
    try:
        logger.info("🔄 Nesine API'den veriler çekiliyor...")
        start_time = time.time()
        
        response = requests.get(
            NESINE_URL, 
            headers=NESINE_HEADERS, 
            timeout=15,
            verify=True
        )
        response.raise_for_status()
        
        fetch_time = time.time() - start_time
        logger.info(f"⚡ Nesine yanıt süresi: {fetch_time:.2f}s")
        
        data = response.json()
        
        matches = []
        stats = {
            "total_processed": 0,
            "with_ms": 0,
            "with_ou": 0,
            "with_btts": 0,
            "complete_odds": 0,
            "skipped": 0
        }
        
        sports_data = data.get("sg", {})
        if not sports_data:
            logger.warning("⚠️ Nesine'den spor verisi gelmedi")
            return {
                'matches': cached_matches if cached_matches else [],
                'from_cache': bool(cached_matches),
                'cache_age': None
            }
        
        football_matches = sports_data.get("EA", [])
        logger.info(f"🔍 {len(football_matches)} futbol maçı bulundu")
        
        for m in football_matches:
            if m.get("GT") != 1:  # Sadece futbol
                stats["skipped"] += 1
                continue
            
            stats["total_processed"] += 1
            
            # ✅ GELİŞTİRİLMİŞ: Daha fazla bilgi
            match_info = {
                "match_id": str(m.get("C", "")),
                "home_team": m.get("HN", ""),
                "away_team": m.get("AN", ""),
                "home": m.get("HN", ""),  # HTML uyumluluğu için
                "away": m.get("AN", ""),  # HTML uyumluluğu için
                "league_code": m.get("LC", ""),
                "league_name": m.get("LN", str(m.get("LID", ""))),
                "date": f"{m.get('D', '')}T{m.get('T', '')}:00",
                "tarih": m.get('D', ''),  # Nesine formatı
                "saat": m.get('T', ''),   # Nesine formatı
                "is_live": m.get("L", False),
                "odds": {},
                "oranlar": []  # ✅ HTML'inizin extractOddsFromNesine() için
            }
            
            has_ms = False
            has_ou = False
            has_btts = False
            
            # Oranları hem standart hem Nesine formatında kaydet
            for bahis in m.get("MA", []):
                bahis_tipi = bahis.get("MTID")
                oranlar = bahis.get("OCA", [])
                
                # ✅ Nesine formatını koru (HTML için)
                match_info["oranlar"].append({
                    "bahis_tipi": bahis_tipi,
                    "oranlar": oranlar
                })
                
                # Maç Sonucu (1, X, 2)
                if bahis_tipi == 1 and len(oranlar) >= 3:
                    try:
                        match_info["odds"]["1"] = float(oranlar[0].get("O", 2.0))
                        match_info["odds"]["X"] = float(oranlar[1].get("O", 3.2))
                        match_info["odds"]["2"] = float(oranlar[2].get("O", 3.5))
                        has_ms = True
                    except (ValueError, TypeError, KeyError):
                        logger.debug(f"⚠️ MS oranları işlenemedi: {match_info['home_team']}")
                
                # Alt/Üst 2.5
                elif bahis_tipi == 450 and len(oranlar) >= 2:
                    try:
                        match_info["odds"]["Over/Under +2.5"] = {
                            "Over +2.5": float(oranlar[0].get("O", 1.9)),
                            "Under +2.5": float(oranlar[1].get("O", 1.9))
                        }
                        has_ou = True
                    except (ValueError, TypeError, KeyError):
                        logger.debug(f"⚠️ OU oranları işlenemedi")
                
                # Karşılıklı Gol (BTTS)
                elif bahis_tipi == 38 and len(oranlar) >= 2:
                    try:
                        match_info["odds"]["Both Teams To Score"] = {
                            "Yes": float(oranlar[0].get("O", 1.85)),
                            "No": float(oranlar[1].get("O", 1.95))
                        }
                        has_btts = True
                    except (ValueError, TypeError, KeyError):
                        logger.debug(f"⚠️ BTTS oranları işlenemedi")
            
            # Sadece maç sonucu oranı olan maçları ekle
            if has_ms:
                stats["with_ms"] += 1
                if has_ou:
                    stats["with_ou"] += 1
                if has_btts:
                    stats["with_btts"] += 1
                if has_ms and has_ou and has_btts:
                    stats["complete_odds"] += 1
                
                matches.append(match_info)
        
        # Cache'i güncelle
        cached_matches = matches
        cache_timestamp = datetime.now()
        
        process_time = time.time() - start_time
        logger.info(f"✅ Nesine'den {len(matches)} maç çekildi ({process_time:.2f}s)")
        logger.info(f"📊 İstatistikler: MS={stats['with_ms']}, OU={stats['with_ou']}, BTTS={stats['with_btts']}, TAM={stats['complete_odds']}")
        
        return {
            'matches': matches,
            'from_cache': False,
            'cache_age': 0,
            'stats': stats,
            'fetch_time': fetch_time,
            'process_time': process_time
        }
        
    except requests.Timeout:
        logger.error("⏱️ Nesine API timeout!")
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': 'timeout'
        }
    except requests.RequestException as e:
        logger.error(f"❌ Nesine API bağlantı hatası: {str(e)}")
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {str(e)}", exc_info=True)
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': str(e)
        }

@app.route('/', methods=['GET'])
def index():
    """API Ana Sayfa"""
    return jsonify({
        "name": "PredictaAI API",
        "version": "2.1 - Hybrid Edition",
        "status": "online",
        "description": "Nesine.com canlı oran entegrasyonu + HTML uyumlu format",
        "endpoints": {
            "health": "/health",
            "matches": "/api/matches/upcoming",
            "match_detail": "/api/matches/<match_id>",
            "clear_cache": "/api/cache/clear",
            "cache_status": "/api/cache/status"
        },
        "features": [
            "Nesine.com canlı oran entegrasyonu",
            "Dual format support (Standard + Nesine)",
            "5 dakikalık akıllı cache sistemi",
            "Rate limiting koruması",
            "HTML PredictaAI ile %100 uyumlu"
        ],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    cache_age = None
    if cache_timestamp:
        cache_age = int((datetime.now() - cache_timestamp).total_seconds())
    
    return jsonify({
        "status": "online",
        "service": "PredictaAI API - Hybrid",
        "timestamp": datetime.now().isoformat(),
        "cache": {
            "age_seconds": cache_age,
            "cached_matches": len(cached_matches),
            "is_fresh": cache_age < CACHE_DURATION if cache_age else False
        },
        "format": "dual_support",
        "uptime": "OK"
    })

@app.route('/api/matches/upcoming', methods=['GET'])
@rate_limit(max_requests=30, window=60)
def get_upcoming_matches():
    """Yaklaşan maçları getir - GELİŞTİRİLMİŞ"""
    try:
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        result = fetch_nesine_matches(force_refresh=force_refresh)
        matches = result['matches']
        
        # İstatistikler
        complete_stats = {
            "total": len(matches),
            "with_all_odds": sum(1 for m in matches if 
                "Over/Under +2.5" in m["odds"] and 
                "Both Teams To Score" in m["odds"]
            ),
            "with_ou": sum(1 for m in matches if "Over/Under +2.5" in m["odds"]),
            "with_btts": sum(1 for m in matches if "Both Teams To Score" in m["odds"]),
            "with_ms_only": sum(1 for m in matches if 
                len(m["odds"]) == 3 and "1" in m["odds"]
            )
        }
        
        response_data = {
            "success": True,
            "matches": matches,
            "count": len(matches),
            "stats": complete_stats,
            "source": "nesine.com",
            "format": "hybrid (standard + nesine)",
            "timestamp": datetime.now().isoformat(),
            "from_cache": result.get('from_cache', False),
            "cache_age_seconds": result.get('cache_age')
        }
        
        # Performans bilgileri
        if 'fetch_time' in result:
            response_data['performance'] = {
                'fetch_time': result['fetch_time'],
                'process_time': result['process_time']
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ API hatası: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "matches": [],
            "count": 0
        }), 500

@app.route('/api/matches/<match_id>', methods=['GET'])
@rate_limit(max_requests=60, window=60)
def get_match_details(match_id):
    """Belirli bir maçın detaylarını getir"""
    try:
        result = fetch_nesine_matches()
        matches = result['matches']
        
        match_id_str = str(match_id)
        for match in matches:
            if str(match["match_id"]) == match_id_str:
                return jsonify({
                    "success": True,
                    "match": match,
                    "format": "hybrid",
                    "timestamp": datetime.now().isoformat()
                })
        
        return jsonify({
            "success": False,
            "error": "Maç bulunamadı",
            "match_id": match_id
        }), 404
        
    except Exception as e:
        logger.error(f"❌ Match details hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Cache'i temizle"""
    global cached_matches, cache_timestamp
    
    old_count = len(cached_matches)
    cached_matches = []
    cache_timestamp = None
    
    logger.info(f"🗑️ Cache temizlendi ({old_count} maç)")
    
    return jsonify({
        "success": True,
        "message": "Cache temizlendi",
        "cleared_matches": old_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/cache/status', methods=['GET'])
def cache_status():
    """Cache durumu"""
    cache_age = None
    is_fresh = False
    
    if cache_timestamp:
        cache_age = int((datetime.now() - cache_timestamp).total_seconds())
        is_fresh = cache_age < CACHE_DURATION
    
    return jsonify({
        "cache_enabled": True,
        "cache_duration_seconds": CACHE_DURATION,
        "current_cache": {
            "match_count": len(cached_matches),
            "age_seconds": cache_age,
            "is_fresh": is_fresh,
            "created_at": cache_timestamp.isoformat() if cache_timestamp else None
        },
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint bulunamadı",
        "available_endpoints": ["/", "/health", "/api/matches/upcoming"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Hatası: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Sunucu hatası",
        "message": "Lütfen daha sonra tekrar deneyin"
    }), 500

# Render için
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info("=" * 80)
    logger.info("🚀 PredictaAI Hybrid API Başlatılıyor")
    logger.info(f"📡 Port: {port}")
    logger.info(f"📦 Cache: {CACHE_DURATION}s")
    logger.info("🎯 Format: Dual Support (Standard + Nesine)")
    logger.info("✅ HTML PredictaAI ile tam uyumlu")
    logger.info("=" * 80)
    app.run(debug=False, host='0.0.0.0', port=port)