#!/bin/bash
# Koyeb için başlatma script'i

echo "🔄 Predicta PRO başlatılıyor..."

# Veritabanı dizinini oluştur (gerekirse)
mkdir -p /app/instance

# Veritabanını başlat
echo "📦 Veritabanı başlatılıyor..."
python init_db.py

# Başarı kontrolü
if [ $? -eq 0 ]; then
    echo "✅ Veritabanı hazır!"
else
    echo "⚠️ Veritabanı uyarısı - devam ediliyor..."
fi

# Gunicorn ile uygulamayı başlat
echo "🚀 Uygulama başlatılıyor..."
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
