FROM python:3.9-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Instance klasörünü oluştur
RUN mkdir -p /app/instance

# Start script'ine çalıştırma izni ver
RUN chmod +x start.sh

# Port
EXPOSE 8000

# Başlat
CMD ["./start.sh"]
