# 🎯 PredictaAI - Football Match Analysis Frontend

Modern ve gelişmiş futbol maç analiz arayüzü.

## ✨ Özellikler

- 🔮 **Akıllı Tahmin Motoru** - Oranlar, form, H2H ve ev/deplasman analizi
- 📊 **Canlı Nesine Oranları** - Alt/Üst 2.5, BTTS ve Maç Sonucu
- 🎨 **Modern Arayüz** - Koyu tema, animasyonlu kartlar
- 📱 **Responsive Tasarım** - Mobil uyumlu
- ⚡ **Gerçek Zamanlı Veri** - 5 dakikalık cache sistemi
- 🔍 **Gelişmiş Filtreler** - Tarih, güven, kalite bazlı sıralama

## 🚀 Kullanım

### Canlı Demo
👉 [PredictaAI Web App](https://predicta-web.pages.dev) *(Cloudflare Pages)*

### Local Kullanım
```bash
# Basitçe index.html'i tarayıcıda açın
open index.html
```

## 🔌 API Entegrasyonu

Bu frontend şu API'yi kullanır:
- **Flask API:** https://predicta-api.onrender.com
- **Endpoint:** `/api/matches/upcoming`

### API Değiştirmek için:
`index.html` içinde `API_BASE_URL` değişkenini güncelleyin:
```javascript
const API_BASE_URL = 'https://YOUR-API-URL.com';
```

## 🎨 Özelleştirme

### Renk Teması
CSS değişkenlerini düzenleyin:
```css
:root {
    --primary: #ffc107;      /* Ana renk */
    --secondary: #667eea;    /* İkincil renk */
    --dark: #0a0e27;         /* Arka plan */
}
```

### Cache Süresi
```javascript
const CACHE_DURATION = 300;  // 5 dakika (saniye)
```

## 📊 Analiz Sistemi

### Güven Skoru Hesaplama
- **%70+** → Yüksek güven (Yeşil kenarlık)
- **%60-69** → Orta güven (Sarı kenarlık)
- **<%60** → Düşük güven (Varsayılan)

### Veri Kaynakları
1. **Oran Analizi** (%40) - Bahis oranlarından olasılık
2. **Form Analizi** (%25) - Son 5 maç performansı
3. **H2H Analizi** (%20) - Geçmiş karşılaşmalar
4. **Ev/Deplasman** (%15) - Saha avantajı

## 🛠️ Teknik Detaylar

### Tarayıcı Desteği
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Kullanılan Teknolojiler
- HTML5
- CSS3 (Grid, Flexbox, Animations)
- Vanilla JavaScript (ES6+)
- Fetch API

## 📝 Lisans

MIT License - Ticari ve kişisel projelerde kullanılabilir.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit atın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing`)
5. Pull Request açın

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**⚽ Made with ❤️ for football fans**