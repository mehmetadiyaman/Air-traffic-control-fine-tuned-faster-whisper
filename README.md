# ✈️ ATC Verileri Üzerinde Whisper Modelinin İnce Ayarı

## 🚀 Havacılık İletişimi İçin Gelişmiş Transkripsiyon

Bu proje, **OpenAI'nin Whisper modelini** özel olarak **Hava Trafik Kontrolü (ATC) verileri** ile ince ayar yaparak, havacılık iletişimlerinde transkripsiyon doğruluğunu artırmayı hedeflemektedir. Yapılan ince ayar sayesinde:

✅ **Kelime Hata Oranı (WER) %84 oranında azaltılmıştır.**  
✅ **Aksan farklılıkları ve belirsiz ifadeler daha iyi işlenmektedir.**  
✅ **Daha hızlı ve optimize edilmiş model formatı sunulmaktadır.**

### 🔍 Hugging Face Üzerinde İnce Ayarlı Model ve Veri Seti
- 🎯 **[ATC için İnce Ayar Yapılmış Whisper Medium EN (Daha Hızlı Whisper)](https://huggingface.co/mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper)**
- 📂 **[ATC Veri Seti](https://huggingface.co/datasets/mehmedadymn/air-traffic-dataset)**

---

## 📌 Depo Yapısı

📂 **Veri Seti İşleme**  
- `veri_seti_olustur_ve_yukle.py` → ATC veri setlerini temizler, birleştirir ve hazırlar.

📂 **Değerlendirme Scriptleri**  
- `ince_ayarli_modeli_degerlendir.py` → İnce ayarlı modelin performansını ölçer.
- `onceden_egitilmis_modeli_degerlendir.py` → Önceden eğitilmiş modelle karşılaştırma yapar.

📂 **Eğitim Scriptleri**  
- `egitim.py` → Whisper modelini ATC verileriyle eğitir ve optimize eder.

📂 **Araçlar**  
- `modeli_cikarim_icin_disa_aktar.py` → Modeli çıkarım için optimize eder.
- `yerel_modelleri_huggingface_yukle.py` → Modeli Hugging Face'e yükler.
- `whisper_optimize_et.bash` → Whisper modelini optimize edilmiş formata dönüştürür.

📄 **requirements.txt** → Projenin bağımlılıklarını içerir.

---

## 🛠️ Kurulum ve Kullanım

### 1️⃣ Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2️⃣ Veri Setini Hazırlayın
```bash
python veri_seti_olustur_ve_yukle.py
```

### 3️⃣ Modeli Eğitin
```bash
python egitim.py
```

### 4️⃣ Modeli Değerlendirin
```bash
python ince_ayarli_modeli_degerlendir.py
```

### 5️⃣ Modeli Hugging Face'e Yükleyin
```bash
python yerel_modelleri_huggingface_yukle.py
```

---

## 🎯 Model Kullanımı

Hugging Face üzerinde yayınlanan ince ayarlı modeli yerel olarak indirerek kullanabilirsiniz:

📥 **[ATC için İnce Ayar Yapılmış Whisper Medium EN (Daha Hızlı Whisper)](https://huggingface.co/mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper)**

---

## 📜 Lisans

Bu proje **MIT Lisansı** altında sunulmuştur. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.

