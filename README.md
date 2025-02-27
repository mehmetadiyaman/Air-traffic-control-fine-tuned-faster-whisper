# ATC Verileri Üzerinde Whisper Modelinin İnce Ayarı

🚀 **Havacılık İletişimi İçin Gelişmiş Transkripsiyon**

Bu proje, OpenAI'nin Whisper modelini **Hava Trafik Kontrolü (ATC) verileri** ile ince ayar yaparak, havacılık iletişimlerinin transkripsiyon doğruluğunu artırmayı amaçlamaktadır. Yapılan ince ayar sayesinde, **Kelime Hata Oranı (WER) %84 oranında azaltılmıştır** ve model, aksan farklılıklarını ve belirsiz ifadeleri daha iyi işleyebilir hale gelmiştir.

💡 **Projenin İçeriği**:
- ATC veri setlerinin hazırlanması ve işlenmesi
- Whisper modelinin ince ayar ile eğitilmesi
- Modelin değerlendirilmesi ve karşılaştırmalar
- Optimizasyon ve dağıtım süreçleri

Hugging Face üzerinden ince ayar yapılmış modele ve veri setine erişebilirsiniz:
- 🎯 **[ATC için İnce Ayar Yapılmış Whisper Medium EN (Daha Hızlı Whisper)](https://huggingface.co/mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper)**
- 📂 **[ATC Veri Seti](https://huggingface.co/datasets/mehmedadymn/air-traffic-dataset)**

---
## 📌 Depo Yapısı

📂 **Özel Veri Seti İşleme**
- `veri_seti_olustur_ve_yukle.py` → Eğitim ve test için ATC veri setlerini hazırlar.

📂 **Değerlendirme Scriptleri**
- `ince_ayarli_modeli_degerlendir.py` → İnce ayarlı modelin performansını ölçer.
- `onceden_egitilmis_modeli_degerlendir.py` → Önceden eğitilmiş modelin sonuçlarıyla kıyaslama yapar.

📂 **Eğitim Scriptleri**
- `egitim.py` → Whisper modelini ATC verileriyle eğitir ve optimize eder.

📂 **Araçlar**
- `modeli_cikarim_icin_disa_aktar.py` → Modeli çıkarım için optimize eder.
- `yerel_modelleri_huggingface_yukle.py` → Hugging Face'e model yükler.
- `whisper_optimize_et.bash` → Whisper modelini optimize edilmiş formata dönüştürür.

📄 **requirements.txt** → Proje için gerekli bağımlılıkları içerir.

---
## 🚀 Nasıl Çalıştırılır?

1️⃣ **Bağımlılıkları Yükleyin**
```bash
pip install -r requirements.txt
```

2️⃣ **Veri Setini Hazırlayın**
```bash
python veri_seti_olustur_ve_yukle.py
```

3️⃣ **Modeli Eğitin**
```bash
python egitim.py
```

4️⃣ **Modeli Değerlendirin**
```bash
python ince_ayarli_modeli_degerlendir.py
```

5️⃣ **Modeli Hugging Face'e Yükleyin**
```bash
python yerel_modelleri_huggingface_yukle.py
```

---
## 🎯 Model Kullanımı

İnce ayarlı model Hugging Face üzerinde yayınlanmıştır. Modeli yerel olarak indirip kullanabilirsiniz:

📥 **[ATC için İnce Ayar Yapılmış Whisper Medium EN (Daha Hızlı Whisper)](https://huggingface.co/mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper)**

---
## 📜 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.

