# Model Optimizasyon Scripti
# Bu script, Whisper modelini optimize edilmiş bir formata dönüştürür
# Kullanım: ./whisper_to_optimized_bin.tr.bash

# --model: Dönüştürülecek model klasörünün yolu
# --output_dir: Dönüştürülmüş modelin kaydedileceği klasör
# --copy_files: Kopyalanacak ek dosyalar
# --quantization: Sayısal hassasiyet seviyesi (float32)

ct2-transformers-converter --model ./PATH_TO_MODEL_TO_CONVERT --output_dir ./CONVERTED_MODEL_OUTPUT_NAME --copy_files preprocessor_config.json --quantization float32 