# İnce Ayarlı Whisper Modelinin Değerlendirme Scripti
# Bu script, ince ayar yapılmış Whisper modelinin performansını test veri seti üzerinde değerlendirir

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import soundfile as sf
import os
import jiwer
from datasets import load_dataset
import pandas as pd
import re

# Test veri setini yükle
dataset = load_dataset("mehmedadymn/air-traffic-dataset")

def generate_transcription_and_process_results(model_name, log_file_path="whisper-medium-tr-degerlendirme-ham-veriler.txt", seed=None):
    """
    İnce ayarlı model ile transkripsiyon oluşturma ve sonuçları işleme fonksiyonu
    
    Parametreler:
    model_name: Kullanılacak modelin adı
    log_file_path: Sonuçların kaydedileceği geçici dosya yolu
    seed: Rastgelelik için tohum değeri
    """
    if seed is not None:
        np.random.seed(seed)

    # Model ve işlemciyi yükle
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_file_path, "w") as log_file:
        wer_list = []

        # Her test örneği için transkripsiyon oluştur
        for idx in range(len(dataset['test'])):
            sample = dataset['test'][idx]
            audio_array = np.array(sample['audio']['array'])
            audio_sr = sample['audio']['sampling_rate']
            ground_truth = sample['text']

            # Geçici ses dosyası oluştur
            audio_path = f"temp_audio_{idx}.wav"
            sf.write(audio_path, audio_array, audio_sr)

            # Ses dosyasını oku
            audio_input, _ = sf.read(audio_path)

            # Model girdilerini hazırla
            inputs = processor(audio_input, return_tensors="pt", sampling_rate=audio_sr)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}

            # Transkripsiyon oluştur
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            normalized_prediction = prediction.strip()

            # WER hesapla
            wer = jiwer.wer(ground_truth, normalized_prediction)
            wer_list.append(wer)

            # Sonuçları kaydet
            log_file.write(f"--------------------------------------------------\n")
            log_file.write(f"Örnek {idx + 1}:\n")
            log_file.write(f"Gerçek Metin: {ground_truth}\n")
            log_file.write(f"Tahmin: {normalized_prediction}\n")
            log_file.write(f"Kelime Hata Oranı (WER): {wer * 100:.2f}%\n")
            log_file.write(f"--------------------------------------------------\n")

            # Geçici ses dosyasını sil
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Ortalama WER hesapla
        avg_wer = np.mean(wer_list) * 100

        log_file.write(f"\n{len(dataset['test'])} örnek üzerindeki Ortalama Kelime Hata Oranı (WER): {avg_wer:.2f}%\n")
        log_file.write(f"--------------------------------------------------\n")

    # Sonuçları işle
    process_evaluation_log(log_file_path, avg_wer)

def process_evaluation_log(log_file_path, avg_wer):
    """
    Değerlendirme sonuçlarını işleyip CSV dosyasına kaydetme fonksiyonu
    
    Parametreler:
    log_file_path: İşlenecek log dosyasının yolu
    avg_wer: Ortalama WER değeri
    """
    # Log dosyasını oku
    with open(log_file_path, 'r') as file:
        data = file.read()

    # Sonuçları düzenli ifadelerle ayıkla
    pattern = r"Örnek\s+(\d+):\s*Gerçek Metin:\s*(.+)\s*Tahmin:\s*(.+)\s*Kelime Hata Oranı \(WER\):\s*([\d.]+)%"
    matches = re.findall(pattern, data)

    # DataFrame oluştur
    df = pd.DataFrame(matches, columns=['Örnek', 'Gerçek Metin', 'Tahmin', 'WER'])

    # Veri tiplerini düzelt
    df['Örnek'] = df['Örnek'].astype(int)
    df['WER'] = df['WER'].astype(float)

    # WER'e göre sırala
    df_sorted = df.sort_values(by='WER', ascending=False)

    # Dosya adı için WER değerini formatla
    wer_str = f"{avg_wer:.2f}"

    # CSV dosyasına kaydet
    csv_filename = f'whisper-medium.en-fine-tuned-for-ATC-{wer_str}-WER-evaluation-data.csv'
    df_sorted.to_csv(csv_filename, index=False)

    print(f"Ortalama WER: {wer_str}%")
    print(f"Sonuçlar şu dosyaya kaydedildi: {csv_filename}")

    # Geçici log dosyasını sil
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

if __name__ == "__main__":
    model_name = "mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper"
    generate_transcription_and_process_results(model_name=model_name) 