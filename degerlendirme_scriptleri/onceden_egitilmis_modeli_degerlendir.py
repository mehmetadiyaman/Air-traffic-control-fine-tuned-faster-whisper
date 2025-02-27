# Önceden Eğitilmiş Whisper Modelinin Değerlendirme Scripti
# Bu script, önceden eğitilmiş Whisper modelinin performansını test veri seti üzerinde değerlendirir

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import soundfile as sf
import os
import jiwer
import string
from num2words import num2words
import re
import pandas as pd
from datasets import load_dataset

# Test veri setini yükle
dataset = load_dataset("mehmedadymn/air-traffic-dataset")

# NATO fonetik alfabesi
phonetic_alphabet = {
    'a': 'alfa', 'b': 'bravo', 'c': 'charlie', 'd': 'delta', 'e': 'echo',
    'f': 'foxtrot', 'g': 'golf', 'h': 'hotel', 'i': 'india', 'j': 'juliett',
    'k': 'kilo', 'l': 'lima', 'm': 'mike', 'n': 'november', 'o': 'oscar',
    'p': 'papa', 'q': 'quebec', 'r': 'romeo', 's': 'sierra', 't': 'tango',
    'u': 'uniform', 'v': 'victor', 'w': 'whiskey', 'x': 'x-ray', 'y': 'yankee', 'z': 'zulu'
}

def normalize_prediction(text):
    """
    Model tahminlerini normalize etme fonksiyonu.
    Sayıları, harfleri ve özel formatları standart bir biçime dönüştürür.
    """
    def convert_flight_level(match):
        """Uçuş seviyesi formatını dönüştürür (örn. FL350 -> flight level three five zero)"""
        flight_level_number = match.group(1)
        expanded_flight_level = ' '.join([num2words(int(digit)) for digit in flight_level_number])
        return f"flight level {expanded_flight_level}"

    def convert_altitude(match):
        """Yükseklik formatını dönüştürür (örn. 5000 feet -> five thousand feet)"""
        number = match.group(1).replace(',', '')
        feet = match.group(2)
        num = int(number)
        number_in_words = num2words(num)
        return f"{number_in_words} {feet}"

    def convert_hyphenated_numbers(match):
        """Tire ile ayrılmış sayı ve harfleri dönüştürür"""
        hyphenated_number = match.group()
        segments = hyphenated_number.split('-')
        result = []
        for segment in segments:
            sub_result = []
            for char in segment:
                if char.isdigit():
                    sub_result.append(num2words(int(char)))
                elif char.isalpha():
                    sub_result.append(phonetic_alphabet[char.lower()])
            result.append(' '.join(sub_result))
        return ' '.join(result)

    def convert_alphanumeric(match):
        """Alfanümerik kodları dönüştürür"""
        token = match.group()
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)

        # Pist numarası kontrolü (örn. 24L -> two four left)
        runway_match = re.match(r'^(\d{1,2})([LR])$', token, re.IGNORECASE)
        if runway_match:
            number = runway_match.group(1)
            direction = runway_match.group(2).upper()
            expanded_number = ' '.join([num2words(int(digit)) for digit in number])
            expanded_direction = 'left' if direction == 'L' else 'right'
            return f"{expanded_number} {expanded_direction}"

        if has_digit and has_alpha:
            result = []
            for char in token:
                if char.isdigit():
                    result.append(num2words(int(char)))
                elif char.isalpha():
                    result.append(phonetic_alphabet[char.lower()])
            return ' '.join(result)
        return token

    def convert_digits(match):
        """Sayıları kelime formatına dönüştürür"""
        number = match.group()
        if re.match(r'^\d+\.\d+$', number):  # Ondalık sayılar
            parts = number.split('.')
            integer_part = ' '.join([num2words(int(digit)) for digit in parts[0]])
            decimal_part = ' '.join([num2words(int(digit)) for digit in parts[1]])
            return f"{integer_part} decimal {decimal_part}"
        elif number.isdigit():  # Tam sayılar
            return ' '.join([num2words(int(digit)) for digit in number])
        return number

    def convert_single_letters(match):
        """Tek harfleri NATO fonetik alfabesine dönüştürür"""
        letter = match.group(1)
        return phonetic_alphabet[letter.lower()] if len(letter) == 1 else letter

    # Dönüşümleri uygula
    text = re.sub(r'\bFL(\d+)\b', convert_flight_level, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d{1,3}(?:,\d{3})?)\s*(feet)', convert_altitude, text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b', convert_hyphenated_numbers, text)
    text = re.sub(r'\b(?=[A-Za-z]*\d)[A-Za-z0-9]+\b', convert_alphanumeric, text)
    text = re.sub(r'\d+(\.\d+)?', convert_digits, text)
    text = re.sub(r'(?<=\s)([b-hj-zB-HJ-Z])(?=\s)', convert_single_letters, text)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = text.replace('take off', 'takeoff')
    text = text.translate(str.maketrans('', '', string.punctuation))
    normalized_text = text.lower()

    return normalized_text

def generate_transcription_and_process(model_name):
    """
    Transkripsiyon oluşturma ve sonuçları işleme fonksiyonu
    
    Parametreler:
    model_name: Kullanılacak modelin adı
    """
    # Model ve işlemciyi yükle
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    log_file_path = "whisper-medium-tr-degerlendirme-ham-veriler.txt"

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

            # Ses dosyasını oku ve model girdilerini hazırla
            audio_input, _ = sf.read(audio_path)
            inputs = processor(audio_input, return_tensors="pt", sampling_rate=audio_sr)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}

            # Transkripsiyon oluştur
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Tahmini normalize et
            normalized_prediction = normalize_prediction(prediction)

            # WER hesapla
            wer = jiwer.wer(ground_truth, normalized_prediction)
            wer_list.append(wer)

            # Sonuçları kaydet
            log_file.write(f"--------------------------------------------------\n")
            log_file.write(f"Örnek {idx + 1}:\n")
            log_file.write(f"Gerçek Metin: {ground_truth}\n")
            log_file.write(f"Tahmin: {prediction}\n")
            log_file.write(f"Normalize Edilmiş Tahmin: {normalized_prediction}\n")
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
    pattern = r"Örnek\s+(\d+):\s*Gerçek Metin:\s*(.+?)\s*Tahmin:.*?\s*Normalize Edilmiş Tahmin:\s*(.+?)\s*Kelime Hata Oranı \(WER\):\s*([\d.]+)%"
    matches = re.findall(pattern, data)

    if not matches:
        print("Eşleşme bulunamadı. Lütfen girdi dosyasının formatını veya regex desenini kontrol edin.")
    else:
        # DataFrame oluştur
        df = pd.DataFrame(matches, columns=['Örnek', 'Gerçek Metin', 'Tahmin', 'WER'])

        # Veri tiplerini düzelt
        df['Örnek'] = df['Örnek'].astype(int)
        df['WER'] = df['WER'].astype(float)

        # Ortalama WER hesapla
        average_wer = df['WER'].mean()
        print(f"Ortalama WER: {average_wer:.2f}%")

        # WER'e göre sırala
        df_sorted = df.sort_values(by='WER', ascending=False)

        # Dosya adı için WER değerini formatla
        wer_str = f"{avg_wer:.2f}"

        # CSV dosyasına kaydet
        csv_filename = f'whisper-medium.en-{wer_str}-WER-evaluation-data.csv'
        df_sorted.to_csv(csv_filename, index=False)

        print(f"Sonuçlar şu dosyaya kaydedildi: {csv_filename}")

        # Geçici log dosyasını sil
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

if __name__ == "__main__":
    model_name = "openai/whisper-medium.en"
    generate_transcription_and_process(model_name=model_name) 