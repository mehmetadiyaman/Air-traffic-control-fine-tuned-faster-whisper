# Model Dışa Aktarma Scripti
# Bu script, eğitilmiş modeli çıkarım için dışa aktarır

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

# Model yolları
checkpoint_path = 'PATH_TO_YOUR_CHECKPOINT'  # Eğitilmiş model kontrol noktası yolu
base_model_id = 'INPUT_MODEL_ID_FOR_BASE_MODEL (e.g., openai/whisper-medium.en)'  # Temel model kimliği

# Model ve işlemciyi yükle
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
processor = WhisperProcessor.from_pretrained(base_model_id)

# Dışa aktarılacak model yolu
exported_model_path = checkpoint_path + '-exported-model'

# Modeli ve işlemciyi kaydet
os.makedirs(exported_model_path, exist_ok=True)
model.save_pretrained(exported_model_path)
processor.save_pretrained(exported_model_path) 