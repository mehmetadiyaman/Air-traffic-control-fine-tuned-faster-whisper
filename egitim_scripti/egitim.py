# ATC Verileri için Whisper Modeli İnce Ayar Eğitim Scripti
# Bu script, ATC iletişimleri için Whisper modelini ince ayar yapar

from datasets import load_dataset, Audio
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, ClippingDistortion, Trim

# Veri setini yükle
dataset = load_dataset("mehmedadymn/air-traffic-dataset")
train_dataset = dataset['train']
test_dataset = dataset['test']

# Model parametreleri
model_id = 'openai/whisper-medium.en'
out_dir = 'whisper-medium.en-atc-dataset'
epochs = 10

# Model bileşenlerini yükle
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language='English', task='transcribe')
processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')

# Ses örneklerini 16kHz'e dönüştür
train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=16000))

def augment_audio_with_clipping(audio, sampling_rate, augmentation_level):
    """Ses verisi artırma fonksiyonu"""
    noise_severity = augmentation_level * 0.05
    pitch_severity = augmentation_level * 2
    time_stretch_severity = 1.0 + (augmentation_level * 0.2)
    shift_severity = augmentation_level * 0.1
    gain_severity = augmentation_level * 10
    min_noise_amplitude = max(0.001, noise_severity / 10)
    
    # Ses artırma işlemleri
    augment = Compose([
        AddGaussianNoise(min_amplitude=min_noise_amplitude, max_amplitude=noise_severity, p=0.5),
        TimeStretch(min_rate=1.0, max_rate=time_stretch_severity, p=0.5),
        PitchShift(min_semitones=-pitch_severity, max_semitones=pitch_severity, p=0.5),
        Shift(min_shift=-shift_severity, max_shift=shift_severity, p=0.5),
        Gain(min_gain_in_db=-gain_severity, max_gain_in_db=gain_severity, p=0.5),
        ClippingDistortion(p=0.3),
        Trim(top_db=30, p=0.3)
    ])
    augmented_audio = augment(samples=audio, sample_rate=sampling_rate)
    return augmented_audio

def prepare_dataset(batch):
    """Veri setini model için hazırlama fonksiyonu"""
    audio = batch['audio']
    batch['input_features'] = []
    for a in audio:
        audio_array = a['array'].astype(np.float32)
        input_features = feature_extractor(audio_array, sampling_rate=a['sampling_rate']).input_features[0]
        batch['input_features'].append(input_features)
    tokenized = tokenizer(batch['text'], padding='longest', return_attention_mask=True)
    batch['labels'] = tokenized['input_ids']
    batch['decoder_attention_mask'] = tokenized['attention_mask']
    return batch

# Test veri setini hazırla
test_dataset = test_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=64,
    num_proc=4
)

class AugmentedDataset(torch.utils.data.Dataset):
    """Dinamik veri artırma için özel veri seti sınıfı"""
    def __init__(
        self, dataset, feature_extractor, tokenizer, augment_audio_with_clipping,
        initial_augmentation_level=0.5, final_augmentation_level=0.1, epochs=10
    ):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.augment_audio_with_clipping = augment_audio_with_clipping
        self.initial_augmentation_level = initial_augmentation_level
        self.final_augmentation_level = final_augmentation_level
        self.epochs = epochs - 1
        self.current_epoch = 0
        self.augmentation_level = initial_augmentation_level

    def set_epoch(self, epoch):
        """Epoch başına artırma seviyesini güncelle"""
        self.current_epoch = epoch
        decay_rate = (self.final_augmentation_level / self.initial_augmentation_level) ** (1 / self.epochs)
        self.augmentation_level = self.initial_augmentation_level * (decay_rate ** self.current_epoch)
        print(f"Epoch {self.current_epoch}: Artırma seviyesi {self.augmentation_level} olarak ayarlandı")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Veri örneğini hazırla ve artır"""
        batch = self.dataset[idx]
        audio = batch['audio']
        audio_array = audio['array'].astype(np.float32)
        augmented_audio = self.augment_audio_with_clipping(
            audio_array, audio['sampling_rate'], self.augmentation_level
        )
        input_features = self.feature_extractor(
            augmented_audio, sampling_rate=audio['sampling_rate']
        ).input_features[0]
        batch['input_features'] = input_features
        tokenized = self.tokenizer(
            batch['text'], padding='longest', return_attention_mask=True
        )
        batch['labels'] = tokenized['input_ids']
        batch['decoder_attention_mask'] = tokenized['attention_mask']
        return batch

# Artırılmış eğitim veri setini oluştur
train_dataset = AugmentedDataset(
    dataset=train_dataset,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    augment_audio_with_clipping=augment_audio_with_clipping,
    initial_augmentation_level=0.5,
    final_augmentation_level=0.1,
    epochs=epochs
)

class UpdateDatasetEpochCallback(TrainerCallback):
    """Her epoch başında veri seti artırma seviyesini güncelleyen callback"""
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(state.epoch)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Batch'leri hazırlayan veri toplayıcı"""
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Girdi özelliklerini hazırla
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors='pt'
        )
        
        # Etiketleri hazırla
        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors='pt'
        )
        labels = labels_batch['input_ids'].masked_fill(
            labels_batch['attention_mask'].ne(1), -100
        )
        batch['labels'] = labels
        batch['decoder_attention_mask'] = labels_batch['attention_mask']
        return batch

# Modeli yükle ve yapılandır
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.pad_token_id = tokenizer.pad_token_id

# Veri toplayıcıyı oluştur
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Değerlendirme metriğini yükle
metric = evaluate.load('wer')

def compute_metrics(pred):
    """WER (Word Error Rate) hesaplama fonksiyonu"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}

# Eğitim argümanlarını ayarla
training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=epochs,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    predict_with_generate=True,
    generation_max_length=225,
    report_to=['tensorboard'],
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    dataloader_num_workers=4,
    save_total_limit=2,
    lr_scheduler_type='cosine',
    seed=42,
    data_seed=42,
    weight_decay=0.01,
    bf16=True,
    fp16=False
)

# Eğiticiyi oluştur
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        UpdateDatasetEpochCallback(train_dataset)
    ]
)

# Eğitimi başlat
trainer.train()

# Son değerlendirme metriklerini hesapla ve yazdır
final_metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Son değerlendirme metrikleri:", final_metrics) 