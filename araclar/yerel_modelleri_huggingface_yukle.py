# Hugging Face'e Model Yükleme Scripti
# Bu script, yerel modelleri Hugging Face model merkezine yükler

from huggingface_hub import HfApi, upload_folder

# Yüklenecek modeller ve hedef depolar
repos = {
    "mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper": "whisper-medium.en-fine-tuned-for-ATC-faster-whisper",
    "mehmedadymn/Air-traffic-control-fine-tuned-faster-whisper": "whisper-medium.en-fine-tuned-for-ATC"
}

# Hugging Face API'sini başlat
api = HfApi()

def upload_model(repo_name, model_folder):
    """
    Modeli Hugging Face'e yükleme fonksiyonu
    
    Parametreler:
    repo_name: Hedef depo adı
    model_folder: Yerel model klasörü
    """
    # Depo oluştur
    try:
        api.create_repo(repo_id=repo_name)
        print(f"{repo_name} deposu başarıyla oluşturuldu.")
    except Exception as e:
        print(f"{repo_name} deposu zaten var veya başka bir hata oluştu: {e}")

    # Modeli yükle
    try:
        upload_folder(
            folder_path=model_folder,
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"{model_folder} klasörü {repo_name} deposuna başarıyla yüklendi.")
    except Exception as e:
        print(f"{model_folder} klasörünü {repo_name} deposuna yüklerken hata oluştu: {e}")

# Her model için yükleme işlemini gerçekleştir
for repo_name, model_folder in repos.items():
    print(f"{model_folder} klasöründen {repo_name} deposuna model yükleniyor...")
    upload_model(repo_name, model_folder)
    print(f"{model_folder} yüklemesi tamamlandı.\n")

print("Her iki model de başarıyla yüklendi.") 