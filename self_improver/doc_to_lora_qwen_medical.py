###############################################################################
# 📓 Google Colab Tutorial: Unsloth + Qwen + Tıbbi Veri Fine-Tuning
#    + SakanaAI Doc-to-LoRA Belge İçselleştirme
#
# Bu script'i Colab'de hücrelere bölüp çalıştırın.
# Runtime -> Change runtime type -> T4 GPU seçmeyi unutmayın!
###############################################################################

# =============================================================================
# HÜCRE 1: Kurulum
# =============================================================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# !pip install datasets

# Doc-to-LoRA kurulumu
# !git clone https://github.com/SakanaAI/doc-to-lora.git
# %cd doc-to-lora
# !pip install -e .

# D2L önceden eğitilmiş ağırlıkları indirme (opsiyonel, Bölüm B için)
# !pip install huggingface_hub
# !huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include "*/"

# =============================================================================
# HÜCRE 2: Qwen Modelini Unsloth ile Yükleme
# =============================================================================
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None           # Otomatik algılama (float16 veya bfloat16)
load_in_4bit = True    # 4-bit quantization (T4 GPU bellek tasarrufu)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("✅ Qwen2.5-7B-Instruct modeli başarıyla yüklendi!")

# =============================================================================
# HÜCRE 3: LoRA Adaptörleri Ekleme
# =============================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                 # LoRA rank (16-64 arası deneyin)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,       # Unsloth optimizasyonu için 0 önerilir
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # VRAM tasarrufu
    random_state = 3407,
)

print("✅ LoRA adaptörleri eklendi!")

# =============================================================================
# HÜCRE 4: Tıbbi Veri Setini Hazırlama
# =============================================================================
from datasets import load_dataset

# Açık kaynak tıbbi QA veri seti
dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

# İlk 5000 örneği kullanalım (demo amaçlı, tam eğitim için tamamını kullanın)
dataset = dataset.select(range(5000))

print(f"📊 Veri seti yüklendi: {len(dataset)} örnek")
print(f"📋 Sütunlar: {dataset.column_names}")
print(f"\n📝 Örnek veri:\n{dataset[0]}")

# Eğitim prompt şablonu
medical_prompt = """Aşağıda bir hastanın tıbbi sorusu ve bir doktorun detaylı yanıtı bulunmaktadır.

### Hasta Sorusu:
{}

### Doktor Yanıtı:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    patients = examples["Patient"]
    doctors  = examples["Doctor"]
    texts = []
    for patient, doctor in zip(patients, doctors):
        text = medical_prompt.format(patient.strip(), doctor.strip()) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"\n✅ Veri seti formatlandı!")
print(f"📝 Formatlanmış örnek:\n{formatted_dataset[0]['text'][:500]}...")

# =============================================================================
# HÜCRE 5: Eğitimi Başlatma (SFT - Supervised Fine-Tuning)
# =============================================================================
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # VRAM azsa True yaparak daha hızlı eğitim sağlanabilir
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # Effective batch = 2*4 = 8
        warmup_steps = 5,
        # --- Demo için 60 adım, tam eğitim için alttaki num_train_epochs'u aktifleştirin ---
        max_steps = 60,
        # num_train_epochs = 1,            # Tam eğitim için bu satırı açın, max_steps'i silin
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "qwen_medical_outputs",
        report_to = "none",  # wandb istemiyorsanız "none"
    ),
)

print("🚀 Eğitim başlıyor...\n")
trainer_stats = trainer.train()

print(f"\n✅ Eğitim tamamlandı!")
print(f"   Toplam süre: {trainer_stats.metrics['train_runtime']:.1f} saniye")
print(f"   Son loss: {trainer_stats.metrics['train_loss']:.4f}")

# =============================================================================
# HÜCRE 6: Eğitilmiş Modeli Test Etme (Inference)
# =============================================================================
FastLanguageModel.for_inference(model)

test_questions = [
    "Sabahtan beri şiddetli baş ağrım var ve midem bulanıyor. Ne yapmalıyım?",
    "Kan tahlilimde demir eksikliği çıktı. Hangi besinleri yemeliyim?",
    "Çocuğumun ateşi 39 derece, ne zaman hastaneye gitmeliyim?",
]

for i, question in enumerate(test_questions, 1):
    prompt = medical_prompt.format(question, "")

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Sadece doktor yanıtı kısmını alalım
    if "### Doktor Yanıtı:" in response:
        response = response.split("### Doktor Yanıtı:")[-1].strip()

    print(f"\n{'='*60}")
    print(f"🏥 Soru {i}: {question}")
    print(f"{'='*60}")
    print(f"🩺 Yanıt: {response}")

# =============================================================================
# HÜCRE 7: Modeli Kaydetme
# =============================================================================
# Sadece LoRA adaptörünü kaydet (küçük boyut, ~50MB)
model.save_pretrained("qwen_medical_lora")
tokenizer.save_pretrained("qwen_medical_lora")
print("💾 LoRA adaptörü kaydedildi: ./qwen_medical_lora/")

# HuggingFace Hub'a yüklemek isterseniz:
# model.push_to_hub("kullanici_adiniz/qwen-medical-lora", token="hf_...")
# tokenizer.push_to_hub("kullanici_adiniz/qwen-medical-lora", token="hf_...")

# Google Drive'a yedeklemek isterseniz:
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r qwen_medical_lora /content/drive/MyDrive/

###############################################################################
# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM B: Doc-to-LoRA ile Belge İçselleştirme (Document Internalization)
# ═══════════════════════════════════════════════════════════════════════════
#
# Doc-to-LoRA, bir belgenin içeriğini hypernetwork aracılığıyla anında
# LoRA ağırlıklarına dönüştürür. Bu sayede model, belgeyi tekrar okumak
# zorunda kalmadan soru cevaplayabilir.
#
# ÖNEMLİ NOT: Doc-to-LoRA'nın önceden eğitilmiş hypernetwork'ü belirli
# temel modellere (Gemma) göre eğitilmiştir. Qwen ile kullanmak için
# hypernetwork'ün yeniden eğitilmesi gerekir. Aşağıdaki kod, D2L'in
# orijinal Gemma checkpoint'i ile nasıl çalıştığını göstermektedir.
###############################################################################

# =============================================================================
# HÜCRE B1: Doc-to-LoRA ile Tıbbi Belge İçselleştirme (Gemma tabanlı demo)
# =============================================================================
# from ctx_to_lora.model_loading import get_tokenizer as d2l_get_tokenizer
# from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
#
# # Doc-to-LoRA modeli yükleme
# checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
# state_dict = torch.load(checkpoint_path, weights_only=False)
# d2l_model = ModulatedPretrainedModel.from_state_dict(
#     state_dict, train=False, use_sequence_packing=False
# )
# d2l_model.reset()
# d2l_tokenizer = d2l_get_tokenizer(d2l_model.base_model.name_or_path)
#
# # Tıbbi bir belge hazırlayalım (örnek: ilaç prospektüsü veya tıbbi makale)
# medical_document = """
# Tip 2 Diyabet Tedavi Kılavuzu:
# Metformin, tip 2 diyabet tedavisinde birinci basamak ilaç olarak önerilmektedir.
# Başlangıç dozu günde 500mg olup, kademeli olarak günde 2000mg'a kadar artırılabilir.
# Böbrek fonksiyonları eGFR < 30 ml/dk olan hastalarda kullanılmamalıdır.
# Laktik asidoz riski nedeniyle kontrast madde kullanılan radyolojik işlemlerden
# 48 saat önce kesilmelidir. En sık yan etkileri gastrointestinal şikayetlerdir.
# Uzun salınımlı formülasyonlar tolerabiliteyi artırabilir.
# HbA1c hedefi genellikle %7'nin altıdır.
# """
#
# # Belgeyi içselleştir (Hypernetwork LoRA ağırlıklarına dönüştürür)
# d2l_model.internalize(medical_document)
# print("✅ Tıbbi belge içselleştirildi!")
#
# # İçselleştirilmiş bilgiyle soru sor
# chat = [{"role": "user", "content": "Metformin ne zaman kesilmelidir?"}]
# chat_ids = d2l_tokenizer.apply_chat_template(
#     chat,
#     add_special_tokens=False,
#     return_attention_mask=False,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(d2l_model.device)
#
# outputs = d2l_model.generate(input_ids=chat_ids, max_new_tokens=512)
# print("\n🩺 D2L Yanıtı (belge içselleştirilmiş):")
# print(d2l_tokenizer.decode(outputs[0]))
#
# # İçselleştirilmiş bilgiyi temizle
# # d2l_model.reset()

print("\n" + "="*60)
print("🎉 Tutorial tamamlandı!")
print("="*60)
print("""
ÖZET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BÖLÜM A (Yukarıdaki kod):
  ✅ Unsloth ile Qwen2.5-7B-Instruct yüklendi (4-bit)
  ✅ LoRA adaptörleri eklendi
  ✅ Tıbbi QA veri setiyle fine-tune yapıldı
  ✅ 3 farklı tıbbi soruyla test edildi
  ✅ Model kaydedildi

BÖLÜM B (Yorum satırlarındaki kod):
  📝 Doc-to-LoRA ile tıbbi belge içselleştirme örneği
  📝 Gemma tabanlı D2L checkpoint kullanır
  📝 Belgeyi okutup anında LoRA ağırlıklarına dönüştürür
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
