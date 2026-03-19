# Google Colab: Unsloth ile Qwen Modelini Tıbbi Verilerle Eğitme (Fine-Tuning) Rehberi

SakanaAI'nin *Doc-to-LoRA* mimarisi anlık belge içselleştirme için (genellikle Llama/Gemma için önceden eğitilmiş hiperağlarla) kullanılır. Ancak modeli kapsamlı bir **tıbbi veri setiyle** kalıcı olarak eğitmek (Fine-Tuning) istiyorsanız, **Unsloth** ile standart SFT (Supervised Fine-Tuning) yapmak en verimli ve doğru yöntemdir.

Bu rehberde, **Qwen** (örn: `unsloth/Qwen2.5-7B-Instruct`) modelini örnek bir tıbbi soru-cevap veri setiyle Google Colab (T4 GPU) üzerinde nasıl eğiteceğinizi adım adım görebilirsiniz.

## Adım 1: Kurulumlar
Google Colab'de yeni bir not defteri açıp (Runtime: T4 GPU), ilk hücreye aşağıdakileri yapıştırın:

```bash
# Unsloth ve gerekli kütüphanelerin kurulumu
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes datasets
```

## Adım 2: Qwen Modelini Yükleme
Unsloth kullanarak Qwen'i bellek dostu 4-bit formatında indireceğiz.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # GPU belleğine göre artırabilirsiniz
dtype = None
load_in_4bit = True # 4-bit Quantization (T4 GPU için şart)

# Qwen2.5 modelini çekiyoruz
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA Adaptörlerini ekleyelim (Eğitilecek parametreler)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank (daha fazla öğrenme kapasitesi için 32-64 yapılabilir)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimize edilmiş eğitim için 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Çok uzun contextler için VRAM tasarrufu
    random_state = 3407,
)
```

## Adım 3: Tıbbi Veri Setini Hazırlama
Eğitim için bir hasta-doktor soru-cevap veri seti (örneğin HuggingFace'deki `ruslanmv/ai-medical-chatbot`) kullanabiliriz. Veriyi Qwen'in `ChatML` veya standart prompt yapısına dönüştüreceğiz.

```python
from datasets import load_dataset

# Örnek bir tıbbi QA veri setini indiriyoruz
dataset = load_dataset("ruslanmv/ai-medical-chatbot", split = "train")

# Chat prompt şablonu (Qwen genellikle ChatML formatını kullanır)
medical_prompt = """Aşağıda bir hastanın sorusu ve bir doktorun tıbbi tavsiyesi bulunmaktadır. 
Uygundur bir şekilde cevapla.

### Hasta:
{}

### Doktor:
{}"""

EOS_TOKEN = tokenizer.eos_token # Cümle sonu token'ı

def formatting_prompts_func(examples):
    patients = examples["Patient"]
    doctors  = examples["Doctor"]
    texts = []
    
    for patient, doctor in zip(patients, doctors):
        # Formatlanmış metni oluştur ve EOS token ekle
        text = medical_prompt.format(patient, doctor) + EOS_TOKEN
        texts.append(text)
        
    return { "text" : texts }

# Veri setine fonksiyonu uyguluyoruz
formatted_dataset = dataset.map(formatting_prompts_func, batched = True)
```

## Adım 4: Eğitimi Başlatma
HuggingFace `SFTTrainer` kullanarak eğitime başlıyoruz.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # VRAM yetersizliğinde hızı artırmak için True yapılabilir
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Demo için 60 adım. Tam eğitim için 'num_train_epochs = 1' kullanın
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Eğitimi Başlat
trainer_stats = trainer.train()
```

## Adım 5: Tıbbi Model ile Çıkarım (Inference) Yapma
Eğitim bittikten sonra yeni tıbbi bilgisiyle modeli test edelim.

```python
# Modeli hızlı çıkarım(inference) moduna alıyoruz
FastLanguageModel.for_inference(model)

# Yeni bir soru testi
test_patient_question = "Sabahtan beri şiddetli baş ağrım var ve midem bulanıyor. Ne yapmalıyım?"

test_prompt = medical_prompt.format(test_patient_question, "")

inputs = tokenizer(
    [test_prompt], 
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens = 256, 
    use_cache = True
)

print("\n--- Eğitilmiş Model Yanıtı ---\n")
print(tokenizer.batch_decode(outputs)[0])
```

## Adım 6: Modeli Kaydetme
Eğittiğiniz LoRA adaptörünü kaydetmek isterseniz:

```python
# Sadece adaptörü kaydet (Hızlı ve düşük boyutlu)
model.save_pretrained("qwen_medical_lora")
tokenizer.save_pretrained("qwen_medical_lora")

# HuggingFace'e yüklemek için:
# model.push_to_hub("kullanici_adiniz/qwen_medical_lora", token = "hf_token_buraya")
```

Bu not defteri ile bir Qwen (2.5) modelini dilediğiniz kendi özel tıbbi `.jsonl` / `.csv` veri setinizle eğitebilirsiniz. Veri kaynağınızı kendi verilerinize göre `Adım 3` bölümünde değiştirmeyi unutmayın!
