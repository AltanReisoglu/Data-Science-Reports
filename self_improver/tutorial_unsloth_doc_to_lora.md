# Google Colab: Unsloth ve SakanaAI Doc-to-LoRA Kullanım Rehberi

Bu tutorial, **SakanaAI/doc-to-lora** (D2L) framework'ünü kullanarak bir modelin anında bilgi öğrenmesini ([Instant Context Internalization]) ve bu modelin **Unsloth** üzerinden hızlı, bellek dostu bir şekilde çekilmesini adım adım gösterir.

## Adım 1: Kurulumlar
Google Colab'de yeni bir not defteri açın ve ilk hücreye aşağıdaki kurulum komutlarını ekleyin. Bu komutlar uv paket yöneticisini, Unsloth'u ve doc-to-lora deposunu kurar.

```bash
# uv kurulumu (SakanaAI d2l genelde uv kullanır)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!export PATH="/root/.local/bin:$PATH"

# Unsloth kurulumu
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# SakanaAI doc-to-lora deposunu klonlayıp kurun
!git clone https://github.com/SakanaAI/doc-to-lora.git
%cd doc-to-lora
!pip install -e .
```

## Adım 2: Modelin Yüklenmesi (Unsloth üzerinden) ve D2L Hazırlığı
Doc-to-LoRA hiperağ modelinin eğitilmiş ağırlıklarını indirin. (*Not: SakanaAI'nin D2L ağırlıkları genellikle belirli temel modellere (örn. Llama 3 8B veya Gemma) ayarlıdır. O yüzden Unsloth sürümünü bu temel modele eşlemeliyiz.*)

```python
import torch
from unsloth import FastLanguageModel
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# 1. Unsloth ile Base Modeli ve Tokenizer'ı çekelim (Örn: Llama-3-8B-Instruct)
max_seq_length = 2048 # Unsloth destekli sequence uzunluğu
dtype = None # Auto detection
load_in_4bit = True # 4-bit qLORA memory tasarrufu için

model_name = "unsloth/llama-3-8b-Instruct"

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Unsloth modelini daha hızlı inference yapacak duruma getirelim
FastLanguageModel.for_inference(base_model)

# 2. SakanaAI Doc-to-LoRA Ağırlıklarını Yükleme
# (Aşağıdaki path repo kuralına göre değişiklik gösterebilir, indirdiğiniz ağırlık path'ini kullanın)
# !huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include "*/"

# D2L State_dict yüklenir ve Modulated Model'e dönüştürülür
# DİKKAT: D2L normalde kendi içinden bir base_model oluşturur. 
# Eğer Base Modeli zorlamak isterseniz objeyi manipüle etmeniz veya D2L mimarisinin peft adapter olarak eklenmesini sağlamanız gerekir.
checkpoint_path = "trained_d2l/llama_demo/checkpoint-xxxxx/pytorch_model.bin" # İndirilen uygun checkpoint

try:
    state_dict = torch.load(checkpoint_path, weights_only=False)
    d2l_model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    # Eğer özel Unsloth baz modelimizi içine enjekte etmek istersek:
    # d2l_model.base_model = base_model 
    d2l_model.reset()
except Exception as e:
    print("D2L Checkpoint yüklenirken hata oluştu. Lütfen model path'ini doğrulayın: ", e)

```

## Adım 3: Belgeyi (Doc) LoRA'ya Dönüştürme
Bir belgeyi okutup modelin bu bilgiyi içselleştirmesini sağlayalım.

```python
# Örnek bir bağlam hazırlıyoruz
document_text = """Sakana AI, Tokyo merkezli yeni nesil bir yapay zeka şirketidir. 
Doğadan ilham alan yapay zeka modelleri ve Doc-to-LoRA gibi anında adaptasyon frameworkleri geliştirmektedir."""

# D2L modeli bu metni direkt alır ve Hypernetwork üzerinden bir LoRA adapter üretip base_model'e uygular.
d2l_model.internalize(document_text)
print("Belge başarıyla LoRA'ya dönüştürüldü ve modele içselleştirildi!")
```

## Adım 4: Inference (Soru Sorup Doğrulama)
Model, az önce verdiğimiz belge metni üzerinden fine-tune edilmiş gibi davranacaktır.

```python
# Test sorusu
chat = [{"role": "user", "content": "Sakana AI nedir ve hangi frameworkleri geliştirir?"}]
chat_ids = tokenizer.apply_chat_template(
    chat, 
    add_special_tokens=False, 
    return_attention_mask=False, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(base_model.device)

# Modelden yanıt alalım
outputs = d2l_model.generate(input_ids=chat_ids, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("--- Modelin Yanıtı ---")
print(response)

# Etkileri sıfırlamak (Internalize edilen bilgiyi silmek) için:
# d2l_model.reset()
```

## Ek Notlar
- `SakanaAI/doc-to-lora` repo'su içindeki `base_model` beklentisi ile Unsloth'un `FastLanguageModel` modeli arasında Peft layer'larından ötürü uyuşmazlık çıkabilir. Eğer uyarı veya hata alırsanız Unsloth'u `load_in_4bit=False` olarak 16-bit formatında kullanmayı deneyebilir veya D2L'in kendi orijinal yükleyicisiyle devam edebilirsiniz.
- GPU Bellek (VRAM) sorunlarını çözmek için Colab'de T4 GPU kullanırken dikkatli olun, `batch_size` vs parametreler önemlidir.
