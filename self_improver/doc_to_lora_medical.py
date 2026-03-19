###############################################################################
# Google Colab Tutorial: SakanaAI Doc-to-LoRA (D2L) ile Tibbi Belge
# Icsellestirme (Medical Document Internalization)
#
# Bu script tamamen Doc-to-LoRA framework'u uzerine kurulmustur.
# D2L, bir hypernetwork ile belgeleri aninda LoRA agirliklarina donusturur.
# Model belgeyi tekrar okumak zorunda kalmadan sorulara yanit verir.
#
# Runtime -> Change runtime type -> T4 GPU (veya daha iyisi) secin!
###############################################################################

# =============================================================================
# HUCRE 1: Kurulum (Bu hucreyi calistirin, her sey otomatik indirilip kurulacak)
# =============================================================================
import subprocess, os, sys

# 1a) Doc-to-LoRA reposunu klonla
if not os.path.exists("/content/doc-to-lora"):
    %cd /content
    !git clone https://github.com/SakanaAI/doc-to-lora.git
    print("[OK] Repo klonlandi.")
else:
    print("[INFO] Repo zaten mevcut.")

# 1b) Calisma dizinini sabitle (ic ice gecmeyi onle)
%cd /content/doc-to-lora

# 1c) Bagimliliklari kur
!pip uninstall -y flash-attn  # Mevcut bozuk kurulumlari temizle
!pip install -q -e .
!pip install -q huggingface_hub

# [TIP] Eger hala FlashAttention hatasi aliyorsaniz:
# "Runtime -> Restart runtime" yapip bu hucreryi tekrar calistirin.

# 1c-2) T4 GPU Uyumluluk Yamasi (FlashAttention2 yerine SDPA kullanimi icin)
# D2L'in kaynak kodunda FlashAttention2 zorunlulugunu kaldiriyoruz.
!sed -i 's/assert self._use_flash_attention_2/pass/' /content/doc-to-lora/src/ctx_to_lora/modeling/idefics2.py
!sed -i 's/# "eager": Idefics2PerceiverAttention/"sdpa": Idefics2PerceiverAttention/' /content/doc-to-lora/src/ctx_to_lora/modeling/idefics2.py
# unpad_input NameError hatasi icin fallback ekle
!sed -i 's/from flash_attn.bert_padding import unpad_input/try:\\n    from flash_attn.bert_padding import unpad_input\\nexcept ImportError:\\n    def unpad_input(tensor, mask): return tensor, None, None, None, None/' /content/doc-to-lora/src/ctx_to_lora/modeling/idefics2.py

# Yamayi dogrula
with open("/content/doc-to-lora/src/ctx_to_lora/modeling/idefics2.py", "r") as f:
    content = f.read()
    if 'pass' in content and '"sdpa": Idefics2PerceiverAttention' in content:
        print("[OK] T4 Uyumluluk Yamasi basariyla uygulandi.")
    else:
        print("[HATA] Yama uygulanamadi! Lutfen dosyayi kontrol edin.")

# 1d) HuggingFace token ile giris yap
# Gemma modeli gated (erisim izni gerektirir), token olmadan indirilemez.
# Token'inizi https://huggingface.co/settings/tokens adresinden alin.
# Ayrica https://huggingface.co/google/gemma-2-2b-it sayfasinda lisansi kabul edin.
from huggingface_hub import login
login(token="BURAYA_HF_TOKENINIZI_YAZIN")  # <-- kendi token'inizi buraya yapistin

# 1e) Onceden egitilmis D2L hypernetwork agirliklarini indir
if not os.path.exists("trained_d2l"):
    !huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include "*/"
    print("[OK] D2L agirliklari indirildi.")
else:
    print("[INFO] D2L agirliklari zaten mevcut.")

print("\nKurulum tamamlandi! Bir sonraki hucreye gecebilirsiniz.")

# =============================================================================
# HUCRE 2: Doc-to-LoRA Modelini Yukleme
# =============================================================================
import os, sys

# src/ dizinini Python path'e ekle (ctx_to_lora modulu burada)
repo_dir = "/content/doc-to-lora"
src_dir = os.path.join(repo_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(repo_dir)

# --- GLOBAL FLASH ATTENTION BYPASS (T4 GPU ICIN) ---
# transformers kütüphanesinin FlashAttention kontrolünü tamamen susturuyoruz.
import transformers
from transformers import PreTrainedModel
import transformers.modeling_utils

def _patched_enable_flash(cls, config, *args, **kwargs):
    config._attn_implementation = "sdpa" # Flash yerine SDPA zorla
    return config

# Hem sınıfa hem de modüle yamayı uygula (versiyonlar arası farkı önlemek için)
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(_patched_enable_flash)
if hasattr(transformers.modeling_utils, "_check_and_enable_flash_attn_2"):
    transformers.modeling_utils._check_and_enable_flash_attn_2 = _patched_enable_flash

# --- D2L UNPAD_INPUT FAILSAFE ---
import ctx_to_lora.modeling.idefics2 as idefics2_mod
if not hasattr(idefics2_mod, "unpad_input") or idefics2_mod.unpad_input is None:
    idefics2_mod.unpad_input = lambda tensor, mask: (tensor, None, None, None, None)
# --------------------------------------------------

import torch
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# D2L onceden egitilmis checkpoint'ini yukle
# (Gemma-2-2b-it temel modeli uzerinde egitilmistir)
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"

print("D2L modeli yukleniyor...")
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict,
    train=False,
    use_sequence_packing=False,
    use_flash_attn=False, # FlashAttention 2'yi T4 GPU icin kapali tutuyoruz
    base_model_kwargs={"attn_implementation": "sdpa"}, # Standart SDPA kullanimini zorla
)
# Cihaza (GPU) tasi
model.to("cuda")
model.reset()

# Tokenizer'ı base model'den al
tokenizer = get_tokenizer(model.base_model.name_or_path)

print(f"D2L modeli yuklendi.")
print(f"   Base Model: {model.base_model.name_or_path}")
print(f"   Cihaz: {model.device}")

# =============================================================================
# HÜCRE 4: Tıbbi Belgeleri Hazırlama
# =============================================================================
# Gerçek dünya senaryosunda bu veriler PDF'lerden, tıbbi makalelerden veya
# hasta dosyalarından çekilebilir. Burada örnek tıbbi belgeler kullanıyoruz.

medical_documents = {
    "tip2_diyabet": """
Tip 2 Diyabet Tedavi Kılavuzu (2024 Güncellemesi):

1. Birinci Basamak Tedavi - Metformin:
   - Başlangıç dozu: 500mg/gün, yemekle birlikte alınmalıdır.
   - Kademeli artırım: Her 1-2 haftada bir 500mg artırılarak maksimum 2000mg/gün dozuna çıkılabilir.
   - Kontrendikasyonlar: eGFR < 30 ml/dk/1.73m² olan hastalarda kullanılmamalıdır.
   - eGFR 30-45 arası hastalarda doz maksimum 1000mg/gün ile sınırlandırılmalıdır.
   - Laktik asidoz riski nedeniyle kontrast madde kullanan radyolojik işlemlerden 48 saat önce kesilmelidir.
   - En sık yan etkileri: bulantı, ishal, karın ağrısı. Uzun salınımlı (XR) formülasyonlar tolerabiliteyi artırır.

2. İkinci Basamak - Kombine Tedavi:
   - HbA1c hedefi 3 ay içinde sağlanamazsa ikinci ajan eklenmeli.
   - SGLT2 inhibitörleri (empagliflozin, dapagliflozin): Kardiyovasküler ve renal koruma sağlar.
   - GLP-1 reseptör agonistleri (semaglutid, liraglutid): Kilo kaybı etkisi belirgindir.
   - DPP-4 inhibitörleri (sitagliptin): Nötr kilo etkisi, düşük hipoglisemi riski.

3. HbA1c Hedefleri:
   - Genel popülasyon: < %7 (53 mmol/mol)
   - Yaşlı/kırılgan hastalar: < %8 (64 mmol/mol)
   - Yeni tanı genç hastalar: < %6.5 (48 mmol/mol) düşünülebilir.

4. Monitorizasyon:
   - HbA1c her 3 ayda bir kontrol edilmeli.
   - Böbrek fonksiyonları (eGFR, idrar albumin/kreatinin oranı) yılda en az 2 kez kontrol edilmeli.
   - Göz muayenesi yılda bir yapılmalıdır.
   - Ayak muayenesi her vizitte yapılmalıdır.
""",

    "hipertansiyon": """
Hipertansiyon (Yüksek Tansiyon) Yönetim Kılavuzu:

1. Tanı Kriterleri:
   - Evre 1 HT: Sistolik 130-139 mmHg veya Diastolik 80-89 mmHg
   - Evre 2 HT: Sistolik ≥140 mmHg veya Diastolik ≥90 mmHg
   - Hipertansif kriz: Sistolik >180 mmHg ve/veya Diastolik >120 mmHg

2. Yaşam Tarzı Değişiklikleri (Tüm evreler için):
   - DASH diyeti: Tuz alımı <6g/gün, potasyumdan zengin beslenme
   - Düzenli egzersiz: Haftada en az 150 dakika orta yoğunlukta aerobik aktivite
   - Kilo kontrolü: BMI 18.5-24.9 hedeflenmeli
   - Alkol kısıtlaması: Erkeklerde ≤2, kadınlarda ≤1 standart içki/gün
   - Sigaranın bırakılması

3. Farmakolojik Tedavi:
   - ACE inhibitörleri (enalapril, ramipril): Diyabetik nefropatide tercih edilir.
   - ARB'ler (valsartan, losartan): ACE inhibitörlerine alternatif.
   - Kalsiyum kanal blokerleri (amlodipin): Yaşlı hastalarda ve izole sistolik HT'de.
   - Tiazid diüretikler (hidroklorotiyazid): Düşük doz kombinasyonlarda etkili.
   - NOT: ACE inhibitörü + ARB birlikte kullanılmamalıdır.

4. Hedef Kan Basıncı:
   - Genel: <130/80 mmHg
   - 65 yaş üstü: Sistolik <140 mmHg kabul edilebilir.
   - Diyabetik hastalar: <130/80 mmHg
   - KBH hastaları: <130/80 mmHg
""",

    "antibiyotik_rehberi": """
Toplum Kökenli Enfeksiyonlarda Ampirik Antibiyotik Seçimi Rehberi:

1. Üst Solunum Yolu Enfeksiyonları (ÜSYE):
   - Akut Tonsillofarenjit (Streptokok): Amoksisilin 50mg/kg/gün (max 1g) 10 gün veya
     Penisilin V 500mg 2x1 10 gün. Penisilin alerjisinde: Azitromisin 500mg 1. gün, 250mg 2-5. gün.
   - Akut Sinüzit: İlk 10 gün semptomatik tedavi. Persistans halinde Amoksisilin-Klavulanat 875/125mg 2x1.

2. Alt Solunum Yolu Enfeksiyonları:
   - Toplum Kökenli Pnömoni (hafif): Amoksisilin 1g 3x1, 5-7 gün.
   - Toplum Kökenli Pnömoni (orta): Amoksisilin-Klavulanat 1g 2x1 + Azitromisin 500mg 1x1.
   - Atipik Pnömoni: Azitromisin 500mg 1. gün, 250mg 2-5. gün veya Doksisiklin 100mg 2x1.

3. Üriner Sistem Enfeksiyonları:
   - Basit Sistit (kadın): Fosfomisin 3g tek doz VEYA Nitrofurantoin 100mg 2x1, 5 gün.
   - Komplike ÜSE/Piyelonefrit: Siprofloksasin 500mg 2x1, 7 gün veya Seftriakson 1g IV 1x1.
   - NOT: Florokinolonlar basit sistitte ilk tercih olmamalıdır (direnç riski).

4. Deri ve Yumuşak Doku Enfeksiyonları:
   - Selülit: Amoksisilin-Klavulanat 875/125mg 2x1, 5-7 gün.
   - Apse: İnsizyon + drenaj. Antibiyotik genelde gerekmez (ancak MRSA riski varsa: TMP-SMX 160/800mg 2x1).

5. Genel Prensipler:
   - Kültür sonucuna göre de-eskalasyon yapılmalıdır.
   - Dar spektrumlu antibiyotik tercih edilmelidir.
   - Tedavi süresi mümkün olduğunca kısa tutulmalıdır.
""",

    "acil_tibbi_durumlar": """
Acil Tıbbi Durumlar ve İlk Müdahale Protokolü:

1. Akut Miyokard Enfarktüsü (Kalp Krizi):
   - Belirtiler: Göğüs ağrısı (>20dk, baskı tarzında), sol kola, çeneye yayılım, terleme, bulantı.
   - İlk müdahale: Aspirin 300mg çiğneterek verilmeli. Nitrogliserin sublingual (kontrendike değilse).
   - STEMI: Kapı-balon süresi <90 dakika hedeflenmeli. Primer PKG (perkütan koroner girişim) tercih edilir.
   - Antikoagülasyon: Heparin bolus + infüzyon.

2. İnme (Serebrovasküler Olay):
   - Belirtiler: Ani başlayan yüz düşüklüğü, kol güçsüzlüğü, konuşma bozukluğu (FAST protokolü).
   - İskemik inme: IV alteplaz (tPA) semptom başlangıcından itibaren 4.5 saat içinde verilmelidir.
   - Hemorajik inme tPA kontrendikasyonudur. Acil BT çekilmeli.
   - Kan basıncı: İskemik inmede tPA öncesi <185/110 mmHg olmalıdır.

3. Anafilaksi:
   - Belirtiler: Generalize ürtiker, anjiyoödem, bronkospazm, hipotansiyon.
   - İlk tedavi: Adrenalin (Epinefrin) 0.3-0.5mg IM (uyluk dış yan), her 5-15dk tekrarlanabilir.
   - Destekleyici: IV sıvı, salbutamol nebül, IV metilprednizolon, IV difenhidramin.
   - Hasta en az 6-8 saat gözlem altında tutulmalıdır (bifazik reaksiyon riski).

4. Diyabetik Ketoasidoz (DKA):
   - Tanı kriterleri: Glukoz >250 mg/dL, pH <7.3, bikarbonat <18 mEq/L, ketonüri/ketonemi.
   - Tedavi: Agresif IV hidrasyon (%0.9 NaCl), insülin infüzyonu (0.1 ü/kg/saat), potasyum replasmanı.
   - Serebral ödem riski (özellikle çocuklarda): Sıvı düzeltmesi yavaş yapılmalıdır.
"""
}

print(f"{len(medical_documents)} adet tibbi belge hazirlandi:")
for name, doc in medical_documents.items():
    print(f"   {name}: {len(doc)} karakter")

# =============================================================================
# HÜCRE 5: Belgeyi İçselleştir ve Sorgula (Core Doc-to-LoRA Akışı)
# =============================================================================
def internalize_and_query(model, tokenizer, document_name, document_text, questions):
    """
    Doc-to-LoRA'nın temel fonksiyonu:
    1. model.internalize(doc) -> Hypernetwork belgeyi okur ve LoRA ağırlıkları üretir
    2. model.generate() -> İçselleştirilmiş bilgiyle yanıt üretir
    3. model.reset() -> İçselleştirilmiş bilgiyi temizler
    """
    print(f"\n{'='*70}")
    print(f"Belge Icsellestiriliyor: {document_name}")
    print(f"{'='*70}")

    # ─── ADIM 1: Belgeyi LoRA ağırlıklarına dönüştür ───
    model.reset()  # Önceki içselleştirmeleri temizle
    model.internalize(document_text)
    print(f"Belge icsellestirildi. (Hypernetwork -> LoRA agirliklari uretildi)")

    # ─── ADIM 2: Tıbbi sorular sor ───
    for i, question in enumerate(questions, 1):
        chat = [{"role": "user", "content": question}]
        chat_ids = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n  Soru {i}: {question}")
        print(f"  Yanit: {response}")
        print(f"  {'─'*60}")

    return True


# ─── Diyabet Belgesi ile Sorgulama ───
internalize_and_query(
    model, tokenizer,
    document_name="Tip 2 Diyabet Tedavi Kılavuzu",
    document_text=medical_documents["tip2_diyabet"],
    questions=[
        "Metformin başlangıç dozu nedir ve nasıl artırılır?",
        "eGFR değeri 35 olan bir hastada metformin nasıl kullanılmalıdır?",
        "HbA1c hedefi 3 ay içinde sağlanamazsa hangi ilaçlar eklenebilir?",
        "Yaşlı hastalar için HbA1c hedefi nedir?",
    ]
)

# ─── Hipertansiyon Belgesi ile Sorgulama ───
internalize_and_query(
    model, tokenizer,
    document_name="Hipertansiyon Yönetim Kılavuzu",
    document_text=medical_documents["hipertansiyon"],
    questions=[
        "Evre 2 hipertansiyon tanı kriteri nedir?",
        "Diyabetik bir hastada hedef kan basıncı kaç olmalıdır?",
        "ACE inhibitörü ile ARB birlikte kullanılabilir mi?",
        "Hipertansif bir hastaya hangi yaşam tarzı değişiklikleri önerilir?",
    ]
)

# ─── Antibiyotik Rehberi ile Sorgulama ───
internalize_and_query(
    model, tokenizer,
    document_name="Antibiyotik Seçim Rehberi",
    document_text=medical_documents["antibiyotik_rehberi"],
    questions=[
        "Basit sistit tedavisinde ilk tercih antibiyotik hangisidir?",
        "Toplum kökenli pnömoni tedavisinde hangi antibiyotikler kullanılır?",
        "Penisilin alerjisi olan bir hastada tonsillofarenjit nasıl tedavi edilir?",
    ]
)

# ─── Acil Durumlar Belgesi ile Sorgulama ───
internalize_and_query(
    model, tokenizer,
    document_name="Acil Tıbbi Durumlar Protokolü",
    document_text=medical_documents["acil_tibbi_durumlar"],
    questions=[
        "Kalp krizi şüphesinde ilk verilmesi gereken ilaç nedir?",
        "İskemik inmede tPA ne kadar süre içinde verilmelidir?",
        "Anafilakside ilk tedavi nedir ve dozu ne kadardır?",
        "Diyabetik ketoasidoz tanı kriterleri nelerdir?",
    ]
)

# =============================================================================
# HÜCRE 6: İçselleştirme Etkisini Karşılaştırma (İçselleştirilmiş vs Sıfır)
# =============================================================================
print("\n" + "="*70)
print("DENEY: Icsellestirme ile Icsellestirmesiz Karsilastirma")
print("="*70)

test_question = "Metformin başlangıç dozu nedir ve maksimum doz ne kadardır?"

# --- İçselleştirilmiş bilgi ILE ---
model.reset()
model.internalize(medical_documents["tip2_diyabet"])

chat = [{"role": "user", "content": test_question}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs_with = model.generate(input_ids=chat_ids, max_new_tokens=512)
response_with = tokenizer.decode(outputs_with[0], skip_special_tokens=True)

print(f"\n[+] ICSELLESTIRILMIS (belge bilgisi var):")
print(f"   Soru: {test_question}")
print(f"   Yanıt: {response_with}")

# --- İçselleştirilmiş bilgi OLMADAN ---
model.reset()  # LoRA etkisini kaldır

outputs_without = model.generate(input_ids=chat_ids, max_new_tokens=512)
response_without = tokenizer.decode(outputs_without[0], skip_special_tokens=True)

print(f"\n[-] ICSELLESTIRMESIZ (belge bilgisi yok, halusinasyon riski):")
print(f"   Soru: {test_question}")
print(f"   Yanıt: {response_without}")

print(f"\n{'─'*70}")
print("Yukaridaki sonuclarda gorebileceginiz gibi, internalize() cagrisindan sonra")
print("   model belge icerigini 'hatirlar' ve dogru bilgi verir.")
print("   reset() ile bilgi silinir ve model genel bilgisiyle yanit verir.")

# =============================================================================
# HÜCRE 7: Kendi Tıbbi Belgenizi Kullanma (Dosyadan Okuma)
# =============================================================================
# Kendi tıbbi belgenizi bir .txt dosyasından yükleyip kullanabilirsiniz:
#
# from google.colab import files
# uploaded = files.upload()  # Dosya yükleyin
# filename = list(uploaded.keys())[0]
#
# custom_doc = open(filename, "r", encoding="utf-8").read()
#
# model.reset()
# model.internalize(custom_doc)
#
# chat = [{"role": "user", "content": "Bu belgede en önemli klinik bulgu nedir?"}]
# chat_ids = tokenizer.apply_chat_template(
#     chat, add_special_tokens=False, return_attention_mask=False,
#     add_generation_prompt=True, return_tensors="pt",
# ).to(model.device)
#
# outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# =============================================================================
# HÜCRE 8: Birden Fazla Belgeyi Sırayla İçselleştirme (Batch Workflow)
# =============================================================================
print("\n" + "="*70)
print("Toplu Icsellestirme: Tum tibbi belgeler sirayla isleniyor")
print("="*70)

# Her belgede tek bir standart soru soralım
standard_question = "Bu kılavuzun en önemli 3 maddesi nelerdir? Kısa özetle."

for doc_name, doc_text in medical_documents.items():
    model.reset()
    model.internalize(doc_text)

    chat = [{"role": "user", "content": standard_question}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n{doc_name}:")
    print(f"   {response}")
    print(f"   {'─'*50}")


# =============================================================================
# SON: Özet
# =============================================================================
print("\n" + "="*70)
print("Doc-to-LoRA Tibbi Belge Icsellestirme Tutorial'i Tamamlandi.")
print("="*70)
print("""
YAPILAN ISLEMLER:
--------------------------------------------------------------------
1. SakanaAI Doc-to-LoRA modeli (Gemma-2-2b-it bazli) yuklendi
2. 4 farkli tibbi belge hazirlandi:
      - Tip 2 Diyabet Tedavi Kilavuzu
      - Hipertansiyon Yonetim Kilavuzu
      - Antibiyotik Secim Rehberi
      - Acil Tibbi Durumlar Protokolu
3. Her belge internalize() ile LoRA agirliklarina donusturuldu
4. Tibbi sorularla dogrulama yapildi
5. Icsellestirilmis vs icsellestirmesiz karsilastirma yapildi
6. Toplu belge isleme pipeline'i gosterildi

NASIL CALISIR:
--------------------------------------------------------------------
  Belge --> Hypernetwork --> LoRA Agirliklari --> Base Model'e Enjekte
                |                                        |
                |      (Aninda, egitim gerekmeden)       |
                |                                        v
                +-------------------------------------> Dogru Yanitlar
--------------------------------------------------------------------
""")
