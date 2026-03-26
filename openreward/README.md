# OpenReward AI - Tutorial & Quickstart Guide 🚀

Bu rehber, [OpenReward AI](https://docs.openreward.ai) framework'ünü kullanarak LLM (Büyük Dil Modelleri) ve Agent'ları değerlendirme, test etme ve Reinforcement Learning (RL) ile eğitme üzerine 4 aşamalı bir tutorial sunar.

## OpenReward Nedir?

OpenReward, Reinforcement Learning (RL) ortamlarını barındıran ve LLM ajanlarını eğitmek & değerlendirmek için standart bir protokol sunan açık kaynaklı bir platformdur. 

Temel yapısı **ORS (Open Reward Standard)** üzerine kuruludur:
- **Tasks (Görevler):** Çözülmesi gereken temel problemler ve başlangıç prompt'ları.
- **Tools (Araçlar):** Ajanın çevrede yapabileceği eylemler (örn. `bash`, `python_repl`).
- **Splits:** Görevleri eğitim (`train`) veya değerlendirme (`eval`) gibi gruplara ayırır.
- **Sandboxes:** Ajan kodunun güvenli, izole bir ortamda çalıştırılmasını sağlar.

---

## 🛠 Kurulum ve Hazırlık

### 1. Repository ve Bağımlılıklar
Öncelikle gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

*(Yüklenen ana paketler: `openreward`, `openai`, `anthropic`, `google-genai`, `python-dotenv`, `rich`)*

### 2. API Key Konfigürasyonu
`openreward` ortamlarına bağlanabilmek ve modelleri kullanabilmek için API anahtarlarına ihtiyacınız var:
1. [OpenReward API Key](https://openreward.ai/keys) alın.
2. Favori LLM sağlayıcınızın API anahtarını alın (örn. OpenAI).
3. `.env.example` dosyasının adını `.env` olarak değiştirin ve içini doldurun:

```env
OPENREWARD_API_KEY=your-openreward-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

---

## 📚 Tutorial Scriptleri

Bu klasörde OpenReward yeteneklerini gösteren 4 adet progresif script bulunmaktadır.

### 1. `01_quickstart.py` 🟢
**Ne yapar?** OpenReward'a bağlanmanın ve tek bir görevi basit bir OpenRouter (DeepSeek) agent döngüsü ile çözmenin en sade halidir.
**Nasıl Çalışır:** `kanishk/EndlessTerminals` ortamına bağlanır, görevi alır, `session` başlatır ve LLM'i bir `while` döngüsünde tool'ları kullanarak hedefe ulaşmaya teşvik eder. `rollout.log_openai_completions` ile adımları loglar.
```bash
python 01_quickstart.py
```

### 2. `02_multi_provider.py` 🔵
**Ne yapar?** Farklı modellerin OpenReward ortamında nasıl performans gösterdiğini karşılaştırır. API standardizasyonu için OpenRouter kullanır.
**Desteklenen Sağlayıcılar (OpenRouter Üzerinden):** OpenAI (gpt-4o), Anthropic (Claude 3.5 Sonnet), Google (Gemini 2.5 Flash), DeepSeek (v3.2).
**Nasıl Çalışır:** Tüm modeller aynı göreve (`kanishk/EndlessTerminals`) tabi tutulur ve sonuçlar tablo halinde sunulur.
```bash
python 02_multi_provider.py
```

### 3. `03_async_evaluation.py` 🟣
**Ne yapar?** Birden fazla görevi (batch task) aynı anda hızlıca değerlendirmek için tasarlanmıştır. DeepSeek v3.2 modeli kullanılır.
**Nasıl Çalışır:** `AsyncOpenReward` client'ını ve Python'ın `asyncio.gather` yapısını kullanarak, eşzamanlı session'lar başlatır. Progress bar ile görevlerin anlık durumu izlenir ve sonunda ortalama reward ile başarı oranı hesaplanır. 
```bash
python 03_async_evaluation.py
```

### 4. `04_training_setup.py` 🟠
**Ne yapar?** Reinforcement Learning (GRPO, PPO) eğitimi için **Tinker** ve **Slime** framework'lerine uygun YAML config dosyalarını otomatik üretir.
**Nasıl Çalışır:** İnteraktif bir menü sunarak eğitim ortamını, kullanılacak base modeli ve hiperparametreleri (LoRA rank, batch size vb.) seçmenizi sağlar. Ardından `configs/` dizini altına `.yaml` dosyalarını yazar.
```bash
python 04_training_setup.py
```

---

## 🧠 Nasıl Eğitirsiniz? (Tinker ve Slime)

OpenReward sadece değerlendirme için değil, aynı zamanda ajan eğitimi için de kullanılır. Model eğitmek istiyorsanız:

1. [OpenReward Cookbook](https://github.com/OpenRewardAI/openreward-cookbook) repository'sini clone'layın.
2. `04_training_setup.py` ile `tinker_config.yaml` dosyanızı oluşturun.
3. Tinker framework içinden main dosyayı çalıştırın:
```bash
python main.py --config_path /path/to/your/configs/tinker_config.yaml
```

---
*Daha fazla bilgi için: [docs.openreward.ai](https://docs.openreward.ai)*
