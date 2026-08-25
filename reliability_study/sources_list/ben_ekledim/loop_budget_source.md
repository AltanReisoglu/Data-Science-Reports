# Ne Zaman Durması Gerektiğini Bilen Ajanlar
Pratik döngü korumaları — bütçeler, zaman aşımları, ilerleme kontrolleri ve "çekimser kalma" (abstain) yolları — böylece ajanınız token, araç ve parayı kendi etrafında dönerek harcamaz.

Her ajan demosu harika görünür... ta ki patlayana kadar.
Bir gün "yardımsever" ajanınız bir döngüye sıkışır:
* Sürekli aynı araç çağrısını tekrar dener,
* Sürekli aynı dokümanları yeniden okur,
* Kendi kendine sürekli "adım adım düşün" der,
* Ve sanki limitsiz bir kredi kartı bulmuş gibi bütçenizi harcamaya devam eder.

Gerçekçi olalım: Ajanlar sadece yanıldıkları için başarısız olmazlar.
Israrcı oldukları için başarısız olurlar.
Ve ısrar pahalıdır.

Bu makale, sizin başında beklemenize gerek kalmadan, bilerek ve isteyerek durabilen bir ajan inşa etmek hakkındadır. Token'larınızı, araç kotalarınızı, API'lerinizi ve akıl sağlığınızı koruyan döngü korumalarından (loop guards) bahsedeceğiz.
Çünkü durma koşulları olmayan "otonomi", sadece bir yangındır.

## Ajanlar en başta neden döngüye girer?
Ajanlar, insanların döngüye girmesiyle aynı nedenden dolayı döngüye girerler: ilerleme kaydetmediklerinin farkında değildirler.
Çoğu döngü şunlardan birinden kaynaklanır:

**1) Araç, ajanın beklediği yanıtı döndürmedi:** API kısmi veri döndürmüş olabilir, veritabanı sorgusu boş dönmüş olabilir, arama sonuçları gürültülü olabilir veya araç zaman aşımına uğrayıp ajan tekrar deniyor olabilir.
**2) Hedef yeterince belirtilmemiş:** "Şunu düzelt", bir gereksinim spesifikasyonu değildir. Başarı kriteri belirsizse, ajan saatlerce aynı paragrafı yeniden yazan bir öğrenci gibi farklı açılardan denemeye devam eder.
**3) Ajan, hareket etmeyi ilerleme ile karıştırır:** Loglama yapar, arama yapar, tekrar dener. Üretken hisseder. Ancak dünyanın durumunu anlamlı bir şekilde değiştirmiyordur.

Bu yüzden çözüm "daha akıllı istemler (prompts)" değildir.
Çözüm, döngüleri tespit eden ve onları durduran bariyerlerdir (guardrails).

## Döngü koruması zihniyeti: Bütçe bir özelliktir
Bütçeye sonradan akla gelen bir şey değil, birinci sınıf bir ürün kısıtlaması gibi yaklaşın. Güvenli bir ajan her zaman şunları bilir:

* Kaç adım atabileceğini
* Ne kadar zaman harcayabileceğini
* Kaç tane araç çağrısı yapabileceğini
* "İlerlemenin" neye benzediğini
* Konuyu ne zaman bir insana devredeceğini

İşin sırrı budur: Durabilmek bir yetenektir.

## Gerçekten para tasarrufu sağlayan 5 döngü koruması

### Koruma 1: Kesin Sınırlar (adım bütçesi + araç bütçesi)
Bu en basit ve en etkili korumadır. Şunları belirleyin:

* Maksimum mantık yürütme adımı (kendi iç döngüleri)
* Maksimum araç çağrısı
* Araç başına maksimum tekrar deneme (retry)
* Maksimum toplam token (veya maksimum maliyet)

**Örnek politika:** toplam adım ≤ 12, araç çağrıları ≤ 8, araç başına deneme ≤ 2, toplam süre ≤ 60 saniye.
Ajan bir sınıra ulaştığında şunları döndürmelidir: Ne denediğini, ne öğrendiğini, onu neyin engellediğini ve bir sonraki adım için ne önerdiğini. Bu, "sonsuz döngüyü" "faydalı kısmi sonuca" dönüştürür.

### Koruma 2: Üstel Gecikme + Jitter (Tekrar denemeler için)
Eğer bir araç tutarsız çalışıyorsa (flaky), körü körüne yapılan tekrar denemeler bir istek fırtınasına neden olur. Bunun yerine: Birkaç kez deneyin, bekleme süresini giderek artırın, rastgelelik (jitter) ekleyin ve sonra durun. 
Genel kural: Aynı çağrı iki kez başarısız olursa, aksi kanıtlanana kadar bunun geçici bir hata olmadığını varsayın. Ve asla riskli yan etkileri olan işlemleri (ödemeler, e-postalar) "idempotency" (aynı işlemin tekrar tekrar yapılmasının sonucu değiştirmemesi durumu) olmadan tekrar denemeyin.

### Koruma 3: İlerleme Kontrolleri ("Hareket ediyor muyuz?" testi)
Bu, değeri en az bilinen korumadır. Görev için bir ilerleme metriği tanımlayın:

* Çekilen benzersiz kaynak sayısı
* Çıkarılan yeni varlık (entity) sayısı
* Hata sayısındaki azalma
* Kod farkının (diff) küçülmesi
* Test hatalarının azalması
* Belirsizlik skorunun düşmesi

Ardından şunu uygulayın: İlerleme N adım sonra iyileşmediyse, dur veya strateji değiştir. Basit bir ilerleme kuralı: Çıktıların `state_hash`'ini (durum özetini) takip edin. Eğer bu hash 2-3 döngü boyunca değişmiyorsa, döngüye girmişsinizdir.

### Koruma 4: Döngü Parmak İzleri (Tekrarlayan kalıpları tespit et)
Ajanlar genellikle aynı diziyi tekrarlar: `arama → özetleme → arama → özetleme`. Ya da: `API çağrısı → zaman aşımı → tekrar dene → zaman aşımı`. Bunu, eylemlerin hafif bir "parmak izi" (fingerprint) ile tespit edebilirsiniz. Son K sayıdaki eylemi saklayın (araç adı, temel argümanlar, hata kodları, yanıt türü). Aynı imzanın tekrarlandığını görürseniz, şalteri indirin (circuit breaker): Farklı bir araca geçin, açıklayıcı bir soru sorun veya işi insana devredin.

### Koruma 5: "Çekimser Kalma" Yolu (Dur ve bilgi iste)
Bazı döngüler, ajanın elinde kilit bir girdi eksik olduğu için gerçekleşir. Sonsuza kadar tahmin yürütmek yerine, ajan çekimser kalmalıdır.
Örnekler: "Hangi ortam: test (staging) mi yoksa canlı (prod) mı?", "Hangi müşteri ID'si?", "Silmek mi yoksa devre dışı bırakmak mı istiyorsunuz?".
Bu daha yavaş hissettirebilir, ancak 40 adım boyunca yanlış tahmin yürütmekten daha ucuzdur.

## Mimari Akış: Bir kontrol düzlemi olarak döngü korumaları

```text
┌───────────────┐
│ Kullanıcı İst. │
└───────┬───────┘
        v
┌────────────────────────┐
│ Plan + Başarı Kriteri  │
└───────┬────────────────┘
        v
┌────────────────────────┐
│ Koruma Kontrolcüsü     │
│ - adım/araç bütçeleri  │
│ - tekrar deneme kuralları│
│ - ilerleme takipçisi   │
│ - döngü parmak izi     │
└───────┬────────────────┘
        v
┌────────────────────────┐
│ Çalıştır (araç/kod/sql)│
└───────┬────────────────┘
        v
┌────────────────────────┐
│ İlerlemeyi Değerlendir │
│ - durum değişti mi?    │
│ - hedefe yaklaştı mı?  │
└───────┬────────────────┘
  evet  │             hayır
        │              v
        │      ┌──────────────────┐
        │      │ Strateji Değiştir│
        │      │ Sor / Dur        │
        │      └──────────────────┘
        v
┌────────────────────────┐
│ Son Cevap + Kanıtlar   │
└────────────────────────┘
```
"Koruma Kontrolcüsü" süslü bir şey değildir. Sadece ajanın kontrolden çıkmasına izin vermeyi reddeden küçük, deterministik (kuralları belli) bir katmandır.

---

## Örnek Olay: Bir aylık bütçeyi yiyen "arama sarmalı"
Bir ekip, müşteri sorularını yanıtlamak için bir ajan inşa etti. Ajan bir cevap bulamadığında aramaya devam etti: dokümanlarda ara, web'de ara, dokümanlarda tekrar ara, tekrar özetle... Döngü çökmedi. Sadece... fatura yazdı.
**Çözüm:** Soru başına maksimum 3 arama (kesin sınır), "bulunan yeni benzersiz kaynaklar" (ilerleme metriği) ve 2 aramadan sonra yeni kaynak yoksa kullanıcıdan açıklayıcı soru isteme (çekimser kalma). 
**Sonuç:** Yanıtlar hızlandı, maliyetler düştü ve ajan öngörülebilir hale geldi.

## Altın Kural: Onurunla Dur
Ajanınız durduğunda, faydalı bir şekilde durmalıdır. "İyi bir duruş" yanıtı şunları içerir:
* Ne denediği (kısaca)
* Ne bulduğu (eğer bir şey bulduysa)
* Neden durduğu (bütçe/ilerleme/izinler)
* En iyi bir sonraki eylem (sor/devret/manuel)

Bu güven inşa eder. Ve "pes etti" algısını engeller. Durmak başarısızlık değildir. Durmak kontroldür.

## Sonuç: Otonomi pahalıdır — onu koruyun
Ajanlar inşa ediyorsanız soru "döngüye girebilir mi?" değildir. Girecektir. Soru şudur: Döngüleri ucuz, güvenli ve görünür kılan döngü korumalarınız var mı?

---

## Kodun Açıklaması ve Yorum Satırları

Yazarın verdiği Python kodu, yukarıda anlatılan **Koruma Kontrolcüsü** (Loop Guard) katmanının çok temiz ve temel bir örneğidir. 

```python
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ajanın sınırlarını belirlediğimiz veri sınıfı (Bütçeler)
@dataclass
class GuardLimits:
    max_steps: int = 12                 # Maksimum iç adım sayısı
    max_tool_calls: int = 8             # Toplamda bir aracı maksimum kaç kez çağırabileceği
    max_retries_per_tool: int = 2       # Aynı aracın başarısız olduğunda en fazla kaç kez deneneceği
    max_seconds: int = 60               # Ajanın işlemi tamamlaması için verilen maksimum süre

# Ajanın mevcut durumunu tuttuğumuz veri sınıfı (Sayaçlar ve Geçmiş)
@dataclass
class GuardState:
    step: int = 0                                       # Atılan adım sayısı
    tool_calls: int = 0                                 # Yapılan araç çağrısı sayısı
    started_at: float = field(default_factory=time.time)# İşlemin başlama zamanı
    retries: Dict[str, int] = field(default_factory=dict) # Hangi aracın kaç kez hata verdiğini tutan sözlük
    last_state_hashes: List[str] = field(default_factory=list) # Son durumların hash'lenmiş hali (ilerleme kontrolü için)
    last_actions: List[str] = field(default_factory=list) # Son yapılan eylemlerin parmak izleri (tekrarları bulmak için)

# Ajandaki verilerin/dünyanın durumunun değişip değişmediğini anlamak için hash üreten fonksiyon
def state_hash(obj: Any) -> str:
    # Sadece önemli "durum" özetini hash'ler, tüm ham logları dahil etmeyiz.
    # obj'yi string'e çevirip utf-8 formatında kodlarız, sonra SHA256 ile kısaltılmış bir kimlik (hash) çıkarırız.
    s = str(obj).encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()[:16]

# Ajanın tam olarak ne yaptığının parmak izini çıkaran fonksiyon
def action_fingerprint(tool: str, args: Dict[str, Any], outcome: str) -> str:
    # Hangi araç kullanıldı + Hangi argümanlar verildi + Sonuç ne oldu?
    # Bunları birleştirip bir şifre (hash) oluştururuz. Eğer ajan aynı şeyleri tekrar ederse aynı şifre üretilecektir.
    key = f"{tool}|{sorted(args.items())}|{outcome}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

# Tüm güvenlik kontrollerini yöneten ana sınıf
class LoopGuard:
    def __init__(self, limits: GuardLimits):
        self.limits = limits       # Başlangıçta belirlenen sınırlarımız
        self.state = GuardState()  # Saymaya başladığımız boş durumumuz

    # 1. KORUMA: Bütçe Kontrolü. Ajan adıma başlamadan önce çağrılır.
    def check_budget(self) -> Optional[str]:
        if self.state.step >= self.limits.max_steps:
            return "Adım bütçesine ulaşıldı."
        if self.state.tool_calls >= self.limits.max_tool_calls:
            return "Araç çağrısı bütçesine ulaşıldı."
        if (time.time() - self.state.started_at) > self.limits.max_seconds:
            return "Zaman bütçesine ulaşıldı."
        return None # Eğer hiçbir sınır aşılmadıysa, sorun yok (None dön)

    # 3. KORUMA: İlerleme Kaydediliyor mu? Her iterasyondan sonra çağrılır.
    def record_progress(self, summary_state: Any) -> bool:
        h = state_hash(summary_state) # Mevcut durumun özet şifresini al
        self.state.last_state_hashes.append(h) # Listeye ekle
        self.state.last_state_hashes = self.state.last_state_hashes[-3:] # Sadece son 3 durumu hafızada tut

        # Eğer son 3 durum hafızada var VE bu 3 durum da birbiriyle tamamen aynıysa (set uzunluğu 1 ise)
        # Bu, ajanın 3 adımdır hiçbir şeyi değiştirmediğini (döngüye girdiğini) gösterir.
        if len(self.state.last_state_hashes) == 3 and len(set(self.state.last_state_hashes)) == 1:
            return False # İlerleme durdu
        return True # İlerleme var

    # İşlem geçmişini ve hataları kaydettiğimiz fonksiyon
    def record_tool_call(self, tool: str, args: Dict[str, Any], outcome: str):
        self.state.tool_calls += 1 # Toplam araç kullanımını artır
        fp = action_fingerprint(tool, args, outcome) # Eylemin parmak izini çıkar
        self.state.last_actions.append(fp)
        self.state.last_actions = self.state.last_actions[-6:] # Sadece son 6 eylemi hafızada tut

        # Eğer araç hata verdiyse veya zaman aşımına uğradıysa, tekrar deneme (retry) sayacını o araç için artır.
        if outcome in ("timeout", "error"):
            self.state.retries[tool] = self.state.retries.get(tool, 0) + 1

    # 2. KORUMA: Bir araç tekrar denenmeli mi?
    def should_retry(self, tool: str) -> bool:
        # Eğer bu aracın hata sayısı, limitimizin altındaysa True (tekrar dene), aksi halde False döner.
        return self.state.retries.get(tool, 0) < self.limits.max_retries_per_tool

    # 4. KORUMA: Tekrarlayan eylem kalıbı (Parmak izi) algılayıcı
    def detect_repeat_pattern(self) -> bool:
        # Eğer son 6 eylem hafızadaysa VE bu 6 eylem sadece 1 veya 2 benzersiz (unique) parmak izinden oluşuyorsa,
        # Ajan aynı 1-2 eylemi sürekli birbirinin ardına tekrarlıyor demektir. (Örn: Arama -> Hata -> Arama -> Hata)
        if len(self.state.last_actions) == 6 and len(set(self.state.last_actions)) <= 2:
            return True # Döngü tespit edildi
        return False # Her şey normal

    # Ajanın adım sayacını bir artıran yardımcı fonksiyon
    def next_step(self):
        self.state.step += 1
```

**Kısaca Kodun Çalışma Mantığı:**
Bu kod bloğu, bir yapay zeka ajanının içine yerleştirilen bir "bekçi" gibidir. Ajan her yeni adım atacağında `check_budget()` ile sınırları aşıp aşmadığı kontrol edilir. Ajan bir araç kullandığında `record_tool_call()` ile ne yaptığı kaydedilir ve hata alıp almadığına bakılır. İşlem bitiminde ise `record_progress()` ile *gerçekten* dişe dokunur bir ilerleme (state değişikliği) yapıp yapmadığı test edilir. Eğer ajan aynı şeyleri yapıp duruyorsa, bekçi ajanı durdurur ve faturanızın kabarmasını engeller.