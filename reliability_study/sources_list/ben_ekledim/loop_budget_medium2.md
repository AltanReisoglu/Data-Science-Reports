Spark
beta
Yeni sohbet
Sohbetlerde arama yapın
Resimler
Videolar
Kitaplık
Yeni not defteri
Neural Computer: The Emergence of a New Machine Form
Untitled notebook
Tüm not defterleri
Yapay Zeka Ajanlarında Döngü Korumaları
Hackathon İçin Yatırım Odaklı Şirket Fikirleri
Yapay Zeka ile Finansal Uyumluluk Altyapısı
Kurumsal VPN Bağlantısı Nasıl Yapılır
Kurumsal VPN Bağlantısı Nasıl Yapılır
Ajan İletişim Modelleri: Doğrudan vs. Pub/Sub
Ajan İletişimi: Doğrudan vs. Pub/Sub
Çalınan Telefon İçin Yapılacaklar
Çalınan Telefonu Bulma ve Güvenlik Adımları
Turkcell Ek Fatura Görüntüleme Yöntemleri
Hackathon Hazırlık ve Sunum Kriterleri
2.5" SATA SSD Laptop Uyumluluğu
N-gram Modelleri: Tanım ve Kullanım Alanları
N-Gram Modelleri: Doğal Dil İşleme
Lojistik Staj Defteri İçin 10 Günlük Taslak
Maaş Ödemesi Gecikmesi İçin Mesaj Önerileri
M4 Metrodan Metrobüse Geçiş
Staj Süreci ve Evrak Teslim Rehberi
Özel Servis Durağı Bilgisi Talebi
Staj İçin Gerekli Belgeler ve İşlemler
Staj Hesabı İçin Şube Şartı
YouTube Kapağı Yazım Hataları Düzeltme
YouTube Telif Hakkı İhtarı Çözüm Önerileri
Linux'ta Epic Games Among Us Oynama
GRUB Menüsü ve Ubuntu Başlatma Sorunu
Google Play Uygulama Geliştirme Şirket Şart Mı?
bu konuyu anlat resimden refensla
Ev İnterneti Sağlayıcıları Karşılaştırması
Görüntü Üretme Yeteneği Sınırları
Gizli Çekim: Bilgisayar Başında Asker
E-posta Arama Sonucu
Abonelik İptali ve Para İadesi
Gemini ile sohbet
Agents That Know When to Stop

Practical loop guards — budgets, timeouts, progress checks, and “abstain” lanes — so your agent doesn’t burn tokens, tools, and money in circles.



Praxen

Following



6 min read

·

Feb 1, 2026

15













Press enter or click to view image in full size





Prevent runaway agent loops with loop guards: step budgets, timeouts, progress metrics, idempotency, and circuit breakers that protect cost and reliability.

Every agent demo looks great… until it doesn’t.

One day your “helpful” agent gets stuck in a loop:

keeps retrying the same tool call,

keeps re-reading the same docs,

keeps asking itself to “think step-by-step,”

and keeps spending your budget like it found an infinite credit card.

Let’s be real: agents don’t just fail by being wrong.

They fail by being persistent.

And persistence is expensive.

This article is about building an agent that can stop — on purpose — without you babysitting it. We’ll cover loop guards that protect your tokens, your tool quotas, your APIs, and your sanity.

Because “autonomy” without stop conditions is just a fire.

Why agents loop in the first place

Agents loop for the same reason humans do: they don’t know they’re not making progress.

Most loops come from one of these:

1) The tool didn’t return what the agent expected

the API responded with partial data

the DB query returned empty results

the search results were noisy

the tool timed out and the agent tries again

2) The objective is underspecified

“Fix it” is not a spec.



If the success criteria is vague, the agent keeps trying different angles like a student rewriting the same paragraph for hours.

3) The agent confuses motion with progress

It logs, it searches, it retries. It feels productive.

But it’s not changing the state of the world in a meaningful way.

So the solution is not “smarter prompts.”

It’s guardrails that detect loops and stop them.

The loop guard mindset: budget is a feature

Treat budget like a first-class product constraint, not an afterthought.



A safe agent always knows:

how many steps it can take

how much time it can spend

how many tool calls it can make

what “progress” looks like

when to escalate to a human

This is the secret: stopping is a capability.

The 5 loop guards that save real money

Guard 1: Hard caps (step budget + tool budget)

This is the simplest and most effective guard.



Set:

max reasoning steps (internal turns)

max tool calls

max retries per tool

max total tokens (or max cost)

Example policy:

total steps ≤ 12

tool calls ≤ 8

retries per tool ≤ 2

total time ≤ 60 seconds

When the agent hits a cap, it must return:

what it tried

what it learned

what’s blocking it

what it recommends next

This turns “infinite loop” into “useful partial result.”

Guard 2: Exponential backoff + jitter (for retries)

If a tool is flaky, blind retries create a retry storm.



Instead:

retry a few times

with increasing wait

with randomness (jitter)

then stop

Rule of thumb:

If the same call fails twice, assume it’s not transient until proven otherwise.

And never retry risky side effects (payments, emails) without idempotency.

Guard 3: Progress checks (the “are we moving?” test)

This is the most underrated guard.



Define a progress metric for the task:

number of unique sources retrieved

number of new entities extracted

reduction in error count

diff size shrinking

test failures decreasing

uncertainty score going down

Then enforce:

If progress hasn’t improved after N steps, stop or change strategy.

A simple progress rule:

track a state_hash of key outputs

if the hash doesn’t change across 2–3 iterations, you’re looping

If you’re repeatedly generating the same plan, you’re not thinking. You’re circling.

Guard 4: Loop fingerprints (detect repeated patterns)

Agents often repeat the same sequence:

search → summarize → search → summarize → search → summarize

Or:

call API → timeout → retry → timeout → retry

You can detect this with a lightweight “fingerprint” of actions.

Store the last K actions like:

tool name

key arguments

error codes

response type

If you see the same signature repeating, trigger a circuit breaker:

switch to a different tool lane

ask a clarifying question

escalate to human

Guard 5: The “abstain lane” (stop + ask for info)

Some loops happen because the agent lacks a key input.



Instead of guessing forever, the agent should abstain.

Examples:

“Which environment: staging or prod?”

“Which customer ID?”

“Do you want to delete or disable?”

“What’s the allowed spend limit for this run?”

This feels slower, but it’s cheaper than guessing wrong for 40 steps.

A good agent is confident enough to say:

“I can’t proceed safely without X.”

Architecture flow: loop guards as a control plane

Here’s a clean execution loop with guardrails in the right places:



┌───────────────┐

│ User Request │

└───────┬───────┘

v

┌────────────────────────┐

│ Plan + Success Criteria │

└───────┬────────────────┘

v

┌────────────────────────┐

│ Guard Controller │

│ - step/tool budgets │

│ - retry rules │

│ - progress tracker │

│ - loop fingerprinting │

└───────┬────────────────┘

v

┌────────────────────────┐

│ Execute (tool/code/sql) │

└───────┬────────────────┘

v

┌────────────────────────┐

│ Evaluate Progress │

│ - changed state? │

│ - closer to goal? │

└───────┬────────────────┘

yes │ no

│ v

│ ┌──────────────────┐

│ │ Change Strategy / │

│ │ Ask / Stop │

│ └──────────────────┘

v

┌────────────────────────┐

│ Final Answer + Evidence │

└────────────────────────┘

The “Guard Controller” isn’t fancy.

It’s just a small deterministic layer that refuses to let the agent spiral.

Code sample: a minimal loop guard controller

Below is a lightweight example you can adapt to most agent frameworks:



import time

import hashlib

from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional





@dataclass

class GuardLimits:

max_steps: int = 12

max_tool_calls: int = 8

max_retries_per_tool: int = 2

max_seconds: int = 60





@dataclass

class GuardState:

step: int = 0

tool_calls: int = 0

started_at: float = field(default_factory=time.time)

retries: Dict[str, int] = field(default_factory=dict)

last_state_hashes: List[str] = field(default_factory=list)

last_actions: List[str] = field(default_factory=list)





def state_hash(obj: Any) -> str:

# Hash only the important “state” summary, not full raw logs

s = str(obj).encode("utf-8", errors="ignore")

return hashlib.sha256(s).hexdigest()[:16]





def action_fingerprint(tool: str, args: Dict[str, Any], outcome: str) -> str:

key = f"{tool}|{sorted(args.items())}|{outcome}"

return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]





class LoopGuard:

def __init__(self, limits: GuardLimits):

self.limits = limits

self.state = GuardState()



def check_budget(self) -> Optional[str]:

if self.state.step >= self.limits.max_steps:

return "Step budget reached."

if self.state.tool_calls >= self.limits.max_tool_calls:

return "Tool call budget reached."

if (time.time() - self.state.started_at) > self.limits.max_seconds:

return "Time budget reached."

return None



def record_progress(self, summary_state: Any) -> bool:

h = state_hash(summary_state)

self.state.last_state_hashes.append(h)

self.state.last_state_hashes = self.state.last_state_hashes[-3:]



# If state hasn’t changed across 3 checkpoints, likely looping

if len(self.state.last_state_hashes) == 3 and len(set(self.state.last_state_hashes)) == 1:

return False

return True



def record_tool_call(self, tool: str, args: Dict[str, Any], outcome: str):

self.state.tool_calls += 1

fp = action_fingerprint(tool, args, outcome)

self.state.last_actions.append(fp)

self.state.last_actions = self.state.last_actions[-6:]



# Track retries by tool

if outcome in ("timeout", "error"):

self.state.retries[tool] = self.state.retries.get(tool, 0) + 1



def should_retry(self, tool: str) -> bool:

return self.state.retries.get(tool, 0) < self.limits.max_retries_per_tool



def detect_repeat_pattern(self) -> bool:

# If last 6 actions collapse to 2 unique fingerprints, it’s looping

if len(self.state.last_actions) == 6 and len(set(self.state.last_actions)) <= 2:

return True

return False



def next_step(self):

self.state.step += 1

How you use it:

before each step → check_budget()

after each tool call → record_tool_call()

after each iteration → record_progress()

if progress fails or patterns repeat → change strategy / ask / stop

This isn’t about perfect detection.

It’s about refusing to let “infinite” happen.

Case study: the “search spiral” that ate a month of budget

A team built an agent to answer customer questions. When it couldn’t find an answer, it kept searching:

docs search

web search

docs search again

summarize again

The loop didn’t crash. It just… billed.

Fix:

hard cap: max 3 searches per question

progress metric: “new unique sources found”

if no new sources after 2 searches → ask user a clarifying question

fallback: route to support ticket creation

Result: answers got faster, costs dropped, and the agent became predictable.

The golden rule: stop with dignity

When your agent stops, it should stop usefully.



A “good stop” response includes:

what it attempted (brief)

what it found (if anything)

why it stopped (budget/progress/permissions)

the next best action (ask/handoff/manual)

This builds trust.

And it prevents the “it gave up” perception.

Stopping isn’t failure.

Stopping is control.

Conclusion: autonomy is expensive — guard it

If you’re building agents, the question isn’t “can it loop?”



It will.

The question is: do you have loop guards that make loops cheap, safe, and visible?

Add budgets. Add progress checks. Add repeat detection. Add abstain lanes.

And watch your reliability — and costs — improve overnight.

If you want, comment with the tool stack your agent uses (search, SQL, code, webhooks), and I’ll suggest a minimal loop-guard policy that fits your workload and budget. Follow for more “agent reliability” patterns that keep autonomy from turning into bills.





bu konuyu bana anlat.daha doğrusu kelimesi kelimesin eçevir.kodları da anlat ek olarak yorum satırlarıyla

Bu makale, otonom yapay zeka ajanlarının (AI agents) karşılaştığı en büyük mühendislik problemlerinden birini ele alıyor: Sonsuz döngülere girmek ve bütçeyi (token/para) tüketmek. Bir ajan beklediği veriyi alamadığında veya ne yapacağını tam bilemediğinde aynı işlemleri defalarca tekrar edebilir. Makale, bu sorunu çözmek için ajana "durmayı" öğreten 5 pratik güvenlik önlemini (loop guards) anlatıyor.

İşte makalenin aslına sadık kalınarak yapılan tam çevirisi ve ardından kodun detaylı açıklaması:

Ne Zaman Durması Gerektiğini Bilen Ajanlar
Pratik döngü korumaları — bütçeler, zaman aşımları, ilerleme kontrolleri ve "çekimser kalma" (abstain) yolları — böylece ajanınız token, araç ve parayı kendi etrafında dönerek harcamaz.

Her ajan demosu harika görünür... ta ki patlayana kadar.
Bir gün "yardımsever" ajanınız bir döngüye sıkışır:

Sürekli aynı araç çağrısını tekrar dener,

Sürekli aynı dokümanları yeniden okur,

Kendi kendine sürekli "adım adım düşün" der,

Ve sanki limitsiz bir kredi kartı bulmuş gibi bütçenizi harcamaya devam eder.

Gerçekçi olalım: Ajanlar sadece yanıldıkları için başarısız olmazlar.
Israrcı oldukları için başarısız olurlar.
Ve ısrar pahalıdır.

Bu makale, sizin başında beklemenize gerek kalmadan, bilerek ve isteyerek durabilen bir ajan inşa etmek hakkındadır. Token'larınızı, araç kotalarınızı, API'lerinizi ve akıl sağlığınızı koruyan döngü korumalarından (loop guards) bahsedeceğiz.
Çünkü durma koşulları olmayan "otonomi", sadece bir yangındır.

Ajanlar en başta neden döngüye girer?
Ajanlar, insanların döngüye girmesiyle aynı nedenden dolayı döngüye girerler: ilerleme kaydetmediklerinin farkında değildirler.
Çoğu döngü şunlardan birinden kaynaklanır:

1) Araç, ajanın beklediği yanıtı döndürmedi: API kısmi veri döndürmüş olabilir, veritabanı sorgusu boş dönmüş olabilir, arama sonuçları gürültülü olabilir veya araç zaman aşımına uğrayıp ajan tekrar deniyor olabilir.
2) Hedef yeterince belirtilmemiş: "Şunu düzelt", bir gereksinim spesifikasyonu değildir. Başarı kriteri belirsizse, ajan saatlerce aynı paragrafı yeniden yazan bir öğrenci gibi farklı açılardan denemeye devam eder.
3) Ajan, hareket etmeyi ilerleme ile karıştırır: Loglama yapar, arama yapar, tekrar dener. Üretken hisseder. Ancak dünyanın durumunu anlamlı bir şekilde değiştirmiyordur.

Bu yüzden çözüm "daha akıllı istemler (prompts)" değildir.
Çözüm, döngüleri tespit eden ve onları durduran bariyerlerdir (guardrails).

Döngü koruması zihniyeti: Bütçe bir özelliktir
Bütçeye sonradan akla gelen bir şey değil, birinci sınıf bir ürün kısıtlaması gibi yaklaşın. Güvenli bir ajan her zaman şunları bilir:

Kaç adım atabileceğini

Ne kadar zaman harcayabileceğini

Kaç tane araç çağrısı yapabileceğini

"İlerlemenin" neye benzediğini

Konuyu ne zaman bir insana devredeceğini

İşin sırrı budur: Durabilmek bir yetenektir.

Gerçekten para tasarrufu sağlayan 5 döngü koruması
Koruma 1: Kesin Sınırlar (adım bütçesi + araç bütçesi)
Bu en basit ve en etkili korumadır. Şunları belirleyin:

Maksimum mantık yürütme adımı (kendi iç döngüleri)

Maksimum araç çağrısı

Araç başına maksimum tekrar deneme (retry)

Maksimum toplam token (veya maksimum maliyet)

Örnek politika: toplam adım ≤ 12, araç çağrıları ≤ 8, araç başına deneme ≤ 2, toplam süre ≤ 60 saniye.
Ajan bir sınıra ulaştığında şunları döndürmelidir: Ne denediğini, ne öğrendiğini, onu neyin engellediğini ve bir sonraki adım için ne önerdiğini. Bu, "sonsuz döngüyü" "faydalı kısmi sonuca" dönüştürür.

Koruma 2: Üstel Gecikme + Jitter (Tekrar denemeler için)
Eğer bir araç tutarsız çalışıyorsa (flaky), körü körüne yapılan tekrar denemeler bir istek fırtınasına neden olur. Bunun yerine: Birkaç kez deneyin, bekleme süresini giderek artırın, rastgelelik (jitter) ekleyin ve sonra durun.
Genel kural: Aynı çağrı iki kez başarısız olursa, aksi kanıtlanana kadar bunun geçici bir hata olmadığını varsayın. Ve asla riskli yan etkileri olan işlemleri (ödemeler, e-postalar) "idempotency" (aynı işlemin tekrar tekrar yapılmasının sonucu değiştirmemesi durumu) olmadan tekrar denemeyin.

Koruma 3: İlerleme Kontrolleri ("Hareket ediyor muyuz?" testi)
Bu, değeri en az bilinen korumadır. Görev için bir ilerleme metriği tanımlayın:

Çekilen benzersiz kaynak sayısı

Çıkarılan yeni varlık (entity) sayısı

Hata sayısındaki azalma

Kod farkının (diff) küçülmesi

Test hatalarının azalması

Belirsizlik skorunun düşmesi

Ardından şunu uygulayın: İlerleme N adım sonra iyileşmediyse, dur veya strateji değiştir. Basit bir ilerleme kuralı: Çıktıların state_hash'ini (durum özetini) takip edin. Eğer bu hash 2-3 döngü boyunca değişmiyorsa, döngüye girmişsinizdir.

Koruma 4: Döngü Parmak İzleri (Tekrarlayan kalıpları tespit et)
Ajanlar genellikle aynı diziyi tekrarlar: arama → özetleme → arama → özetleme. Ya da: API çağrısı → zaman aşımı → tekrar dene → zaman aşımı. Bunu, eylemlerin hafif bir "parmak izi" (fingerprint) ile tespit edebilirsiniz. Son K sayıdaki eylemi saklayın (araç adı, temel argümanlar, hata kodları, yanıt türü). Aynı imzanın tekrarlandığını görürseniz, şalteri indirin (circuit breaker): Farklı bir araca geçin, açıklayıcı bir soru sorun veya işi insana devredin.

Koruma 5: "Çekimser Kalma" Yolu (Dur ve bilgi iste)
Bazı döngüler, ajanın elinde kilit bir girdi eksik olduğu için gerçekleşir. Sonsuza kadar tahmin yürütmek yerine, ajan çekimser kalmalıdır.
Örnekler: "Hangi ortam: test (staging) mi yoksa canlı (prod) mı?", "Hangi müşteri ID'si?", "Silmek mi yoksa devre dışı bırakmak mı istiyorsunuz?".
Bu daha yavaş hissettirebilir, ancak 40 adım boyunca yanlış tahmin yürütmekten daha ucuzdur.

Mimari Akış: Bir kontrol düzlemi olarak döngü korumaları
Plaintext
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
"Koruma Kontrolcüsü" süslü bir şey değildir. Sadece ajanın kontrolden çıkmasına izin vermeyi reddeden küçük, deterministik (kuralları belli) bir katmandır.

Örnek Olay: Bir aylık bütçeyi yiyen "arama sarmalı"
Bir ekip, müşteri sorularını yanıtlamak için bir ajan inşa etti. Ajan bir cevap bulamadığında aramaya devam etti: dokümanlarda ara, web'de ara, dokümanlarda tekrar ara, tekrar özetle... Döngü çökmedi. Sadece... fatura yazdı.
Çözüm: Soru başına maksimum 3 arama (kesin sınır), "bulunan yeni benzersiz kaynaklar" (ilerleme metriği) ve 2 aramadan sonra yeni kaynak yoksa kullanıcıdan açıklayıcı soru isteme (çekimser kalma).
Sonuç: Yanıtlar hızlandı, maliyetler düştü ve ajan öngörülebilir hale geldi.

Altın Kural: Onurunla Dur
Ajanınız durduğunda, faydalı bir şekilde durmalıdır. "İyi bir duruş" yanıtı şunları içerir:

Ne denediği (kısaca)

Ne bulduğu (eğer bir şey bulduysa)

Neden durduğu (bütçe/ilerleme/izinler)

En iyi bir sonraki eylem (sor/devret/manuel)

Bu güven inşa eder. Ve "pes etti" algısını engeller. Durmak başarısızlık değildir. Durmak kontroldür.

Sonuç: Otonomi pahalıdır — onu koruyun
Ajanlar inşa ediyorsanız soru "döngüye girebilir mi?" değildir. Girecektir. Soru şudur: Döngüleri ucuz, güvenli ve görünür kılan döngü korumalarınız var mı?

Kodun Açıklaması ve Yorum Satırları
Yazarın verdiği Python kodu, yukarıda anlatılan Koruma Kontrolcüsü (Loop Guard) katmanının çok temiz ve temel bir örneğidir.

Python
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
Kısaca Kodun Çalışma Mantığı:
Bu kod bloğu, bir yapay zeka ajanının içine yerleştirilen bir "bekçi" gibidir. Ajan her yeni adım atacağında check_budget() ile sınırları aşıp aşmadığı kontrol edilir. Ajan bir araç kullandığında record_tool_call() ile ne yaptığı kaydedilir ve hata alıp almadığına bakılır. İşlem bitiminde ise record_progress() ile gerçekten dişe dokunur bir ilerleme (state değişikliği) yapıp yapmadığı test edilir. Eğer ajan aynı şeyleri yapıp duruyorsa, bekçi ajanı durdurur ve faturanızın kabarmasını engeller.

tüm yazdıklarını bir mdye geçir

"Koruma Kontrolcüsü" süslü bir şey değildir. Sadece ajanın kontrolden çıkmasına izin vermeyi reddeden küçük, deterministik (kuralları belli) bir katmandır.

Örnek Olay: Bir aylık bütçeyi yiyen "arama sarmalı"
Bir ekip, müşteri sorularını yanıtlamak için bir ajan inşa etti. Ajan bir cevap bulamadığında aramaya devam etti: dokümanlarda ara, web'de ara, dokümanlarda tekrar ara, tekrar özetle... Döngü çökmedi. Sadece... fatura yazdı.
Çözüm: Soru başına maksimum 3 arama (kesin sınır), "bulunan yeni benzersiz kaynaklar" (ilerleme metriği) ve 2 aramadan sonra yeni kaynak yoksa kullanıcıdan açıklayıcı soru isteme (çekimser kalma).
Sonuç: Yanıtlar hızlandı, maliyetler düştü ve ajan öngörülebilir hale geldi.

Altın Kural: Onurunla Dur
Ajanınız durduğunda, faydalı bir şekilde durmalıdır. "İyi bir duruş" yanıtı şunları içerir:

Ne denediği (kısaca)

Ne bulduğu (eğer bir şey bulduysa)

Neden durduğu (bütçe/ilerleme/izinler)

En iyi bir sonraki eylem (sor/devret/manuel)

Bu güven inşa eder. Ve "pes etti" algısını engeller. Durmak başarısızlık değildir. Durmak kontroldür.

Sonuç: Otonomi pahalıdır — onu koruyun
Ajanlar inşa ediyorsanız soru "döngüye girebilir mi?" değildir. Girecektir. Soru şudur: Döngüleri ucuz, güvenli ve görünür kılan döngü korumalarınız var mı?

Kodun Açıklaması ve Yorum Satırları
Yazarın verdiği Python kodu, yukarıda anlatılan Koruma Kontrolcüsü (Loop Guard) katmanının çok temiz ve temel bir örneğidir.

Python
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
Kısaca Kodun Çalışma Mantığı:
Bu kod bloğu, bir yapay zeka ajanının içine yerleştirilen bir "bekçi" gibidir. Ajan her yeni adım atacağında check_budget() ile sınırları aşıp aşmadığı kontrol edilir. Ajan bir araç kullandığında record_tool_call() ile ne yaptığı kaydedilir ve hata alıp almadığına bakılır. İşlem bitiminde ise record_progress() ile gerçekten dişe dokunur bir ilerleme (state değişikliği) yapıp yapmadığı test edilir. Eğer ajan aynı şeyleri yapıp duruyorsa, bekçi ajanı durdurur ve faturanızın kabarmasını engeller.
"""

file_path = "/mnt/data/ajanlar_ne_zaman_durmali.md"
with open(file_path, "w", encoding="utf-8") as file:
file.write(markdown_content)

print(f"File saved to {file_path}")


```text?code_stdout&code_event_index=1
File saved to /mnt/data/ajanlar_ne_zaman_durmali.md

Your Markdown (.md) file is ready:

MD simgesi
ajanlar_ne_zaman_durmali
 MD 
Açık
Makalenin Türkçe çevirisini ve kodların detaylı açıklamalarını içeren Markdown dosyası oluşturuldu. İndirip inceleyebilirsiniz.

The Agent Loop Problem: When “Smart” Won’t Stop

How to diagnose runaway agent behavior — and design guardrails that turn infinite loops into predictable, shippable systems.



Modexa

Follow

6 min read

·

Feb 12, 2026

64













Press enter or click to view image in full size





The Agent Loop Problem explains why AI agents get stuck in infinite cycles — and how to stop them using budgets, state machines, evaluations, and safe fallbacks.

You ship an agent. It works in demos.

Then production happens.

A user asks a slightly messy question, the agent calls a tool, reads the result, calls another tool… and suddenly it’s been 47 steps and you’re staring at logs that feel like a haunted diary:

“I should verify again.”

“I should re-check the source.”

“I should confirm one more time.”

Let’s be real: this is the part nobody shows in the launch video.

This is the Agent Loop Problem — when systems keep thinking, calling, retrying, or re-planning long after they’ve stopped making progress.

And if you don’t handle it intentionally, your “helpful assistant” becomes a cost machine, a latency machine, or worse: a machine that never returns.

What Is the Agent Loop Problem?

An agent loop happens when an autonomous system repeatedly cycles through:

decide / plan

act (tool call)

observe

revise

repeat

…without converging.

It can look like:

infinite retries (“maybe the tool failed?”)

endless searching (“one more source”)

self-critique spirals (“my answer might be wrong”)

tool ping-pong (A → B → A → B)

“plan churn” (rewriting the plan every step)

In short: the agent keeps moving, but the system isn’t progressing.

That difference matters. Motion is not progress.

Why Systems Get Stuck (The Real Causes)

Most loops aren’t caused by “bad models.” They’re caused by bad incentives and missing constraints.

1) No explicit definition of “done”

Humans have an internal “good enough.” Agents don’t — unless you give them one.

If the prompt says:

“be thorough”

“verify”

“don’t miss anything”

…you’ve basically told the agent to keep searching forever.

The agent isn’t broken. It’s obeying.

2) Unreliable tools + naive retry logic

If a tool is flaky (timeouts, rate limits, partial responses), an agent will interpret that as:

“Try again, but slightly differently.”



Without a cap, it becomes a slot machine.

3) Ambiguous goals and shifting targets

When a user request is unclear, an agent can bounce between interpretations:



“Maybe they meant X.”

“Actually, maybe Y.”

“Let me check again.”

Goal uncertainty creates loop pressure.

4) Context gets worse every step

Each step adds tokens: logs, partial plans, tool outputs.



Eventually the agent is reasoning over a messy transcript of its own confusion.

And confusion feeds more confusion.

5) The agent is optimizing for “not being wrong”

This is subtle.



Many agent prompts push extreme caution. In production, the safest path becomes:

ask for more info

verify again

run one more check

You didn’t build a solver. You built a risk-avoidance machine.

A Simple Model: Loops Are Missing Constraints

If you want a clean mental model, think of agents like search algorithms.



A search algorithm without:

a stopping condition

a budget

a heuristic for progress

…will wander.

So the fix is not “tell the agent to stop.”

The fix is giving the system mechanical reasons to stop.

Architecture Flow: The “Loop-Safe Agent” Pattern

Here’s a practical architecture that prevents runaway behavior:



User Request

|

v

Policy Gate (budgets, permissions, risk rules)

|

v

Planner (produces a bounded plan with max steps)

|

v

Executor Loop

- tool call

- parse result

- update state

- compute progress signal

- check stop rules

|

+--> if stuck: Fallback / Ask user / Escalate

|

v

Final Response + Trace Summary

The key addition is progress detection plus stop rules that are enforced outside the model.

Because the model will always be tempted to “try one more thing.”

How to Detect a Loop Before It Burns Your Budget

Loop detection is a product feature disguised as an engineering detail.



Here are the highest-signal checks.

1) Step budget + time budget (always)

max tool calls (e.g., 6)

max reasoning steps (e.g., 10)

max wall-clock time (e.g., 20s)

Budgets aren’t just safety. They’re UX.

A user would rather get a partial answer quickly than wait forever.

2) Duplicate action detection

If the agent calls the same tool with nearly the same inputs repeatedly, it’s stuck.

Add fingerprints:

tool name

normalized parameters

hash of request payload

If repeated 2–3 times → stop or switch strategy.

3) “No new information” detection

Track whether each step adds meaningful new data.



If the last N steps produce:

the same facts

the same errors

the same uncertainty statements

…it’s a loop.

4) Plan churn detection

If the agent rewrites the plan every step, you’re not executing. You’re circling.



Count plan revisions. If > 2, force execution or fallback.

Practical Stop Mechanisms That Actually Work

1) Convert free-form loops into a state machine

Agents love vague freedom. Systems need crisp states.



Example states:

UNDERSTAND

GATHER

ACT

VERIFY

RESPOND

ESCALATE

And only allow specific transitions.

That alone eliminates a huge class of loops.

2) Add a progress score

You can compute a simple progress heuristic like:



% of required fields filled

number of sources gathered

distance to goal (tasks remaining)

If progress score doesn’t increase after 2 steps → stop.

3) Use “fallback ladders”

Instead of “try again,” define structured fallback paths:



Retry once with backoff

Switch tool/provider

Reduce scope

Ask the user a clarifying question

Return best-effort answer + next steps

The agent shouldn’t invent retries. It should follow a ladder.

4) Make “ask user” a first-class outcome, not a failure

Builders treat clarifying questions like defeat. Users often prefer them.



But do it well:

ask one question, not five

explain what it changes

offer defaults if the user doesn’t care

Code Sample: Loop Guard Wrapper (Python)

Here’s a minimal pattern that stops infinite tool ping-pong. Keep it boring. Boring is reliable.



import time

import hashlib

from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Optional, Tuple



@dataclass

class StepResult:

observation: str

tool_used: Optional[str] = None

tool_args: Optional[Dict[str, Any]] = None

made_progress: bool = True



def fingerprint(tool: str, args: Dict[str, Any]) -> str:

normalized = str(sorted(args.items())).encode("utf-8")

return hashlib.sha256(tool.encode("utf-8") + b"|" + normalized).hexdigest()[:12]



def run_agent_loop(

step_fn: Callable[[str], StepResult],

user_input: str,

max_steps: int = 10,

max_tool_calls: int = 6,

max_seconds: float = 20.0,

repeat_limit: int = 2,

no_progress_limit: int = 2,

) -> Tuple[str, List[str]]:

start = time.time()

tool_calls = 0

seen: Dict[str, int] = {}

no_progress = 0

trace: List[str] = []



state = user_input



for i in range(max_steps):

if time.time() - start > max_seconds:

trace.append("STOP: time budget exceeded")

break



result = step_fn(state)

trace.append(f"STEP {i+1}: {result.observation[:140]}")



if result.tool_used:

tool_calls += 1

if tool_calls > max_tool_calls:

trace.append("STOP: tool call budget exceeded")

break



fp = fingerprint(result.tool_used, result.tool_args or {})

seen[fp] = seen.get(fp, 0) + 1

if seen[fp] > repeat_limit:

trace.append(f"STOP: repeated tool call detected ({result.tool_used})")

break



if not result.made_progress:

no_progress += 1

if no_progress > no_progress_limit:

trace.append("STOP: no-progress streak exceeded")

break

else:

no_progress = 0



# Update "state"—in real systems this would be structured memory.

state = result.observation



return ("Best-effort response generated. See trace for stop reason.", trace)

What this gives you:

hard budgets (steps, time, tool calls)

duplicate tool call detection

no-progress stop condition

a trace you can store for debugging

The important part isn’t the Python.

It’s the idea that stopping is enforced outside the model.

Real-World Example: The “Verify Again” Trap

Agents in support workflows often loop like this:



user asks: “Why was I charged twice?”

agent checks billing system

sees partial data

checks again

pings refund tool

gets ambiguous response

checks billing system again

repeats

This is rarely an intelligence issue.

It’s a systems design issue:

unclear definition of resolution

tool outputs not structured

missing “escalate to human” policy

no maximum retries per tool

Fix it with:

structured billing tool responses (status codes + fields)

one retry max

explicit escalation after ambiguity

user-facing message: “I can’t confirm X automatically, but I can do Y.”

The Builder’s Rule: Every Loop Needs an Exit

If you only remember one thing, make it this:



Every agent loop must have a designed exit — success, fallback, ask, or escalate.



Not “try forever.”

Not “hope the model figures it out.”

Exit conditions are product decisions:

how much latency users tolerate

what uncertainty is acceptable

when human override kicks in

what partial answers are useful

That’s how you build agents that feel professional instead of possessed.

Conclusion: Stop Is a Feature

Agents get stuck because we gave them motion without brakes.



So build brakes:

budgets

progress signals

state machines

fallback ladders

explicit escalation paths

Your agent doesn’t need infinite autonomy.

It needs predictable behavior under messy reality.

CTA: Have you seen an agent loop in your own system — tool retries, plan churn, endless verifying? Comment what it looked like. I’ll suggest a concrete stop rule set for that pattern.



bunu aynı şekilde baştan sona çevir

İşte makalenin aslına sadık kalarak, aynı profesyonel ve pragmatik üslupla yapılmış tam çevirisi:

Ajan Döngüsü Problemi: "Akıllı" Sistemler Durmadığında
Kontrolden çıkmış ajan davranışları nasıl teşhis edilir ve sonsuz döngüleri öngörülebilir, piyasaya sürülebilir sistemlere dönüştüren bariyerler (guardrails) nasıl tasarlanır.

Modexa

Takip Et

6 dakikalık okuma · 12 Şub 2026 · 64

Ajan Döngüsü Problemi, yapay zeka ajanlarının neden sonsuz döngülere sıkıştığını ve bütçeler, durum makineleri (state machines), değerlendirmeler ve güvenli geri dönüş (fallback) senaryoları kullanarak bunları nasıl durduracağınızı açıklar.

Bir ajan (agent) yayınlıyorsunuz. Demolarda harika çalışıyor.
Sonra canlı ortama (production) geçiliyor.
Bir kullanıcı biraz karmaşık bir soru soruyor, ajan bir aracı (tool) çağırıyor, sonucu okuyor, başka bir aracı çağırıyor... ve aniden 47 adım geçiyor ve kendinizi lanetli bir günlük gibi hissettiren loglara (kayıtlara) bakarken buluyorsunuz:

"Tekrar doğrulamalıyım."
"Kaynağı yeniden kontrol etmeliyim."
"Bir kez daha teyit etmeliyim."

Gerçekçi olalım: Bu, kimsenin lansman videosunda göstermediği kısımdır.

Bu duruma Ajan Döngüsü Problemi (Agent Loop Problem) diyoruz; yani sistemlerin ilerleme kaydetmeyi bıraktıktan çok sonra bile düşünmeye, araç çağırmaya, tekrar denemeye veya yeniden planlamaya devam etmesi durumudur.

Ve eğer bu durumu bilinçli bir şekilde ele almazsanız, "yardımsever asistanınız" bir maliyet makinesine, bir gecikme (latency) makinesine veya daha kötüsü: asla yanıt dönmeyen bir makineye dönüşür.

Ajan Döngüsü Problemi Nedir?
Bir ajan döngüsü, otonom bir sistemin sürekli olarak şu adımları tekrarlamasıyla (fakat bir sonuca varmamasıyla) oluşur:

Karar ver / planla

Harekete geç (araç çağrısı)

Gözlemle

Gözden geçir

Tekrarla

Bu durum dışarıdan şu şekillerde görünebilir:

Sonsuz tekrarlar: ("Belki araç başarısız oldu?")

Bitmeyen aramalar: ("Bir kaynak daha bulayım")

Özeleştiri sarmalları: ("Cevabım yanlış olabilir")

Araç ping-pong'u: (A Aracı → B Aracı → A Aracı → B Aracı)

Plan çalkantısı (Plan churn): (Her adımda planı baştan aşağı yeniden yazmak)

Kısacası: Ajan sürekli hareket halindedir ancak sistem ilerlemiyordur.
Bu fark önemlidir. Hareket etmek (motion), ilerlemek (progress) demek değildir.

Sistemler Neden Sıkışır (Gerçek Nedenler)
Çoğu döngü "kötü modellerden" kaynaklanmaz. Kötü teşvikler (incentives) ve eksik kısıtlamalardan kaynaklanırlar.

1) Açık bir "Bitti" (Done) tanımının olmaması
İnsanların içsel bir "bu kadar yeterli" (good enough) algısı vardır. Siz onlara bir tane vermedikçe ajanların yoktur.
Eğer istem (prompt) şunları söylüyorsa:

"Kapsamlı ol"

"Doğrula"

"Hiçbir şeyi kaçırma"

...aslında ajana sonsuza kadar arama yapmasını söylemiş olursunuz.
Ajan bozuk değildir. Sadece verdiğiniz emirlere itaat ediyordur.

2) Güvenilmez araçlar + naif tekrar deneme mantığı
Eğer bir araç tutarsızsa (zaman aşımları, hız sınırları, kısmi yanıtlar), bir ajan bunu şu şekilde yorumlayacaktır:

"Tekrar dene, ama biraz farklı şekilde."

Bir üst sınır (cap) olmazsa, bu sistem bir slot makinesine dönüşür.

3) Belirsiz hedefler ve değişen beklentiler
Bir kullanıcı isteği belirsiz olduğunda, ajan farklı yorumlar arasında gidip gelebilir:

"Belki de X demek istediler."

"Aslında, belki de Y."

"En iyisi tekrar kontrol edeyim."

Hedef belirsizliği, sistem üzerinde döngü baskısı yaratır.

4) Bağlamın her adımda daha da kötüleşmesi
Her adım modele yeni token'lar ekler: loglar, kısmi planlar, araç çıktıları.
Sonunda ajan, kendi kafa karışıklığının dağınık bir transkripti üzerinden mantık yürütmeye başlar.
Ve kafa karışıklığı, daha fazla kafa karışıklığını besler.

5) Ajanın "yanlış yapmamak" üzerine optimize edilmesi
Bu oldukça incelikli bir durumdur.
Birçok ajan istemi aşırı tedbiri teşvik eder. Canlı (production) ortamında en güvenli yol şu hale gelir:

Daha fazla bilgi iste

Tekrar doğrula

Bir kontrol daha yap

Siz bir sorun çözücü (solver) inşa etmediniz. Bir riskten kaçınma makinesi inşa ettiniz.

Basit Bir Model: Döngüler Eksik Kısıtlamalardır
Net bir zihinsel model istiyorsanız, ajanları arama algoritmaları gibi düşünün.
Şunlardan yoksun bir arama algoritması başıboş dolaşacaktır:

Bir durma koşulu

Bir bütçe

İlerleme için bir sezgisel yöntem (heuristic)

Yani çözüm "ajana durmasını söylemek" değildir.
Çözüm, sisteme durması için mekanik nedenler (kurallar) vermektir.

Mimari Akış: "Döngü Güvenlikli Ajan" (Loop-Safe Agent) Modeli
İşte kontrolden çıkmış davranışları önleyen pratik bir mimari akış:

Plaintext
Kullanıcı İsteği
   |
   v
Politika Kapısı (bütçeler, izinler, risk kuralları)
   |
   v
Planlayıcı (maksimum adım sayısıyla sınırlanmış bir plan üretir)
   |
   v
Çalıştırma Döngüsü (Executor Loop)
  - araç çağrısı
  - sonucu ayrıştır
  - durumu güncelle
  - ilerleme sinyalini hesapla
  - durdurma kurallarını kontrol et
   |
   +--> eğer sıkıştıysa: Geri Dönüş (Fallback) / Kullanıcıya Sor / Devret
   |
   v
Nihai Yanıt + İzleme (Trace) Özeti
Buradaki en kilit ekleme, ilerleme (progress) tespiti artı modelin dışında uygulanan durdurma kurallarıdır.
Çünkü model her zaman "bir şey daha deneme" eğiliminde olacaktır.

Bir Döngüyü Bütçenizi Yakmadan Önce Nasıl Tespit Edersiniz?
Döngü tespiti, bir mühendislik detayı kılığına girmiş bir ürün özelliğidir.
İşte tespitte en etkili (yüksek sinyalli) kontroller:

1) Adım bütçesi + zaman bütçesi (Her zaman uygulamalısınız)

Maksimum araç çağrısı (ör. 6)

Maksimum mantık yürütme adımı (ör. 10)

Maksimum geçen süre (ör. 20s)

Bütçeler sadece güvenlik değildir. Onlar Kullanıcı Deneyimidir (UX). Bir kullanıcı sonsuza kadar beklemektense kısmi bir cevabı hızlıca almayı tercih eder.

2) Yinelenen eylem (Duplicate action) tespiti
Eğer ajan aynı aracı neredeyse aynı girdilerle tekrar tekrar çağırıyorsa, sıkışmış demektir.
Parmak izleri (fingerprints) ekleyin: araç adı, normalize edilmiş parametreler, istek yükünün (payload) hash değeri.
Eğer aynı eylem 2-3 kez tekrarlanırsa → durun veya strateji değiştirin.

3) "Yeni bilgi yok" tespiti
Her adımın anlamlı yeni bir veri ekleyip eklemediğini takip edin. Eğer son N adım aynı gerçekleri, aynı hataları veya aynı belirsizlik ifadelerini üretiyorsa... bu bir döngüdür.

4) Plan çalkantısı (churn) tespiti
Eğer ajan her adımda planı yeniden yazıyorsa, işlem yapmıyorsunuzdur. Kendi etrafınızda dönüyorsunuzdur. Plan revizyonlarını sayın. Eğer > 2 ise, çalıştırmayı veya geri dönüş senaryosunu (fallback) zorlayın.

Gerçekten İşe Yarayan Pratik Durdurma Mekanizmaları
1) Serbest formlu döngüleri bir durum makinesine (state machine) dönüştürün
Ajanlar belirsiz özgürlükleri severler. Sistemlerin ise net durumlara ihtiyacı vardır.
Örnek durumlar: ANLA → TOPLA → HAREKETE GEÇ → DOĞRULA → YANITLA → DEVRET
Ve yalnızca belirli geçişlere izin verin. Sadece bu bile çok büyük bir döngü sınıfını ortadan kaldırır.

2) Bir ilerleme puanı (progress score) ekleyin
Aşağıdaki gibi basit bir ilerleme metriği hesaplayabilirsiniz:

Doldurulan zorunlu alanların %'si

Toplanan kaynak sayısı

Hedefe olan uzaklık (kalan görevler)

Eğer ilerleme puanı 2 adım sonra artmıyorsa → durdurun.

3) "Geri dönüş merdivenleri" (fallback ladders) kullanın
Ajanın rastgele "tekrar denemesi" yerine, yapılandırılmış yollar tanımlayın:

Bekleme süresiyle (backoff) bir kez tekrar dene

Aracı/sağlayıcıyı değiştir

Kapsamı (scope) daralt

Kullanıcıya açıklayıcı bir soru sor

Elinizdeki en iyi cevabı (best-effort) + sonraki adımları döndür

Ajan kendi kendine tekrarlar icat etmemelidir. Bir merdiveni takip etmelidir.

4) "Kullanıcıya sor" seçeneğini bir başarısızlık değil, birinci sınıf bir sonuç yapın
Geliştiriciler açıklayıcı sorulara bir yenilgi gibi yaklaşır. Kullanıcılar ise genelde bunları tercih eder. Ancak bunu iyi yapın:

Beş soru değil, tek bir soru sorun.

Bunun neyi değiştireceğini (neden sorduğunuzu) açıklayın.

Kullanıcı umursamazsa varsayılan (default) seçenekler sunun.

Kod Örneği: Döngü Koruması Sargısı (Loop Guard Wrapper)
İşte sonsuz araç ping-pong'unu durduran minimal bir şablon. Basit (boring) tutun. Basit olan güvenilirdir.

Python
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

@dataclass
class StepResult:
    observation: str
    tool_used: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    made_progress: bool = True

# Araç adını ve argümanları alıp tekrarları tespit etmek için hash (parmak izi) üreten fonksiyon
def fingerprint(tool: str, args: Dict[str, Any]) -> str:
    normalized = str(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(tool.encode("utf-8") + b"|" + normalized).hexdigest()[:12]

# Ana döngü koruma kontrolcüsü
def run_agent_loop(
    step_fn: Callable[[str], StepResult],
    user_input: str,
    max_steps: int = 10,
    max_tool_calls: int = 6,
    max_seconds: float = 20.0,
    repeat_limit: int = 2,
    no_progress_limit: int = 2,
) -> Tuple[str, List[str]]:
    start = time.time()
    tool_calls = 0
    seen: Dict[str, int] = {}
    no_progress = 0
    trace: List[str] = []

    state = user_input

    for i in range(max_steps):
        # KORUMA 1: Zaman Sınırı Kontrolü
        if time.time() - start > max_seconds:
            trace.append("STOP: Zaman bütçesi aşıldı")
            break

        result = step_fn(state)
        trace.append(f"ADIM {i+1}: {result.observation[:140]}")

        if result.tool_used:
            tool_calls += 1
            # KORUMA 2: Araç Çağrısı Bütçesi
            if tool_calls > max_tool_calls:
                trace.append("STOP: Araç çağrısı bütçesi aşıldı")
                break

            # KORUMA 3: Tekrarlayan İşlem Parmak İzi
            fp = fingerprint(result.tool_used, result.tool_args or {})
            seen[fp] = seen.get(fp, 0) + 1
            if seen[fp] > repeat_limit:
                trace.append(f"STOP: Yinelenen araç çağrısı tespit edildi ({result.tool_used})")
                break

        # KORUMA 4: İlerleme Olmaması (No-Progress) Limiti
        if not result.made_progress:
            no_progress += 1
            if no_progress > no_progress_limit:
                trace.append("STOP: İlerleme kaydedilemeyen (no-progress) adım serisi sınırı aşıldı")
                break
        else:
            no_progress = 0

        # "Durumu" güncelle — gerçek sistemlerde bu yapılandırılmış bir bellek (memory) olacaktır.
        state = result.observation

    return ("En iyi çabayla (best-effort) oluşturulan yanıt. Durma nedeni için logları inceleyin.", trace)
Bu kodun size sağladıkları:

Kesin bütçeler (adımlar, zaman, araç çağrıları)

Yinelenen araç çağrısı tespiti

İlerleme yoksa durma koşulu

Hata ayıklama (debugging) için saklayabileceğiniz bir izleme günlüğü (trace)

Buradaki önemli kısım Python kodu değildir.
Önemli olan düşünce, durdurmanın modelin dışında zorunlu kılınmasıdır (enforced).

Gerçek Dünyadan Bir Örnek: "Tekrar Doğrula" Tuzağı
Destek (support) iş akışlarındaki ajanlar genellikle şöyle döngülere girer:

Kullanıcı sorar: "Neden benden iki kez ücret alındı?"

Ajan faturalandırma sistemini kontrol eder

Kısmi veri görür

Tekrar kontrol eder

İade aracını yoklar (ping)

Belirsiz bir yanıt alır

Faturalandırma sistemini tekrar kontrol eder

Tekrarlar

Bu nadiren bir zeka (intelligence) sorunudur. Bu bir sistem tasarımı sorunudur:

Çözümlemenin (resolution) ne olduğunun net bir tanımı yoktur.

Araç çıktıları yapılandırılmış (structured) değildir.

"İnsana devret" (escalate) politikası eksiktir.

Araç başına maksimum tekrar deneme sınırı yoktur.

Nasıl düzeltilir:

Yapılandırılmış faturalandırma aracı yanıtları (durum kodları + alanlar) ekleyin.

Maksimum tek bir tekrar deneme (retry) limiti koyun.

Belirsizlik durumunda doğrudan (explicit) insana devretme senaryosunu devreye alın.

Kullanıcıya dönük şu mesajı ayarlayın: "X'i otomatik olarak doğrulayamıyorum, ancak Y yapabilirim."

Geliştirici Kuralı: Her Döngünün Bir Çıkışı Olmalıdır
Eğer tek bir şey hatırlayacaksanız, o da şu olsun:

Her ajan döngüsünün tasarlanmış bir çıkışı (exit) olmalıdır — başarı, geri dönüş (fallback), sorma (ask) veya devretme (escalate).

"Sonsuza kadar dene" değil.
"Umarım model çözer" değil.

Çıkış koşulları ürünle ilgili alınan kararlardır:

Kullanıcılar ne kadar gecikmeye (latency) tahammül edebilir?

Ne kadar belirsizlik kabul edilebilirdir?

İnsan müdahalesi (override) ne zaman devreye girer?

Hangi kısmi cevaplar faydalıdır?

İşte böyle cin çarpmış (possessed) gibi değil, profesyonel hissettiren ajanlar inşa edersiniz.

Sonuç: Durmak Bir Özelliktir
Ajanlar sıkışıp kalır, çünkü onlara frensiz bir hareket kabiliyeti verdik.

Öyleyse frenler inşa edin:

Bütçeler

İlerleme sinyalleri

Durum makineleri (state machines)

Geri dönüş merdivenleri (fallback ladders)

Doğrudan (explicit) devretme (escalation) yolları

Ajanınızın sonsuz otonomiye ihtiyacı yoktur.
Karmaşık (messy) gerçeklik altında öngörülebilir davranışlara ihtiyacı vardır.

CTA: Kendi sisteminizde bir ajan döngüsü — araç tekrarları, plan çalkantısı, bitmek bilmeyen doğrulamalar — gördünüz mü? Neye benzediğini yorumlara yazın. O kalıp için somut bir durdurma kuralı seti önereyim.

son turnimizi bir md yap

Buradaki en kilit ekleme, ilerleme (progress) tespiti artı modelin dışında uygulanan durdurma kurallarıdır.
Çünkü model her zaman "bir şey daha deneme" eğiliminde olacaktır.

Bir Döngüyü Bütçenizi Yakmadan Önce Nasıl Tespit Edersiniz?
Döngü tespiti, bir mühendislik detayı kılığına girmiş bir ürün özelliğidir.
İşte tespitte en etkili (yüksek sinyalli) kontroller:

1) Adım bütçesi + zaman bütçesi (Her zaman uygulamalısınız)

Maksimum araç çağrısı (ör. 6)

Maksimum mantık yürütme adımı (ör. 10)

Maksimum geçen süre (ör. 20s)

Bütçeler sadece güvenlik değildir. Onlar Kullanıcı Deneyimidir (UX). Bir kullanıcı sonsuza kadar beklemektense kısmi bir cevabı hızlıca almayı tercih eder.

2) Yinelenen eylem (Duplicate action) tespiti
Eğer ajan aynı aracı neredeyse aynı girdilerle tekrar tekrar çağırıyorsa, sıkışmış demektir.
Parmak izleri (fingerprints) ekleyin: araç adı, normalize edilmiş parametreler, istek yükünün (payload) hash değeri.
Eğer aynı eylem 2-3 kez tekrarlanırsa → durun veya strateji değiştirin.

3) "Yeni bilgi yok" tespiti
Her adımın anlamlı yeni bir veri ekleyip eklemediğini takip edin. Eğer son N adım aynı gerçekleri, aynı hataları veya aynı belirsizlik ifadelerini üretiyorsa... bu bir döngüdür.

4) Plan çalkantısı (churn) tespiti
Eğer ajan her adımda planı yeniden yazıyorsa, işlem yapmıyorsunuzdur. Kendi etrafınızda dönüyorsunuzdur. Plan revizyonlarını sayın. Eğer > 2 ise, çalıştırmayı veya geri dönüş senaryosunu (fallback) zorlayın.

Gerçekten İşe Yarayan Pratik Durdurma Mekanizmaları
1) Serbest formlu döngüleri bir durum makinesine (state machine) dönüştürün
Ajanlar belirsiz özgürlükleri severler. Sistemlerin ise net durumlara ihtiyacı vardır.
Örnek durumlar: ANLA → TOPLA → HAREKETE GEÇ → DOĞRULA → YANITLA → DEVRET
Ve yalnızca belirli geçişlere izin verin. Sadece bu bile çok büyük bir döngü sınıfını ortadan kaldırır.

2) Bir ilerleme puanı (progress score) ekleyin
Aşağıdaki gibi basit bir ilerleme metriği hesaplayabilirsiniz:

Doldurulan zorunlu alanların %'si

Toplanan kaynak sayısı

Hedefe olan uzaklık (kalan görevler)

Eğer ilerleme puanı 2 adım sonra artmıyorsa → durdurun.

3) "Geri dönüş merdivenleri" (fallback ladders) kullanın
Ajanın rastgele "tekrar denemesi" yerine, yapılandırılmış yollar tanımlayın:

Bekleme süresiyle (backoff) bir kez tekrar dene

Aracı/sağlayıcıyı değiştir

Kapsamı (scope) daralt

Kullanıcıya açıklayıcı bir soru sor

Elinizdeki en iyi cevabı (best-effort) + sonraki adımları döndür

Ajan kendi kendine tekrarlar icat etmemelidir. Bir merdiveni takip etmelidir.

4) "Kullanıcıya sor" seçeneğini bir başarısızlık değil, birinci sınıf bir sonuç yapın
Geliştiriciler açıklayıcı sorulara bir yenilgi gibi yaklaşır. Kullanıcılar ise genelde bunları tercih eder. Ancak bunu iyi yapın:

Beş soru değil, tek bir soru sorun.

Bunun neyi değiştireceğini (neden sorduğunuzu) açıklayın.

Kullanıcı umursamazsa varsayılan (default) seçenekler sunun.

Kod Örneği: Döngü Koruması Sargısı (Loop Guard Wrapper)
İşte sonsuz araç ping-pong'unu durduran minimal bir şablon. Basit (boring) tutun. Basit olan güvenilirdir.

Python
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

@dataclass
class StepResult:
    observation: str
    tool_used: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    made_progress: bool = True

# Araç adını ve argümanları alıp tekrarları tespit etmek için hash (parmak izi) üreten fonksiyon
def fingerprint(tool: str, args: Dict[str, Any]) -> str:
    normalized = str(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(tool.encode("utf-8") + b"|" + normalized).hexdigest()[:12]

# Ana döngü koruma kontrolcüsü
def run_agent_loop(
    step_fn: Callable[[str], StepResult],
    user_input: str,
    max_steps: int = 10,
    max_tool_calls: int = 6,
    max_seconds: float = 20.0,
    repeat_limit: int = 2,
    no_progress_limit: int = 2,
) -> Tuple[str, List[str]]:
    start = time.time()
    tool_calls = 0
    seen: Dict[str, int] = {}
    no_progress = 0
    trace: List[str] = []

    state = user_input

    for i in range(max_steps):
        # KORUMA 1: Zaman Sınırı Kontrolü
        if time.time() - start > max_seconds:
            trace.append("STOP: Zaman bütçesi aşıldı")
            break

        result = step_fn(state)
        trace.append(f"ADIM {i+1}: {result.observation[:140]}")

        if result.tool_used:
            tool_calls += 1
            # KORUMA 2: Araç Çağrısı Bütçesi
            if tool_calls > max_tool_calls:
                trace.append("STOP: Araç çağrısı bütçesi aşıldı")
                break

            # KORUMA 3: Tekrarlayan İşlem Parmak İzi
            fp = fingerprint(result.tool_used, result.tool_args or {})
            seen[fp] = seen.get(fp, 0) + 1
            if seen[fp] > repeat_limit:
                trace.append(f"STOP: Yinelenen araç çağrısı tespit edildi ({result.tool_used})")
                break

        # KORUMA 4: İlerleme Olmaması (No-Progress) Limiti
        if not result.made_progress:
            no_progress += 1
            if no_progress > no_progress_limit:
                trace.append("STOP: İlerleme kaydedilemeyen (no-progress) adım serisi sınırı aşıldı")
                break
        else:
            no_progress = 0

        # "Durumu" güncelle — gerçek sistemlerde bu yapılandırılmış bir bellek (memory) olacaktır.
        state = result.observation

    return ("En iyi çabayla (best-effort) oluşturulan yanıt. Durma nedeni için logları inceleyin.", trace)
Bu kodun size sağladıkları:

Kesin bütçeler (adımlar, zaman, araç çağrıları)

Yinelenen araç çağrısı tespiti

İlerleme yoksa durma koşulu

Hata ayıklama (debugging) için saklayabileceğiniz bir izleme günlüğü (trace)

Buradaki önemli kısım Python kodu değildir.
Önemli olan düşünce, durdurmanın modelin dışında zorunlu kılınmasıdır (enforced).

Gerçek Dünyadan Bir Örnek: "Tekrar Doğrula" Tuzağı
Destek (support) iş akışlarındaki ajanlar genellikle şöyle döngülere girer:

Kullanıcı sorar: "Neden benden iki kez ücret alındı?"

Ajan faturalandırma sistemini kontrol eder

Kısmi veri görür

Tekrar kontrol eder

İade aracını yoklar (ping)

Belirsiz bir yanıt alır

Faturalandırma sistemini tekrar kontrol eder

Tekrarlar

Bu nadiren bir zeka (intelligence) sorunudur. Bu bir sistem tasarımı sorunudur:

Çözümlemenin (resolution) ne olduğunun net bir tanımı yoktur.

Araç çıktıları yapılandırılmış (structured) değildir.

"İnsana devret" (escalate) politikası eksiktir.

Araç başına maksimum tekrar deneme sınırı yoktur.

Nasıl düzeltilir:

Yapılandırılmış faturalandırma aracı yanıtları (durum kodları + alanlar) ekleyin.

Maksimum tek bir tekrar deneme (retry) limiti koyun.

Belirsizlik durumunda doğrudan (explicit) insana devretme senaryosunu devreye alın.

Kullanıcıya dönük şu mesajı ayarlayın: "X'i otomatik olarak doğrulayamıyorum, ancak Y yapabilirim."

Geliştirici Kuralı: Her Döngünün Bir Çıkışı Olmalıdır
Eğer tek bir şey hatırlayacaksanız, o da şu olsun:

Her ajan döngüsünün tasarlanmış bir çıkışı (exit) olmalıdır — başarı, geri dönüş (fallback), sorma (ask) veya devretme (escalate).

"Sonsuza kadar dene" değil.
"Umarım model çözer" değil.

Çıkış koşulları ürünle ilgili alınan kararlardır:

Kullanıcılar ne kadar gecikmeye (latency) tahammül edebilir?

Ne kadar belirsizlik kabul edilebilirdir?

İnsan müdahalesi (override) ne zaman devreye girer?

Hangi kısmi cevaplar faydalıdır?

İşte böyle cin çarpmış (possessed) gibi değil, profesyonel hissettiren ajanlar inşa edersiniz.

Sonuç: Durmak Bir Özelliktir
Ajanlar sıkışıp kalır, çünkü onlara frensiz bir hareket kabiliyeti verdik.

Öyleyse frenler inşa edin:

Bütçeler

İlerleme sinyalleri

Durum makineleri (state machines)

Geri dönüş merdivenleri (fallback ladders)

Doğrudan (explicit) devretme (escalation) yolları

Ajanınızın sonsuz otonomiye ihtiyacı yoktur.
Karmaşık (messy) gerçeklik altında öngörülebilir davranışlara ihtiyacı vardır.

CTA: Kendi sisteminizde bir ajan döngüsü — araç tekrarları, plan çalkantısı, bitmek bilmeyen doğrulamalar — gördünüz mü? Neye benzediğini yorumlara yazın. O kalıp için somut bir durdurma kuralı seti önereyim.
"""

file_path = "/mnt/data/ajan_dongusu_problemi.md"
with open(file_path, "w", encoding="utf-8") as file:
file.write(markdown_content)

print(f"File saved to {file_path}")


```text?code_stdout&code_event_index=1
File saved to /mnt/data/ajan_dongusu_problemi.md

Your second Markdown (.md) file is ready:

MD simgesi
ajan_dongusu_problemi
 MD 
Açık
"Ajan Döngüsü Problemi" adlı ikinci makalenin çevirisini ve kodunu içeren dosyayı yukarıdaki bağlantıdan indirip inceleyebilirsin. Başka çevirmemi veya formatlamamı istediğin bir şeyler var mı?




Gemini bir yapay zeka modeli olduğu için hata yapabilir.

Analiz ediliyor
# Ajan Döngüsü Problemi: "Akıllı" Sistemler Durmadığında
Kontrolden çıkmış ajan davranışları nasıl teşhis edilir ve sonsuz döngüleri öngörülebilir, piyasaya sürülebilir sistemlere dönüştüren bariyerler (guardrails) nasıl tasarlanır.

**Modexa**  
Takip Et  
6 dakikalık okuma · 12 Şub 2026 · 64

*Ajan Döngüsü Problemi, yapay zeka ajanlarının neden sonsuz döngülere sıkıştığını ve bütçeler, durum makineleri (state machines), değerlendirmeler ve güvenli geri dönüş (fallback) senaryoları kullanarak bunları nasıl durduracağınızı açıklar.*

Bir ajan (agent) yayınlıyorsunuz. Demolarda harika çalışıyor.
Sonra canlı ortama (production) geçiliyor.
Bir kullanıcı biraz karmaşık bir soru soruyor, ajan bir aracı (tool) çağırıyor, sonucu okuyor, başka bir aracı çağırıyor... ve aniden 47 adım geçiyor ve kendinizi lanetli bir günlük gibi hissettiren loglara (kayıtlara) bakarken buluyorsunuz:
> "Tekrar doğrulamalıyım."
> "Kaynağı yeniden kontrol etmeliyim."
> "Bir kez daha teyit etmeliyim."

Gerçekçi olalım: Bu, kimsenin lansman videosunda göstermediği kısımdır.

Bu duruma **Ajan Döngüsü Problemi (Agent Loop Problem)** diyoruz; yani sistemlerin ilerleme kaydetmeyi bıraktıktan çok sonra bile düşünmeye, araç çağırmaya, tekrar denemeye veya yeniden planlamaya devam etmesi durumudur.

Ve eğer bu durumu bilinçli bir şekilde ele almazsanız, "yardımsever asistanınız" bir maliyet makinesine, bir gecikme (latency) makinesine veya daha kötüsü: *asla yanıt dönmeyen* bir makineye dönüşür.

---

## Ajan Döngüsü Problemi Nedir?

Bir ajan döngüsü, otonom bir sistemin sürekli olarak şu adımları tekrarlamasıyla (fakat bir sonuca varmamasıyla) oluşur:
1. Karar ver / planla
2. Harekete geç (araç çağrısı)
3. Gözlemle
4. Gözden geçir
5. Tekrarla

Bu durum dışarıdan şu şekillerde görünebilir:
* **Sonsuz tekrarlar:** ("Belki araç başarısız oldu?")
* **Bitmeyen aramalar:** ("Bir kaynak daha bulayım")
* **Özeleştiri sarmalları:** ("Cevabım yanlış olabilir")
* **Araç ping-pong'u:** (A Aracı → B Aracı → A Aracı → B Aracı)
* **Plan çalkantısı (Plan churn):** (Her adımda planı baştan aşağı yeniden yazmak)

Kısacası: Ajan sürekli hareket halindedir ancak sistem ilerlemiyordur.
Bu fark önemlidir. **Hareket etmek (motion), ilerlemek (progress) demek değildir.**

---

## Sistemler Neden Sıkışır (Gerçek Nedenler)

Çoğu döngü "kötü modellerden" kaynaklanmaz. Kötü teşvikler (incentives) ve eksik kısıtlamalardan kaynaklanırlar.

### 1) Açık bir "Bitti" (Done) tanımının olmaması
İnsanların içsel bir "bu kadar yeterli" (good enough) algısı vardır. Siz onlara bir tane vermedikçe ajanların yoktur.
Eğer istem (prompt) şunları söylüyorsa:
* "Kapsamlı ol"
* "Doğrula"
* "Hiçbir şeyi kaçırma"

...aslında ajana sonsuza kadar arama yapmasını söylemiş olursunuz.
Ajan bozuk değildir. Sadece verdiğiniz emirlere itaat ediyordur.

### 2) Güvenilmez araçlar + naif tekrar deneme mantığı
Eğer bir araç tutarsızsa (zaman aşımları, hız sınırları, kısmi yanıtlar), bir ajan bunu şu şekilde yorumlayacaktır:
> "Tekrar dene, ama biraz farklı şekilde."

Bir üst sınır (cap) olmazsa, bu sistem bir slot makinesine dönüşür.

### 3) Belirsiz hedefler ve değişen beklentiler
Bir kullanıcı isteği belirsiz olduğunda, ajan farklı yorumlar arasında gidip gelebilir:
* "Belki de X demek istediler."
* "Aslında, belki de Y."
* "En iyisi tekrar kontrol edeyim."

Hedef belirsizliği, sistem üzerinde döngü baskısı yaratır.

### 4) Bağlamın her adımda daha da kötüleşmesi
Her adım modele yeni token'lar ekler: loglar, kısmi planlar, araç çıktıları.
Sonunda ajan, kendi kafa karışıklığının dağınık bir transkripti üzerinden mantık yürütmeye başlar.
Ve kafa karışıklığı, daha fazla kafa karışıklığını besler.

### 5) Ajanın "yanlış yapmamak" üzerine optimize edilmesi
Bu oldukça incelikli bir durumdur.
Birçok ajan istemi aşırı tedbiri teşvik eder. Canlı (production) ortamında en güvenli yol şu hale gelir:
* Daha fazla bilgi iste
* Tekrar doğrula
* Bir kontrol daha yap

Siz bir sorun çözücü (solver) inşa etmediniz. Bir *riskten kaçınma makinesi* inşa ettiniz.

---

## Basit Bir Model: Döngüler Eksik Kısıtlamalardır

Net bir zihinsel model istiyorsanız, ajanları arama algoritmaları gibi düşünün.
Şunlardan yoksun bir arama algoritması başıboş dolaşacaktır:
* Bir durma koşulu
* Bir bütçe
* İlerleme için bir sezgisel yöntem (heuristic)

Yani çözüm "ajana durmasını söylemek" değildir.
Çözüm, **sisteme durması için mekanik nedenler (kurallar) vermektir.**

---

## Mimari Akış: "Döngü Güvenlikli Ajan" (Loop-Safe Agent) Modeli

İşte kontrolden çıkmış davranışları önleyen pratik bir mimari akış:

```text
Kullanıcı İsteği
   |
   v
Politika Kapısı (bütçeler, izinler, risk kuralları)
   |
   v
Planlayıcı (maksimum adım sayısıyla sınırlanmış bir plan üretir)
   |
   v
Çalıştırma Döngüsü (Executor Loop)
  - araç çağrısı
  - sonucu ayrıştır
  - durumu güncelle
  - ilerleme sinyalini hesapla
  - durdurma kurallarını kontrol et
   |
   +--> eğer sıkıştıysa: Geri Dönüş (Fallback) / Kullanıcıya Sor / Devret
   |
   v
Nihai Yanıt + İzleme (Trace) Özeti
```

Buradaki en kilit ekleme, **ilerleme (progress) tespiti artı modelin dışında uygulanan durdurma kurallarıdır.**
Çünkü model her zaman "bir şey daha deneme" eğiliminde olacaktır.

---

## Bir Döngüyü Bütçenizi Yakmadan Önce Nasıl Tespit Edersiniz?

Döngü tespiti, bir mühendislik detayı kılığına girmiş bir *ürün özelliğidir.*
İşte tespitte en etkili (yüksek sinyalli) kontroller:

**1) Adım bütçesi + zaman bütçesi (Her zaman uygulamalısınız)**
* Maksimum araç çağrısı (ör. 6)
* Maksimum mantık yürütme adımı (ör. 10)
* Maksimum geçen süre (ör. 20s)

Bütçeler sadece güvenlik değildir. Onlar Kullanıcı Deneyimidir (UX). Bir kullanıcı sonsuza kadar beklemektense kısmi bir cevabı hızlıca almayı tercih eder.

**2) Yinelenen eylem (Duplicate action) tespiti**
Eğer ajan aynı aracı neredeyse aynı girdilerle tekrar tekrar çağırıyorsa, sıkışmış demektir.
Parmak izleri (fingerprints) ekleyin: *araç adı, normalize edilmiş parametreler, istek yükünün (payload) hash değeri.*
Eğer aynı eylem 2-3 kez tekrarlanırsa → **durun veya strateji değiştirin.**

**3) "Yeni bilgi yok" tespiti**
Her adımın anlamlı yeni bir veri ekleyip eklemediğini takip edin. Eğer son N adım *aynı gerçekleri, aynı hataları veya aynı belirsizlik ifadelerini* üretiyorsa... bu bir döngüdür.

**4) Plan çalkantısı (churn) tespiti**
Eğer ajan her adımda planı yeniden yazıyorsa, işlem yapmıyorsunuzdur. Kendi etrafınızda dönüyorsunuzdur. Plan revizyonlarını sayın. Eğer > 2 ise, çalıştırmayı veya geri dönüş senaryosunu (fallback) zorlayın.

---

## Gerçekten İşe Yarayan Pratik Durdurma Mekanizmaları

### 1) Serbest formlu döngüleri bir durum makinesine (state machine) dönüştürün
Ajanlar belirsiz özgürlükleri severler. Sistemlerin ise net durumlara ihtiyacı vardır.
Örnek durumlar: `ANLA` → `TOPLA` → `HAREKETE GEÇ` → `DOĞRULA` → `YANITLA` → `DEVRET`
Ve yalnızca belirli geçişlere izin verin. Sadece bu bile çok büyük bir döngü sınıfını ortadan kaldırır.

### 2) Bir ilerleme puanı (progress score) ekleyin
Aşağıdaki gibi basit bir ilerleme metriği hesaplayabilirsiniz:
* Doldurulan zorunlu alanların %'si
* Toplanan kaynak sayısı
* Hedefe olan uzaklık (kalan görevler)

Eğer ilerleme puanı 2 adım sonra artmıyorsa → durdurun.

### 3) "Geri dönüş merdivenleri" (fallback ladders) kullanın
Ajanın rastgele "tekrar denemesi" yerine, yapılandırılmış yollar tanımlayın:
1. Bekleme süresiyle (backoff) bir kez tekrar dene
2. Aracı/sağlayıcıyı değiştir
3. Kapsamı (scope) daralt
4. Kullanıcıya açıklayıcı bir soru sor
5. Elinizdeki en iyi cevabı (best-effort) + sonraki adımları döndür

Ajan kendi kendine tekrarlar icat etmemelidir. Bir merdiveni takip etmelidir.

### 4) "Kullanıcıya sor" seçeneğini bir başarısızlık değil, birinci sınıf bir sonuç yapın
Geliştiriciler açıklayıcı sorulara bir yenilgi gibi yaklaşır. Kullanıcılar ise genelde bunları tercih eder. Ancak bunu iyi yapın:
* Beş soru değil, tek bir soru sorun.
* Bunun neyi değiştireceğini (neden sorduğunuzu) açıklayın.
* Kullanıcı umursamazsa varsayılan (default) seçenekler sunun.

---

## Kod Örneği: Döngü Koruması Sargısı (Loop Guard Wrapper)

İşte sonsuz araç ping-pong'unu durduran minimal bir şablon. Basit (boring) tutun. Basit olan güvenilirdir.

```python
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

@dataclass
class StepResult:
    observation: str
    tool_used: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    made_progress: bool = True

# Araç adını ve argümanları alıp tekrarları tespit etmek için hash (parmak izi) üreten fonksiyon
def fingerprint(tool: str, args: Dict[str, Any]) -> str:
    normalized = str(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(tool.encode("utf-8") + b"|" + normalized).hexdigest()[:12]

# Ana döngü koruma kontrolcüsü
def run_agent_loop(
    step_fn: Callable[[str], StepResult],
    user_input: str,
    max_steps: int = 10,
    max_tool_calls: int = 6,
    max_seconds: float = 20.0,
    repeat_limit: int = 2,
    no_progress_limit: int = 2,
) -> Tuple[str, List[str]]:
    start = time.time()
    tool_calls = 0
    seen: Dict[str, int] = {}
    no_progress = 0
    trace: List[str] = []

    state = user_input

    for i in range(max_steps):
        # KORUMA 1: Zaman Sınırı Kontrolü
        if time.time() - start > max_seconds:
            trace.append("STOP: Zaman bütçesi aşıldı")
            break

        result = step_fn(state)
        trace.append(f"ADIM {i+1}: {result.observation[:140]}")

        if result.tool_used:
            tool_calls += 1
            # KORUMA 2: Araç Çağrısı Bütçesi
            if tool_calls > max_tool_calls:
                trace.append("STOP: Araç çağrısı bütçesi aşıldı")
                break

            # KORUMA 3: Tekrarlayan İşlem Parmak İzi
            fp = fingerprint(result.tool_used, result.tool_args or {})
            seen[fp] = seen.get(fp, 0) + 1
            if seen[fp] > repeat_limit:
                trace.append(f"STOP: Yinelenen araç çağrısı tespit edildi ({result.tool_used})")
                break

        # KORUMA 4: İlerleme Olmaması (No-Progress) Limiti
        if not result.made_progress:
            no_progress += 1
            if no_progress > no_progress_limit:
                trace.append("STOP: İlerleme kaydedilemeyen (no-progress) adım serisi sınırı aşıldı")
                break
        else:
            no_progress = 0

        # "Durumu" güncelle — gerçek sistemlerde bu yapılandırılmış bir bellek (memory) olacaktır.
        state = result.observation

    return ("En iyi çabayla (best-effort) oluşturulan yanıt. Durma nedeni için logları inceleyin.", trace)
```

Bu kodun size sağladıkları:
* Kesin bütçeler (adımlar, zaman, araç çağrıları)
* Yinelenen araç çağrısı tespiti
* İlerleme yoksa durma koşulu
* Hata ayıklama (debugging) için saklayabileceğiniz bir izleme günlüğü (trace)

Buradaki önemli kısım Python kodu değildir.
**Önemli olan düşünce, durdurmanın modelin dışında zorunlu kılınmasıdır (enforced).**

---

## Gerçek Dünyadan Bir Örnek: "Tekrar Doğrula" Tuzağı

Destek (support) iş akışlarındaki ajanlar genellikle şöyle döngülere girer:
1. Kullanıcı sorar: *"Neden benden iki kez ücret alındı?"*
2. Ajan faturalandırma sistemini kontrol eder
3. Kısmi veri görür
4. Tekrar kontrol eder
5. İade aracını yoklar (ping)
6. Belirsiz bir yanıt alır
7. Faturalandırma sistemini tekrar kontrol eder
8. *Tekrarlar*

Bu nadiren bir zeka (intelligence) sorunudur. **Bu bir sistem tasarımı sorunudur:**
* Çözümlemenin (resolution) ne olduğunun net bir tanımı yoktur.
* Araç çıktıları yapılandırılmış (structured) değildir.
* "İnsana devret" (escalate) politikası eksiktir.
* Araç başına maksimum tekrar deneme sınırı yoktur.

**Nasıl düzeltilir:**
* Yapılandırılmış faturalandırma aracı yanıtları (durum kodları + alanlar) ekleyin.
* Maksimum tek bir tekrar deneme (retry) limiti koyun.
* Belirsizlik durumunda doğrudan (explicit) insana devretme senaryosunu devreye alın.
* Kullanıcıya dönük şu mesajı ayarlayın: *"X'i otomatik olarak doğrulayamıyorum, ancak Y yapabilirim."*

---

## Geliştirici Kuralı: Her Döngünün Bir Çıkışı Olmalıdır

Eğer tek bir şey hatırlayacaksanız, o da şu olsun:

**Her ajan döngüsünün tasarlanmış bir çıkışı (exit) olmalıdır — başarı, geri dönüş (fallback), sorma (ask) veya devretme (escalate).**

*"Sonsuza kadar dene"* değil.
*"Umarım model çözer"* değil.

Çıkış koşulları ürünle ilgili alınan kararlardır:
* Kullanıcılar ne kadar gecikmeye (latency) tahammül edebilir?
* Ne kadar belirsizlik kabul edilebilirdir?
* İnsan müdahalesi (override) ne zaman devreye girer?
* Hangi kısmi cevaplar faydalıdır?

İşte böyle cin çarpmış (possessed) gibi değil, profesyonel hissettiren ajanlar inşa edersiniz.

---

## Sonuç: Durmak Bir Özelliktir

Ajanlar sıkışıp kalır, çünkü onlara frensiz bir hareket kabiliyeti verdik.

Öyleyse frenler inşa edin:
* Bütçeler
* İlerleme sinyalleri
* Durum makineleri (state machines)
* Geri dönüş merdivenleri (fallback ladders)
* Doğrudan (explicit) devretme (escalation) yolları

Ajanınızın sonsuz otonomiye ihtiyacı yoktur.
Karmaşık (messy) gerçeklik altında **öngörülebilir davranışlara** ihtiyacı vardır.

---
**CTA:** Kendi sisteminizde bir ajan döngüsü — araç tekrarları, plan çalkantısı, bitmek bilmeyen doğrulamalar — gördünüz mü? Neye benzediğini yorumlara yazın. O kalıp için somut bir durdurma kuralı seti önereyim.