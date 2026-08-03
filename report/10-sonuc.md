# 10. Sonuç

## 10.1 Tezin cevabı

Rapor şu iddiayla başladı:

> **Bir ajanın yeteneği modelinin değil, bağlamının fonksiyonudur.**

İncelenen altı mekanizma bu iddiayı farklı açılardan destekliyor:

| Mekanizma | Tezi nasıl destekliyor |
|---|---|
| **Tool** | Şeması bağlamda olmayan tool, adı bilinse dahi çağrılamaz (§03.4.2) |
| **Skill** | Alan bilgisi ağırlıklarda değil, koşullu enjekte edilen metinde (§04.4) |
| **Memory** | Oturumlar arası süreklilik modelde değil, dosya sisteminde (§05) |
| **Getirme** | "Bilgiye erişim" bir yetenek değil, bir bütçe kararı (§06.9) |
| **Artefakt** | 5 MB'lık veri, bağlama hiç girmeden işlenebiliyor (§07.7) |
| **Bağlam basıncı** | Aynı model, bağlam yönetimine göre farklı performans gösteriyor (§08) |

En keskin **gözlemsel** kanıt §03.4.2'dedir: aynı oturumda, aynı model, aynı ağırlıklarla, **tek değişken şemanın bağlamda olup olmamasıdır** ve bu tek değişken tool'un kullanılabilirliğini belirler.

En keskin **ölçülmüş** kanıt ise sektörden gelir (§01.1): LangChain, modeli sabit tutup yalnızca harness'i değiştirerek Terminal-Bench'te +13,7 puan; Vercel, tool'larının %80'ini **kaldırarak** daha yüksek güvenilirlik ve 3,5× düşük gecikme bildirmiştir. İkincisi tezin en doğrudan sonucudur: **yetenek eksiltmek performansı artırabilir**, çünkü kısıt yetenekte değil dikkat bütçesindedir.

---

## 10.2 Bulgular

| # | Bulgu | Bölüm |
|---|---|---|
| **1** | ReAct ile native tool calling arasındaki fark bir desen tercihi değil, katman farkıdır. Halüsinasyon, paralellik, doğrulama, hata sinyali ve muhakeme görünürlüğü — beşi de bu ayrımın doğrudan sonucudur | §03.2 |
| **2** | Docstring bir yorum değil, prompt'tur. Şema üretimi otomatiktir; anlam üretimi değildir | §03.3 |
| **3** | Bilme = bağlamda olma. Şeması yüklenmemiş bir tool, adı bilinse de çağrılamaz | §03.4 |
| **4** | "Döngüyü kim çevirir" ile "halüsinasyonu ne engeller" bağımsız eksenlerdir | §03.5 |
| **5** | Framework'ler tool calling'i sarmalar, icat etmez. Modelin gördüğü şey framework'ten bağımsızdır | §03.6 |
| **6** | Skill özel bir model yeteneği değil, bir yükleme protokolüdür: ~200 token sürekli maliyet ↔ ~50.000 token koşullu yük | §04.9 |
| **7** | Hafızada asıl problem saklama değil, geri çağırmanın **yetkisi** ve **tazeliğidir** | §05.4 |
| **8** | Hafıza, bağlam mühendisliğinin zaman eksenine yayılmış hâlidir; değeri sıkıştırma oranındadır | §05.10 |
| **9** | İyi organize edilmiş bir depo, bağlama hiç girmeden ajana bilgi verir — dizin yapısı bir prompt'tur | §06.8 |
| **10** | Kod, bağlamın sıkıştırma katmanıdır. Veriyi değil, veriyi işleyen kodu bağlama almak sıkıştırmayı yüzlerce kata çıkarır | §07.8 |
| **11** | Kritik olan kırpmanın kendisi değil, **görünürlüğüdür.** Sessiz kırpma, eksik veriyle tam güvenle cevap üretir | §08.2 |
| **12** | Cache ekonomisi prompt tasarımını doğrudan şekillendirir (donmuş git durumu örneği) | §08.10 |
| **13** | Bağlam kalitesi çıktı kalitesinden bağımsız ölçülebilir ve ölçülmelidir | §09.8 |
| **14** | Harness değişiminin itici gücü yalnızca model kapasitesi değil, **iş hacmidir.** Modeller güçlendikçe harness basitleşebilir; ajanlara verilen iş büyüdükçe karmaşıklaşmak zorundadır. İkinci kuvvet birinciden hızlı büyüdüğü sürece bağlam yönetimi merkezi problem olarak kalır | §01.1 |
| **15** | Progressive disclosure'ın kendisi de ölçek sınırına çarpar: yüzlerce skill'in yalnızca ad+açıklaması bile bağlam vergisidir. Çözüm bir katman daha aşağıda — **aranabilir indeks**, ki haritanın kendisi de bağlam dışında tutulsun | §04.2 |

---

## 10.3 Sentez: tek bir problemin çözümleri

Her mekanizma, aynı kısıtın farklı bir cevabıdır: **bağlam penceresi en pahalı kaynaktır.**

| Mekanizma | Çözdüğü problem | Yöntemi | Bedeli | Eksen |
|---|---|---|---|---|
| Tool şeması | Modelin dış dünyaya erişimi | Prefix'e sabit metin | Her istekte token | — |
| `defer_loading` + tool search | Çok tool → token vergisi | İsim önde, şema sonradan | Ekstra tur | Uzaysal |
| Skill (3 katman) | Alan bilgisi her zaman gerekmiyor | Koşullu enjeksiyon | Yüklenince ağır ve **geri alınamaz** | Uzaysal |
| **Skill indeksi (katman 0)** | Yüzlerce skill'in ad+açıklaması bile pahalı | Harita bile bağlam dışında; sorgula | Arama turu, indeks bakımı | Uzaysal |
| Memory | Oturumlar arası süreklilik | Dosya + indeks, seçici okuma | Bayatlama, yetki yüzeyi | Zamansal |
| JIT retrieval / grep | Veri bağlama sığmıyor | Tanımlayıcı tut, gerekince yükle | Gecikme, tur maliyeti | Uzaysal |
| Artefakt işleme | Binary/XML bağlama sığmıyor | Veriyi değil, işleyen kodu al | Sandbox gereksinimi | Uzaysal |
| **PTC (sandbox)** | Çok adımlı iş, konuşmada pahalı | 20 tool çağrısı tek çalıştırmada; ara veri değişkenlerde | Adım adım gate'lenemez | Uzaysal |
| Çıktı kırpma | Tek sonuç bağlamı doldurabilir | Sert tavan + **görünür** uyarı | Sayfalama turu | Uzaysal |
| Context editing | Bayat tool sonuçları | Eskiyi **sil** | Geri dönülemez | Zamansal |
| Compaction | Konuşma pencereyi aşar | Eskiyi **özetle**; büyük çıktıları diske taşı | Detay kaybı, poisoning kalıcılaşabilir | Zamansal |
| Subagent | Keşif gürültüsü | Ayrı pencere, sadece sonuç dön | Soğuk başlangıç | Uzaysal |
| Prompt caching | Aynı prefix tekrar işleniyor | Sabit önde, değişken arkada | Sıra disiplini | — |

**Ortak payda:** her satır ya *"bağlama girmesin"*, ya *"gerektiğinde girsin"*, ya da *"girdiyse ucuza girsin"* diyor.

**İki eksen (§08.8):** *uzaysal* mekanizmalar detayın orchestrator'ın penceresine **hiç girmemesini** sağlar; *zamansal* mekanizmalar girmiş olanı zaman içinde yönetir. İkisi ikame değil, tamamlayıcıdır — yalnızca zamansal savunma kuran bir sistem, her ara sonucu önce içeri alıp sonra özetlemek zorunda kalır.

---

## 10.4 Mimari sonuç: karmaşıklık döngünün çevresinde

Raporun en genel gözlemi §01.8'de kurulmuş, bölümler boyunca doğrulanmıştır:

```
ReAct dönemi:   zekâ prompt şablonundaydı
                 → döngü karmaşık, çevresi boş

Bugün:          zekâ bağlam yönetimindedir
                 → döngü ≈20 satır, çevresi katman katman
```

Bu yüzden "hangi ajan deseni" sorusu artık ayırt edici değildir. ReAct, plan-execute ve türevleri, modellerin tool kullanmayı bilmediği dönemin prompt teknikleriydi. Bugün herkes aynı basit döngüyü kullanıyor; farklılaşma **döngünün çevresinde** — bağlamın nasıl kurulduğunda, neyin dışarıda tutulduğunda, durumun nerede yaşadığında.

Framework karşılaştırmasının (§03.6) ortaya çıkardığı üç eksen, bu gözlemin operasyonel hâlidir:

1. **Döngünün sahibi kim** — kod mu, model mi
2. **Durum nerede yaşıyor** — bağlamda mı, dışarıda mı
3. **Araya girme noktaları birinci sınıf mı**

---

## 10.5 Raporun sınırları

Dürüstlük gereği belirtilmelidir:

| Sınır | Etkisi |
|---|---|
| **Gözlem, modelin kendi bağlamıyla sınırlı** | Harness'in iç implementasyonu (serialization şablonu, kırpma algoritması, cache anahtarlama) gözlemlenemedi; bu noktalarda çıkarım yapıldı |
| **Sayısal iddialar kısmen temsilî** | §02.5'teki `usage` değerleri yapıyı gösterir; kesin sayılar §09'daki deneylerle üretilmelidir |
| **Deneyler henüz çalıştırılmadı** | §09'daki dört deney tasarlandı, sonuçları rapora dahil edilmedi |
| **Kaynakların bir kısmı ikincil aktarım** | *Lost in the Middle*, *How Long Contexts Fail* ve *ReAct* birincil metinlerinden doğrulanmalıdır |
| **Tek sağlayıcı odaklı** | Mekanizmalar Anthropic API'si üzerinden incelendi; OpenAI/Google karşılıkları yalnızca §03.6'da yüzeysel karşılaştırıldı |
| **Bilgi kesimi** | Framework detayları (CrewAI, ADK) Mayıs 2026 itibarıyladır; bu kütüphaneler hızlı değişmektedir |

---

## 10.6 Sonraki adımlar

Raporu güçlendirecek çalışmalar, öncelik sırasıyla:

1. **§09'daki dört deneyi çalıştır.** Özellikle Deney 4 (arama hunisi vs tam okuma) tek bir grafikle raporun ana iddiasını gösterebilir.
2. **Wire log üret.** `turn-01-request.json` ↔ `turn-02-request.json` diff'i, "bağlam her turda yeniden gönderiliyor" iddiasını gözleme dayandırır.
3. **İkincil kaynakları birincilden doğrula.** Özellikle context rot / lost-in-the-middle ayrımı akademik referansla desteklenmeli.
4. **Üç framework'te minimal örnek çalıştır** ve ürettikleri `tools` dizisini yan yana koy — hepsinin aynı JSON'a indiğini gösteren ekran görüntüsü, §03.6'nın kanıtı olur.
5. **Probe detektörünü gerçek bir uzun oturumda dene** ve drift göstergelerinin çıktı bozulmasından önce ateşlenip ateşlenmediğini ölç.

---

## 10.7 Kapanış

Alanın kendi ifadesiyle:

> *"the challenge isn't just crafting the perfect prompt — it's thoughtfully curating what information enters the model's **limited attention budget** at each step."*
> — Anthropic, Eyl 2025

Ve bu rapor boyunca gösterildiği gibi, o "curating" işi tek bir tekniğin adı değil; tool şemasından dosya sistemi düzenine, skill katmanlarından cache sırasına kadar uzanan **bir tasarım disiplinidir.**

Modeller güçlendikçe bu disiplinin bazı parçaları basitleşiyor — daha az kuralcı prompt, daha çok otonomi. Ancak ters yönde ikinci bir kuvvet var ve daha hızlı büyüyor: **ajanlara verilen iş.** Her ek tool çağrısı, skill çağrımı, arama sonucu ve yürütme çıktısı bağlam penceresine ekleniyor (Bulgu 14).

Bu iki kuvvetin bileşkesi, harness'i sürekli yeniden inşa edilen bir yapı hâline getiriyor. Glean'in üçüncü nesil harness'i, Anthropic'in yazısından sonra ürünleşen tool search ve sunucu taraflı compaction, Vercel'in tool setini %80 küçültme kararı — hepsi aynı denklemin farklı noktalarındaki cevaplar.

Temel kısıt ise yerinde duruyor: bağlam sonludur, dikkat bütçelidir, ve **neyin gireceğine karar vermek her zaman bir mühendislik işi olarak kalacaktır.**

---

**Devamı:** [§11 — Güncel durum ve harness atlası](11-guncel-durum-ve-harness-atlasi.md). Bu bölüm raporun bulgularını iki yönde tamamlar: alanın Ağustos 2026 hâlinden çekirdeğe inen sekiz katmanlık bir harita (ACE, öğrenilmiş sıkıştırma, bağlam grafikleri, harness mühendisliği), ve on beş harness yapısının her birinin **wire düzeyinde tam giriş/çıkış formu**.
