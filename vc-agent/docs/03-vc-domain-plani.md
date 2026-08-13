# 03 — VC Pipeline: Domain Planı

*Bu belge **ne** ve **neden** sorularını cevaplar. **Nasıl** için:
[04-vc-agentic-akis.md](04-vc-agentic-akis.md)*

**Kapsam:** uluslararası · tek kullanıcılı araç · 2026-08-13

---

## 1 — Problem

Bir VC'nin işi iki cümleyle: **doğru girişimi rakiplerden önce bulmak**, ve
**bulduğunu doğru değerlendirmek.** Her ikisi de bilgi işidir ve her ikisi de
şu an insan saatiyle yapılıyor.

Somut darboğazlar:

| Darboğaz | Bugün nasıl çözülüyor | Maliyeti |
|---|---|---|
| Keşif | Twitter/HN/haber takibi, network | Dağınık, kaçırmaya açık |
| Ön eleme | Analist tek tek bakıyor | Saatler, tutarsız |
| Zenginleştirme | Elle araştırma | Şirket başına 1-3 saat |
| İzleme | Elle hatırlama, tablo | Sistematik değil, unutuluyor |
| Not yazımı | Analist yazıyor | Yarım gün |

Bu sistem dördünü de otomatikleştirmiyor — **maliyetini düşürüyor** ve
insanın zamanını huninin en dibine, karar anına kaydırıyor.

---

## 2 — Sistemin özü: maliyet gradyanlı huni

Bir VC pipeline'ı, her katmanı bir öncekinden pahalı olan bir huni:

```
~5.000 ham sinyal/gün  →  toplayıcılar      bedava, deterministik, LLM yok
   ~200 aday           →  triyaj            ucuz model, %95'ini eler
    ~20 zenginleştirme →  paralel analiz    pahalı: 3 dal × LLM + API çağrısı
     ~5 yatırım notu   →  güçlü model       en pahalı üretim
     ~1 partner bakışı →  insan             en pahalı kaynak
```

**Sistemin tek işi: çoğa az harca, aza çok harca.**

Mimarideki her karar bundan türüyor:
- Toplayıcılarda LLM yok → hacim burada, model çağırmak israf
- Triyaj ucuz modelle → ikili karar için yeterli
- Zenginleştirme paralel → az sayıda şirkete çok kaynak, ama eşzamanlı
- Not en güçlü modelle → insanın okuyacağı tek çıktı
- İnsan en sonda → en pahalı kaynak en az kullanılır

Rakamlar hedef; gerçek oranlar ölçümle kalibre edilecek (bkz. §9).

---

## 3 — Dört ilke

### 3.1 İki hata türü eşit değil

| Hata | Görünürlük | Maliyet |
|---|---|---|
| **Kaçırılan iyi girişim** (false negative) | **Görünmez** | Fonun tüm getirisi tek bir kaçırılmış deal'e bağlı olabilir |
| Boşa incelenen kötü girişim (false positive) | Görünür | Bir analist saati |

Sonuç: huninin **üstünde recall**, **altında precision** kovalanır.

Pratik karşılığı — ve sistemin en önemli tek kuralı:

> **Bilgi yokluğu eleme sebebi değildir.**
> Bir şirketi "uymuyor" *kanıtıyla* elersin; "bulamadık" diye eleyemezsin.

Bu yüzden `Skor` şemasında **`eksik_veri` zorunlu alan**. Sistem "ekip puanı 2"
derken "çünkü kurucu bilgisi bulunamadı, şu üç kaynağa bakıldı" demek zorunda.
Düşük puan ile bilgi yokluğu farklı şeylerdir ve karışırlarsa huni sessizce
kötü kararlar üretir.

### 3.2 Tazelik tamlıktan değerli

3 günlük bir sinyal, 3 haftalık tam profilden kıymetli — çünkü rakip VC de aynı
şirketi görüyor. Bir tur kapandıktan sonra öğrenmek, öğrenmemekle aynı.

Sonuç: sistemin ağırlık merkezi **tek seferlik derin araştırma değil**,
sürekli izleme + değişiklik tespiti (§8). Ve kaynak seçiminde "ne kadar erken"
sorusu "ne kadar kapsamlı" sorusundan önce gelir — SEC Form D'nin öne çıkmasının
sebebi bu.

### 3.3 Ürün skor değil, kanıt paketi

Hiçbir yatırımcı "7.4 puan" diye yatırım yapmaz. Bakacağı şey:

> *"Salı günü şu Form D dolduruldu (link), GitHub commit grafiği son 90 günde
> 3× arttı (link), kurucunun önceki şirketi 2023'te satıldı (link)."*

Sonuç: her iddia bir `Kaynak` linki taşır; **kaynaksız cümle nota giremez**.
Skor yalnızca bir **sıralama aracıdır**, karar gerekçesi değil.

### 3.4 Huni aşağı akar, izleme döngüseldir

Elenen şirket ölmez. `atla` kararı bile geri alınabilir olmalı: bugün
pre-seed'de tezine uymayan şirket 6 ay sonra seed'de tam hedef olabilir.

Sonuç: bu bir liste değil, **durum makinesi** (§8). Her şirket bir durumda
oturur ve sinyaller onu durumlar arasında taşır.

---

## 4 — Kaynak envanteri

Hepsi 2026-08-13'te canlı test edildi.

| Kaynak | HTTP | Ne verir | Anahtar | Tazelik |
|---|---|---|---|---|
| **SEC EDGAR Form D** | 200 | ABD'de **gerçek fon turu başvurusu** | Yok (UA zorunlu) | **basından ~15 gün önce** |
| **HN Algolia** | 200 | Fon haberleri, Show HN, lansmanlar | Yok | saatler |
| **GitHub API** | 200 | Repo ivmesi, katkıcı dağılımı, kurucu profili | `gh` mevcut | günlük |
| **arXiv** | 200 | Derin teknoloji girişiminin akademik kökeni | Yok | aylar |
| **DeepWiki MCP** | kurulu | Repo'nun mimari kalitesi | Yok | isteğe bağlı |
| RSS / haber | — | Sektör basını, hızlandırıcı duyuruları | Yok | saatler |
| OpenCorporates | 401 | Resmî sicil | Ücretsiz kademe | — |
| Product Hunt | 403 | Ürün lansmanı | Resmî API | — |

**Anlamlı bir MVP tek kuruş ödemeden kurulabiliyor.** Form D + HN + GitHub üçlüsü
zaten "yeni fon almış, teknik, aktif" girişimleri yakalıyor.

### Form D neden en güçlü sinyal

ABD'de özel sermaye toplayan şirket, turdan sonra **15 gün içinde** SEC'e Form D
bildirmek zorunda. Bu bir yasal yükümlülük, isteğe bağlı bir duyuru değil. Yani:

- Şirket basın açıklaması yapmasa bile görünür
- Basın açıklamasından **önce** görünür
- Tutar, tarih ve yatırımcı sayısı yapılandırılmış olarak gelir

VC'nin "erken haber alma" avantajı tam olarak burada ve API'si bedava.

**Sınırı:** yalnızca ABD. Avrupa/Asya tarafında eşdeğer bir zorunlu bildirim
yok; oralarda basın + GitHub + sicil karışımına düşülüyor. Bu, sistemin bilinen
bir kör noktası ve nota yazılıyor.

---

## 5 — Sinyal taksonomisi

| Tip | Kaynak | Ne anlatır | Tazelik değeri | Gürültü |
|---|---|---|---|---|
| `fon_turu` | Form D, haber | Tur kapandı | **çok yüksek** | düşük |
| `urun_lansmani` | HN, PH, blog | Pazara çıktı | yüksek | orta |
| `repo_ivme` | GitHub | Teknik çekiş | orta (kümülatif) | düşük |
| `ise_alim` | kariyer sayfası | Ölçekleniyor | orta | orta |
| `akademik` | arXiv | Derin teknoloji kökeni | düşük (yavaş) | düşük |
| `haber` | RSS | Genel görünürlük | düşük | **yüksek** |

Tek bir sinyal nadiren yeterli. Sistemin aradığı şey **sinyal birleşimi**:
"arXiv makalesi (2024) + GitHub ivmesi (2026 Q1) + Form D (bu hafta)" üçlüsü,
tek başına bir haberden çok daha güçlü bir adaydır.

---

## 6 — Varlık çözümleme: asıl zorluk

Aynı şirket dört kaynakta dört farklı isimle görünür:

```
SEC Form D  : "ACME AI, INC."
GitHub      : "acme-ai"
HN başlığı  : "Acme"
alan adı    : acme.ai
```

**Yanlış birleştirme, skoru anlamsız kılan tek hatadır.** İki farklı "Nova"
şirketini birleştirirsen ortaya hiç var olmayan bir şirket profili çıkar ve
sistem ona puan verir.

Çözümleme sırası — güçlüden zayıfa:

1. **Alan adı** (en güçlü) — `acme.ai`
2. **GitHub org** — repo'nun sahibi org, homepage alanı
3. **Resmî sicil adı** — Form D'deki tüzel kişi adı
4. **Bulanık isim eşleşmesi** (en zayıf) — yalnızca yukarıdakiler yoksa

**Kural: belirsizse birleştirme.** İki ayrı kayıt bırakmak, yanlış birleştirmekten
iyidir. Belirsizlik `eksik_veri`'ye yazılır ve insan bakışına kalır.

---

## 7 — Tez konfigürasyonu ve skorlama

### 7.1 Tez

Tez kodda değil, `pipeline/ayarlar.py`'de ve **sürüm kontrollü** — "tezi ne
zaman, neden değiştirdik" sorusu cevaplanabilir olmalı:

```
sektor        : ["ai-altyapi", "gelistirici-araclari", "veri"]
asama         : ["pre-seed", "seed"]
cografya      : ["global"]
kirmizi_cizgi : ["savunma", "kumar", "kripto-spekulasyon"]
esikler       : incele >= 18 · takip >= 12
```

### 7.2 Rubrik

**Sabit rubrik, vibes yok.** Aynı şirket iki koşuda iki farklı puan alırsa
sistem güvenilmez olur.

| Eksen | 0 | 3 | 5 |
|---|---|---|---|
| **Tez uyumu** | Sektör/aşama dışı | Kısmi örtüşme | Tam hedef profil |
| **Ekip** | Kurucu bilgisi yok | Alanda deneyim var | Önceki exit / derin uzmanlık |
| **İvme** | Sinyal yok | Tek kaynakta hareket | Çok kaynakta 90 günde artan |
| **Teknik derinlik** | Kanıt yok | Çalışan ürün | Açık kaynak + özgün yaklaşım + akademik köken |
| **Zamanlama** | Pazar doymuş | Rekabet var | Yeni açılan pencere |

Kurallar:
- Her eksen puanı **en az bir `Kaynak`** göstermek zorunda
- Kaynak yoksa puan **0 değil** → `eksik_veri`'ye kayıt, ortalamadan düşer (§3.1)
- Eşikler konfigürasyonda; tez değişince rubrik yeniden kalibre edilir
- `toplam ≥ 18` → `incele` · `12–17` → `takip` · `< 12` → `atla`

---

## 8 — İzleme: durum makinesi

```
      yeni ──► aday ──triyaj──► incele ──not──► portfoy_adayi
                 │                │                    │
                 └──► atla        └──► takip ◄─────────┘
                                        │
                            180 gün sinyal yok → sogudu
                            yeni fon turu      → incele (yükselt)
```

Günlük tarama, önceki snapshot ile karşılaştırılır. **Anlamlı değişiklik**
tanımları:

| Değişiklik | Ne anlatır | Aksiyon |
|---|---|---|
| Yeni Form D | Yeni tur, basından önce | `takip` → `incele` yükselt |
| GitHub ivme kırılması | Teknik çekiş hızlanıyor | Skor yeniden hesapla |
| İşe alım artışı | Ölçekleniyor | Skor yeniden hesapla |
| Kurucu değişikliği | Ayrılık riski | Uyarı + insan bakışı |
| 180 gün sessizlik | Soğuma | `sogudu` durumuna al |

Sonuncusu önemli: **sessizlik de bir sinyaldir** ve sistemin bunu ayrıca
raporlaması gerekir, yoksa takip listesi ölü kayıtlarla şişer.

---

## 9 — Başarı metrikleri

| Metrik | Ne ölçer | Neden |
|---|---|---|
| **Şirket başına maliyet** (token + süre) | Huni ekonomisi | §2 çalışıyor mu |
| **Sinyalden uyarıya süre** | Tazelik | §3.2 tutuyor mu |
| **Triyaj eleme oranı** | Huni daralması | %95 hedefi gerçekçi mi |
| **Kanıtlı iddia oranı** | Kaynaksız cümle sızıyor mu | §3.3 |
| **Geri-test recall** | Kaçırma oranı | **En önemlisi** |

### Geri-test: tek gerçek kalite ölçüsü

Son 6 ayın **bilinen** fon turlarını al, sistemi o tarihe kadarki veriyle
koştur ve sor: *kaçını yakalardık?*

Bu, §3.1'deki "kaçırma görünmezdir" problemini görünür kılan tek yöntem.
Diğer bütün metrikler sistemin *çalıştığını* ölçer; bu metrik *işe yaradığını*
ölçer.

---

## 10 — Çıktı artefaktları

| Artefakt | Ne zaman | Nereye |
|---|---|---|
| **Anlık uyarı** | Yüksek skorlu yeni sinyal | MCP → OpenClaw → Telegram/Slack |
| **Günlük özet** | Her sabah | Markdown + kanal |
| **Yatırım notu** | `incele` kararı | `notlar/<sirket>-<tarih>.md` |
| **Pipeline CSV** | İstenince | Ortak toplantısı tablosu |

Uyarının telefonda çalması tesadüfi bir tercih değil: tazelik ilkesi (§3.2)
ancak teslim de hızlıysa anlamlı. Terminalde biten bir script kullanılmaz.

---

## 11 — Sınırlar

Bu bölüm mimarinin parçası; `pipeline/politika.py` bunları **kodla** zorluyor.

| Yapılır ✅ | Yapılmaz ❌ |
|---|---|
| Halka açık API'ler | Paywall/giriş arkasını atlamak |
| `robots.txt`'e uyan, oran sınırlı erişim | Agresif tarama, IP rotasyonu, bot tespiti aşma |
| Resmî başvurular (Form D) | **LinkedIn scraping — ToS ihlali** |
| Şirket düzeyi bilgi | Kurucuların özel hayatı, iletişim bilgisi |
| Halka açık mesleki geçmiş | Kapalı profillerden veri çekme |
| `User-Agent`'ta gerçek kimlik | Kimlik gizleme |

**LinkedIn yerine ne kullanıyoruz:** kurucuların **kendi yayınladığı** kaynaklar
— GitHub profili, kişisel site, konferans konuşmaları, arXiv makaleleri, şirketin
"hakkımızda" sayfası — ve resmî sicil kayıtları. Ciddi VC araçları da bu sınırlar
içinde çalışıyor; iş daralmıyor.

**Kişisel veri (KVKK / GDPR):** Kurucu adı ve halka açık mesleki geçmiş işlenir.
Kişi kayıtları `kisi_kaydi` tablosunda **ayrı** tutulur, şirket kaydından
bağımsız silinebilir. Saklama süresi konfigürasyonda tanımlı.

**Yatırım tavsiyesi değildir.** Üretilen her not bir taslaktır; kararı insan
verir. Her notun başına bu ibare basılır ve insan onayı olmadan hiçbir belge
sistemden dışarı çıkmaz.

---

## 12 — Bilinen kör noktalar

Dürüst olmak sistemi güçlendirir; bunlar nota da yazılıyor:

| Kör nokta | Etki | Azaltma |
|---|---|---|
| Form D yalnız ABD | Avrupa/Asya turları geç görülür | Basın + GitHub ağırlığı artırılır, nota yazılır |
| Stealth mod şirketler | Hiç görünmezler | Kurucu sinyali (arXiv, GitHub) tek yol |
| Kapalı kaynak ürünler | Teknik derinlik ölçülemez | `eksik_veri`, puan düşürülmez |
| İngilizce olmayan basın | Yerel turlar kaçar | Sonraki fazda dil genişletme |
| Sinyal ≠ kalite | Gürültülü şirket yüksek ivme puanı alabilir | `RiskDenetcisi` + insan onayı |

---

*Sonraki belge: [04-vc-agentic-akis.md](04-vc-agentic-akis.md) — bu domain
modelinin AutoGen ile nasıl kodlanacağı.*
