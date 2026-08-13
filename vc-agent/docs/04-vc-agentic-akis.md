# 04 — VC Pipeline: Agentic Akış ve Kod Planı

*Bu belge **nasıl** sorusunu cevaplar. **Ne** ve **neden** için:
[03-vc-domain-plani.md](03-vc-domain-plani.md)*

**Temel:** AutoGen v0.7.5 · OpenAI-uyumlu endpoint · tek kullanıcılı araç
API referansı: [02-autogen-el-kitabi.md](02-autogen-el-kitabi.md)

---

## 1 — Katman mimarisi

```
┌ KATMAN 1 · TOPLAYICILAR ─────────────────── LLM YOK ────────────────────┐
│  hn.py · sec_edgar.py · github.py · arxiv.py · rss.py                   │
│  ortak taban: politika kapısı · oran sınırı · disk cache · retry        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ list[Sinyal]
┌────────────────────────────▼──────────────── LLM YOK ───────────────────┐
│ KATMAN 2 · NORMALİZASYON                                                │
│  varlık çözümleme (alan adı → GitHub org → sicil → bulanık) · dedup     │
│  ChromaDB: "bu şirketi daha önce gördük mü"                             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ Girisim
┌────────────────────────────▼────────────────────────────────────────────┐
│ KATMAN 3 · AJANLAR (AutoGen)                                            │
│                                                                         │
│   Triyaj ──selector_func──►  GraphFlow paralel zenginleştirme           │
│     │  (ucuz model)            ├─ TeknikAnalist (GitHub + DeepWiki MCP) │
│     │                          ├─ PazarAnalisti (HN + RSS)              │
│     └─ atla (LLM harcamadan)   └─ EkipAnalisti  (GitHub profil + arXiv) │
│                                        │ join: activation_condition=all │
│                                        ▼                                │
│                     RiskDenetcisi → Skorlayici → NotYazari              │
│                                        │                                │
│                            UserProxyAgent  ← partner onayı              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ YatirimNotu
┌────────────────────────────▼────────────────────────────────────────────┐
│ KATMAN 4 · TESLİM                                                       │
│  mcp_sunucu.py → OpenClaw → Telegram/Slack · markdown · CSV · SQLite    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Neden ilk iki katmanda LLM yok

Domain planındaki huni ekonomisinin (§2) doğrudan sonucu — hacim burada.
Ayrıca üç pratik fayda:

1. **Test edilebilirlik** — kayıtlı fixture'larla ağsız test yazılabilir
2. **Temiz ölçüm** — token farkı desenden gelir, modelin keyfinden değil
3. **Maliyet** — 5.000 sinyale model çağırmak israf

Bu, POC'taki `ReplayChatCompletionClient` disiplininin aynısı: **ölçmek
istemediğin her değişkeni sabitle.**

---

## 2 — Veri sözleşmesi

`pipeline/semalar.py` — **ilk yazılacak dosya.** Ajanlar arasında serbest metin
dolaşmaz; her şey Pydantic ve `output_content_type` ile zorlanır.

```python
class Kaynak(BaseModel):
    ad: str                              # "sec_form_d" | "hn" | "github" | ...
    url: str                             # doğrulanabilir link — ZORUNLU
    alindi: datetime
    guven: Literal["resmi", "birincil", "ikincil"]

class Sinyal(BaseModel):
    tip: Literal["fon_turu","urun_lansmani","ise_alim","repo_ivme","akademik","haber"]
    ozet: str
    tarih: datetime
    kaynak: Kaynak
    ham: dict                            # orijinal yanıt — denetim için

class Girisim(BaseModel):
    ad: str
    alan_adi: str | None
    tanim: str | None
    sektor: list[str]
    ulke: str | None
    kurulus_yili: int | None
    sinyaller: list[Sinyal]
    github: str | None
    kurucular: list[str]                 # yalnız halka açık isim

class Skor(BaseModel):
    tez_uyumu: int                       # 0-5
    ekip: int
    ivme: int
    teknik_derinlik: int
    zamanlama: int
    toplam: int
    gerekce: dict[str, str]              # eksen → tek cümle + kaynak
    eksik_veri: list[str]                # ← ZORUNLU (domain §3.1)
    karar: Literal["takip","incele","atla"]

class YatirimNotu(BaseModel):
    girisim: Girisim
    skor: Skor
    tez: str
    riskler: list[str]
    sorular: list[str]                   # kurucuya sorulacaklar
    kaynakca: list[Kaynak]
```

`Kaynak.url` ve `Skor.eksik_veri` zorunlu olduğu için, "kaynaksız iddia" ve
"sessiz bilgi eksikliği" **şema seviyesinde** engellenmiş oluyor.

---

## 3 — Ajan topolojisi ve model kademelendirme

| Ajan | AutoGen deseni | Model kademesi | Tool / kaynak |
|---|---|---|---|
| **Triyaj** | `selector_func` + LLM yedeği | **ucuz** | — |
| TeknikAnalist | `AssistantAgent` | orta | GitHub API + **DeepWiki MCP** |
| PazarAnalisti | `AssistantAgent` | orta | HN + RSS arama |
| EkipAnalisti | `AssistantAgent` | orta | GitHub profil, arXiv |
| RiskDenetcisi | `AssistantAgent` | orta-güçlü | — |
| Skorlayici | `output_content_type=Skor` | orta-güçlü | — |
| NotYazari | `output_content_type=YatirimNotu` | **en güçlü** | — |
| Partner | `UserProxyAgent` | — | onay kapısı |

Model kademelendirme AutoGen'de doğal: her `AssistantAgent` kendi
`model_client`'ını alır. `ayarlar.py`'de üç kademe tanımlanır ve
`motor.istemci(kademe)` fabrikası döndürür.

### Neden Selector + GraphFlow birlikte

POC ölçümü (aynı görev, aynı ajanlar):

| desen | mesaj | LLM | token |
|---|---:|---:|---:|
| **SelectorGroupChat** | 8 | 5 | **204** |
| GraphFlow | 11 | 7 | 270 |
| RoundRobin | 9 | 6 | 274 |
| Swarm | 14 | 7 | **334** |

Triyaj kararı **dinamik** (hangi uzman gerekli) → Selector, en ucuz desen.
Zenginleştirme **sabit ve paralel** → GraphFlow, paralellik veren tek desen.
İkisi kendi güçlü oldukları yerde.

Zenginleştirme grafı `TeamTool` ile paketlenir: alt takımın 20 mesajlık iç
konuşması Triyaj'ın bağlamını kirletmez (el kitabı §8).

---

## 4 — Dosya düzeni

```
vc-agent/
├── .venv/                    ← kurulu (autogen 0.7.5, adk 2.6.3, mcp 1.29)
├── requirements.txt          ← mcp>=1.24,<2 pini ZORUNLU (§7)
├── docs/                     ← 01-04 belgeler
├── poc/                      ← beş desenin ölçüm POC'u (referans)
└── pipeline/
    ├── ayarlar.py            tez · eşikler · oran sınırları · model kademeleri
    ├── semalar.py            §2 veri sözleşmesi
    ├── politika.py           KaynakPolitikasi — robots · oran · kara liste · denetim
    ├── motor.py              model fabrikası + Olcum   ← poc/motor.py'den
    ├── toplayicilar/
    │   ├── temel.py          cache · retry · UA · oran sınırı · politika kapısı
    │   ├── hn.py  sec_edgar.py  github.py  arxiv.py  rss.py
    ├── normalize.py          varlık çözümleme + dedup
    ├── ajanlar/
    │   ├── triyaj.py  teknik.py  pazar.py  ekip.py  risk.py
    │   ├── skorlayici.py  not_yazari.py
    ├── graf.py               GraphFlow: fan-out + join
    ├── izleme.py             durum makinesi · diff · snapshot
    ├── mcp_sunucu.py         pipeline'ı MCP tool'u olarak açar
    ├── tara.py               CLI giriş noktası
    ├── testler/              fixture'lı toplayıcı testleri
    └── veri/                 SQLite + ChromaDB + cache   (gitignore)
```

---

## 5 — Yeniden kullanılacak mevcut kod

**Hiçbiri yeniden yazılmayacak.**

| Nereden | Ne | Nasıl kullanılacak |
|---|---|---|
| `poc/motor.py` | `Olcum` dataclass + `olc()` sarmalayıcı | Şirket başına token/süre ölçümü |
| `poc/desen_4_graphflow.py` | Fan-out + join + **ajan başına ayrı istemci** | `graf.py`'nin iskeleti |
| `poc/desen_2_selector.py` | `selector_func` + LLM yedeği kalıbı | `ajanlar/triyaj.py` |
| **`../demo-brain-agent/taskboard.py`** | WAL'lı SQLite · lease'li kuyruk · checkpoint · olay kaydı · bayat iş kurtarma (601 satır) | İzleme kuyruğu ve iş dağıtımı |
| `../demo-brain-agent/scheduler.py` | Periyodik koşum | Günlük tarama döngüsü |
| `../saf-motorlar/kiyas.py` | Numaralı soru + JSON çıktı deseni | Geri-test raporu (domain §9) |
| `../.mcp.json` (DeepWiki) | Zaten kurulu MCP server | TeknikAnalist'in repo analizi |

`taskboard.py` özellikle önemli: izleme katmanının ihtiyaç duyduğu her şey
(kuyruk, lease, checkpoint, çökme kurtarma) orada zaten test edilmiş hâlde var.

---

## 6 — Politika kapısı ve denetim

Tüm dış çağrılar tek kapıdan geçer — domain §11'in kodla zorlanması:

```python
class KaynakPolitikasi:
    def izin_var_mi(self, url: str) -> bool:     # robots.txt + kara liste
    def bekle(self, kaynak: str) -> None:        # kaynak başına oran sınırı
    def kaydet(self, istek, yanit) -> None:      # denetim kaydı
```

Kara liste sabit: `linkedin.com`, `facebook.com`, giriş gerektiren her şey.
`izin_var_mi` bunlara **her zaman `False`** döner ve bu bir testle korunur.

Ajan tarafında `InterventionHandler` (el kitabı §17) mesaj hattına takılır ve
her tool çağrısını denetim kaydına yazar → *"bu puan nereden geldi"* sorusu
her zaman cevaplanabilir.

---

## 7 — Bilinen tuzaklar (bu oturumda ölçüldü)

| Tuzak | Karşılık |
|---|---|
| **Paralel dalda sessiz veri kaybı** — çöken bir handler `asyncio.gather`'ı erken döndürüyor, `task_done()` çağrılıyor, `stop_when_idle()` bariyeri kırılıyor, `close()` yarım kalanları iptal ediyor | Zenginleştirme sonrası **beklenen dal sayısı sayılır**; eksikse `eksik_veri`'ye yazılır. **Bariyere güvenilmez.** (POC `desen_5_core_aktor.py`, üç koşuda tekrarlandı) |
| `autogen-ext` `mcp>=1.11.0` üst sınırsız → MCP SDK 2.0 ile `ImportError: RequestContext` | `requirements.txt`'e **`mcp>=1.24,<2`** |
| `Handoff` tool adı küçük harfe düşüyor (`transfer_to_veriuzmani`) | Elle yazma, `Handoff(target=X).name` ile üret |
| Sonsuz ajan döngüsü = gerçek fatura | Her takımda `MaxMessageTermination` + `TokenUsageTermination` sigortası |
| `description` boşsa `SelectorGroupChat` kör seçim yapar | Her ajana anlamlı `description` |
| `max_tool_iterations=1` varsayılan | Zincirleme tool gereken ajanlarda artır |

---

## 8 — Fazlar

| Faz | İçerik | Çıktı | Süre |
|---|---|---|---|
| **-1** ✅ | Yeniden yapılandırma + iki doküman | `vc-agent/` | tamam |
| **0** | `semalar.py` · `politika.py` · `ayarlar.py` | Veri sözleşmesi + kapı | ½ gün |
| **1** | Toplayıcılar: HN, SEC Form D, GitHub — LLM yok, fixture'lı test | Gerçek sinyal listesi | 1½ gün |
| **2** | Normalizasyon + varlık çözümleme + dedup | Tekilleştirilmiş `Girisim` | 1 gün |
| **3** | Triyaj + GraphFlow paralel zenginleştirme | Ajan katmanı çalışıyor | 2 gün |
| **4** | Rubrik + `RiskDenetcisi` + `Skorlayici` | Skorlu tablo | 1 gün |
| **5** | İzleme durum makinesi (taskboard.py üstünde) | Watchlist + diff | 1 gün |
| **6** | `NotYazari` + `UserProxyAgent` onay kapısı | Yatırım notu | 1 gün |
| **7** | MCP sunucusu + OpenClaw kanalı | Telefonda uyarı | ½ gün |
| **8** | Denetim, OTel, geri-test | Ölçüm raporu | ½ gün |

### Hafta 1 teslimi (Faz 0-3)

```bash
cd vc-agent
.venv/bin/python pipeline/tara.py --sektor "ai infra" --gun 7
```
→ HN + SEC + GitHub taranır → adaylar çıkar → her biri için paralel
zenginleştirme koşar → **skorlu tablo, her satırda kaynak linkleriyle**.

Tek başına gösterilebilir bir demo; sonraki fazlar üstüne biner.

---

## 9 — Doğrulama

| Ne | Nasıl |
|---|---|
| **Toplayıcılar** | `pytest pipeline/testler/` — kayıtlı fixture'larla, **ağsız**. Canlı duman testi: `python -m pipeline.toplayicilar.sec_edgar --gun 3` |
| **Politika kapısı** | LinkedIn URL ver → `izin_var_mi` **False** dönmeli. Ardışık çağrıda oran sınırı gecikmesi ölçülür |
| **Veri sözleşmesi** | Kaynaksız `Skor` üretmeyi dene → Pydantic reddetmeli |
| **Paralel graf (§7)** | 3 dallı zenginleştirmede bir dalı **kasten çökert** → sonuç 2 olmalı **ve** `eksik_veri`'ye yazılmalı |
| **Uçtan uca** | `tara.py --sektor "ai infra" --gun 7` → ≥1 `Girisim`, her iddia `Kaynak` linkli |
| **Ölçüm** | `Olcum` çıktısı: şirket başına token/süre; triyaj eleme oranı %90+ |
| **Geri-test** | Son 6 ayın bilinen turlarını koştur → recall raporu (`kiyas.py` deseninde JSON) |
| **MCP teslimi** | `openclaw mcp add` ile bağla → OpenClaw'dan *"bu hafta hangi fintech'ler"* sorusuna cevap gelmeli |

---

## 10 — Riskler

| Risk | Karşılık |
|---|---|
| **Halüsinasyon** — ajan olmayan bir tur uydurur | Kaynaksız cümle nota giremez (şema zorluyor) |
| **Varlık karışması** | Alan adı → GitHub org → sicil sırası; belirsizse **birleştirme** |
| **Kaynak çürümesi** — API değişir/kapanır | Toplayıcılar izole; biri düşerse diğerleri koşar, eksik kaynak nota yazılır |
| **Sessiz kısmi başarısızlık** | §7'deki dal sayımı; "0 sonuç" her zaman sebep yazar |
| **Oran sınırı / IP engeli** | Tek kapı `KaynakPolitikasi` + disk önbelleği + saygılı gecikme |
| **Maliyet patlaması** | Triyaj ucuz modelle eler + termination sigortaları + `Olcum` takibi |
| **Yanlı tez** | Rubrik konfigürasyonda ve sürüm kontrollü |

---

*Önceki belge: [03-vc-domain-plani.md](03-vc-domain-plani.md) · AutoGen API
referansı: [02-autogen-el-kitabi.md](02-autogen-el-kitabi.md)*
