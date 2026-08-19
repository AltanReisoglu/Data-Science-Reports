# vc-agent

AutoGen üzerine kurulan **VC pipeline agent** sistemi — halka açık kaynakları
tarayarak girişim keşfeden, skorlayan, izleyen ve yatırım notu üreten çok-ajanlı
sistem. Yanında, temelini oluşturan AutoGen araştırması ve ölçüm POC'u.

---

## Ne nerede

| Klasör | İçerik |
|---|---|
| **`docs/`** | Araştırma ve planlar — okuma sırası aşağıda |
| **`poc/`** | Beş AutoGen orkestrasyon desenini aynı görevde ölçen çalışır POC |
| **`pipeline/`** | VC sisteminin kendisi — gateway, ajanlar, toplayıcılar, MCP köprüsü |
| `.venv/` | Kurulu ortam — `autogen 0.7.5` · `google-adk 2.6.3` · `mcp 1.29` |

### Okuma sırası

| # | Belge | Ne anlatır |
|---|---|---|
| 1 | [docs/01-autogen-kaynak-haritasi.md](docs/01-autogen-kaynak-haritasi.md) | AutoGen'in durumu (bakım modu), birincil kaynaklar, makaleler |
| 2 | [docs/02-autogen-el-kitabi.md](docs/02-autogen-el-kitabi.md) | **Baştan sona API el kitabı** — 22 bölüm, MCP · sandbox · bellek · aktör modeli |
| 3 | [docs/03-vc-domain-plani.md](docs/03-vc-domain-plani.md) | **VC sistemi: ne ve neden** — huni ekonomisi, kaynaklar, rubrik, izleme |
| 4 | [docs/04-vc-agentic-akis.md](docs/04-vc-agentic-akis.md) | **VC sistemi: nasıl** — katmanlar, ajan topolojisi, dosya düzeni, fazlar |
| 5 | [docs/15-vc-gateway-mimarisi.md](docs/15-vc-gateway-mimarisi.md) | **Gateway** — OpenClaw mimarisi, AutoGen motoru, iki yönlü MCP köprüsü |

| Referans | Ne anlatır |
|---|---|
| [05](docs/05-autogen-core-user-guide.md) · [08](docs/08-autogen-agentchat-user-guide.md) | AutoGen'in iki resmî kılavuzunun **tam metni** (42 + 25 sayfa) |
| [11](docs/11-core-guide-turkce.md) · [10](docs/10-agentchat-turkce.md) | İkisinin Türkçe rehberi, `05:satır` / `08:satır` atıflı |
| [06](docs/06-autogen-incelikleri.md) | **Koşarken bulunmuş 13 tuzak** — en yüksek getirili belge |
| [07](docs/07-kod-rehberi.md) | Kavram → kod köprüsü, dosya dosya |
| [12](docs/12-autogen-bastan-sona.md) · [14](docs/14-autogen-protokoller-ve-farklar.md) | Uçtan uca anlatım · protokoller ve altı framework karşılaştırması |
| [09](docs/09-framework-karsilastirma.md) | Kısa framework karşılaştırması |
| [13](docs/13-openclaw-teknik-analiz.md) | OpenClaw mimari analizi — 15'in kaynağı |
| [20](docs/20-maf-user-guide.md) · [21](docs/21-maf-tasarim-kararlari.md) | **MAF'ın tam metni** — Learn kılavuzunun tamamı (177 sayfa) + 35 tasarım kaydı (ADR) |
| [22](docs/22-maf-turkce.md) | MAF Türkçe rehberi, `20:satır` / `21:satır` atıflı — AutoGen ile mekanizma mekanizma karşılaştırma |
| [github-starred-repos.md](docs/github-starred-repos.md) | 893 starlı reponun kategorize envanteri |

Yeni başlıyorsan: **03 → 04 → 15** oku. AutoGen'i öğrenmek istiyorsan **12**,
sonra **06**. AutoGen'den MAF'a geçiş sorusu için **22**.

---

## Kurulum

Ortam **zaten kurulu**. Sıfırdan kurman gerekirse:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> `requirements.txt`'teki `mcp>=1.24,<2` **pini şart**. `autogen-ext` bu
> bağımlılığa üst sınır koymamış; MCP SDK 2.0 ile
> `ImportError: cannot import name 'RequestContext'` alırsın.

---

## Koşma

### POC — beş deseni ölçer

```bash
cd poc
../.venv/bin/python kiyas.py              # beş desen + karşılaştırma tablosu
../.venv/bin/python desen_5_core_aktor.py # tek desen
```

**API anahtarı gerekmiyor.** Anahtar yoksa `ReplayChatCompletionClient` devreye
girer: yanıtlar önceden yazılı, sonuç deterministik, ağ gerekmez. Gerçek modelle
koşmak için `.env-example`'daki değişkenlerden birini ver.

Beklenen çıktı:

| desen | mesaj | LLM | token |
|---|---:|---:|---:|
| SelectorGroupChat | 8 | 5 | **204** |
| GraphFlow | 11 | 7 | 270 |
| RoundRobinGroupChat | 9 | 6 | 274 |
| Swarm | 14 | 7 | **334** |
| `autogen_core` aktör | 9 | 0 | 0 |

Aynı görev, aynı ajanlar, **%63.7 token farkı** — ödenen şey yönlendirme özerkliği.

### Pipeline — tarama

```bash
.venv/bin/python pipeline/scan.py --query "ai infra" --days 7
```

HN + SEC Form D + GitHub taranır, adaylar çıkar, her biri için paralel
zenginleştirme koşar, skorlu tablo üretilir — **her satırda kaynak linkiyle**.
LLM yapılandırılmamışsa kuru modda koşar: deterministik, ağsız, ücretsiz.

### Gateway — sohbet, oturumlar, onaylar

```bash
.venv/bin/python -m pipeline.server --port 8777
```

| Uç nokta | Ne verir |
|---|---|
| `/` | Arayüz — tarama, sohbet, canlı kontrol |
| `/api/sessions` | Gateway'in tuttuğu bütün konuşmalar |
| `/api/approvals` | Onay bekleyen dışa dönük tool çağrıları |
| `/api/health` | Oturum sayısı, bağlam durumu, karantinadaki hook'lar, OpenClaw |

Durum **repo dışında**: `~/.vcagent` (`VC_STATE_DIR` ile taşınır). İçinde
oturumlar, transcript'ler, audit defteri, taramalar ve bellek workspace'i.

### OpenClaw köprüsü

İki yön, ikisi de MCP. Ayrıntı: [docs/15](docs/15-vc-gateway-mimarisi.md).

**OpenClaw → bize.** Telefonundan sorduğun soru pipeline'a ulaşır:

```bash
openclaw mcp set vc-agent "$(python -c '
import json, pathlib; r = pathlib.Path.cwd()
print(json.dumps({"command": str(r/".venv/bin/python"),
                  "args": ["-m", "pipeline.mcp_server"], "cwd": str(r)}))')"
openclaw mcp probe vc-agent     # → vc-agent: 8 tools
openclaw mcp doctor             # → vc-agent: ok
```

Sekiz tool: `vc_scan_facts` · `vc_company` · `vc_company_live` ·
`vc_search_docs` · `vc_memory_search` · `vc_memory_get` · `vc_start_scan` ·
`vc_status`. Geri almak: `openclaw mcp unset vc-agent`.

**Bizden OpenClaw'a.** Ajan kanal konuşmalarını okuyabilir:

```bash
VC_MCP_OPENCLAW=1 .venv/bin/python -m pipeline.server
```

> **Kapalı gelir, ve gönderme ayrıca onay ister.** `messages_send` ve
> `permissions_respond` varsayılan olarak **bloklanır**; onay `/api/approvals`
> üzerinden verilir ve **tek çağrılıktır**. `VC_ALLOW_OUTBOUND=1` kapıyı tümden
> açar — blast radius'u kabul ettiğini söylemenin tek dürüst yolu.

---

## POC'un bulgusu

`poc/desen_5_core_aktor.py` "aktör modeli hata izolasyonu verir" iddiasını test
ediyor ve **kısmen yanlış** olduğunu gösteriyor: çöken bir handler
`asyncio.gather`'ı erken döndürüyor, `task_done()` çağrılıyor, `stop_when_idle()`
kuyruğu boş sanıyor ve `close()` yarım kalan kardeş handler'ları iptal ediyor.
Sağlam ajanların sonuçları **sessizce kayboluyor** — ne exception yükseliyor ne
uyarı çıkıyor.

Üç koşuda birebir tekrarlandı. Sonuç: **aktör modeli runtime'ı korur, veriyi
korumaz.** VC pipeline'ının paralel zenginleştirme katmanı bu tuzağa doğrudan
maruz; bu yüzden orada bariyere güvenilmiyor, **beklenen dal sayısı sayılıyor**
(bkz. [04](docs/04-vc-agentic-akis.md) §7).

---

## Durum

| Faz | Durum |
|---|---|
| -1 · Yeniden yapılandırma + planlar | ✅ |
| 0 · Veri sözleşmesi + politika kapısı | ✅ |
| 1 · Toplayıcılar (HN, SEC Form D, GitHub) | ✅ |
| 2 · Normalizasyon + dedup | ✅ |
| 3 · Triyaj + GraphFlow paralel zenginleştirme | ✅ |
| 4 · Rubrik + risk denetçisi + skorlayıcı | ✅ |
| 5 · İzleme durum makinesi | ✅ `gateway/cron.py` |
| 6 · Not yazarı + onay kapısı | ⏳ onay kapısı ✅, not yazarı kısmi |
| 7 · MCP sunucusu + OpenClaw kanalı | ✅ **iki yön, canlı doğrulandı** |
| 8 · Denetim, OTel, geri-test | ⏳ audit ✅, OTel ve geri-test yok |

**214 test geçiyor.**

```bash
cd pipeline && ../.venv/bin/python -m unittest discover -s tests -t .
```

Açık kalanlar ve ölçülmemişler: [docs/15](docs/15-vc-gateway-mimarisi.md) §6.
Tam faz listesi: [docs/04](docs/04-vc-agentic-akis.md) §8.

> **Tez hâlâ placeholder.** `config.THESIS.is_placeholder=True` olduğu sürece
> `thesis_fit` ekseni kalibre değil ve her tarama bunu uyarı olarak basıyor.
