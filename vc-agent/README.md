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
| **`pipeline/`** | VC sisteminin kendisi *(Faz 0'dan itibaren inşa ediliyor)* |
| `.venv/` | Kurulu ortam — `autogen 0.7.5` · `google-adk 2.6.3` · `mcp 1.29` |

### Okuma sırası

| # | Belge | Ne anlatır |
|---|---|---|
| 1 | [docs/01-autogen-kaynak-haritasi.md](docs/01-autogen-kaynak-haritasi.md) | AutoGen'in durumu (bakım modu), birincil kaynaklar, makaleler |
| 2 | [docs/02-autogen-el-kitabi.md](docs/02-autogen-el-kitabi.md) | **Baştan sona API el kitabı** — 22 bölüm, MCP · sandbox · bellek · aktör modeli |
| 3 | [docs/03-vc-domain-plani.md](docs/03-vc-domain-plani.md) | **VC sistemi: ne ve neden** — huni ekonomisi, kaynaklar, rubrik, izleme |
| 4 | [docs/04-vc-agentic-akis.md](docs/04-vc-agentic-akis.md) | **VC sistemi: nasıl** — katmanlar, ajan topolojisi, dosya düzeni, fazlar |
| — | [docs/github-starred-repos.md](docs/github-starred-repos.md) | 893 starlı reponun kategorize envanteri |

Yeni başlıyorsan: **03 → 04** oku. AutoGen'i öğrenmek istiyorsan **02**.

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

### Pipeline — VC sistemi

*Faz 0'da yazılmaya başlanacak.*

```bash
.venv/bin/python pipeline/tara.py --sektor "ai infra" --gun 7
```

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
| 0 · Veri sözleşmesi + politika kapısı | ⏳ sırada |
| 1 · Toplayıcılar (HN, SEC Form D, GitHub) | — |
| 2-8 | — |

Tam faz listesi: [docs/04](docs/04-vc-agentic-akis.md) §8.
