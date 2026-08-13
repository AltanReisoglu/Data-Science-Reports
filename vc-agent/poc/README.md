# AutoGen — Çok-Ajanlı Sistem POC'u

AutoGen'in **beş farklı çok-ajan desenini** aynı görev üzerinde koşan, hepsini aynı
metriklerle ölçen çalışır bir POC. Amaç tutorial tekrarı değil: aynı işi farklı
orkestrasyon modelleriyle yaptırıp **farkın sayısal karşılığını** görmek.

- **Baştan sona el kitabı:** [../docs/02-autogen-el-kitabi.md](../docs/02-autogen-el-kitabi.md)
  — kurulum, tüm API yüzeyi, MCP, sandbox, bellek, kalıcılık, aktör modeli,
  süper agent blueprint'i ve kaynakça
- Bağlam ve kaynak haritası: [../docs/01-autogen-kaynak-haritasi.md](../docs/01-autogen-kaynak-haritasi.md)

> **Sürüm notu:** AutoGen v0.7.5 ile yazıldı ve doğrulandı. AutoGen 2026 Nisan'dan beri
> [bakım modunda](https://github.com/microsoft/autogen); halefi Microsoft Agent Framework.
> Bu POC tam da o yüzden değerli — "neden konsolide edildi" sorusunun cevabı ölçümlerde.

---

## Kurulum ve koşu

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python kiyas.py                # beş deseni koşar + karşılaştırma tablosu
python desen_3_swarm.py        # tek bir deseni koşar
```

**API anahtarı gerekmiyor.** Anahtar yoksa POC `ReplayChatCompletionClient` ile
çalışır: LLM yanıtları önceden yazılıdır, sonuç deterministiktir, ağ gerekmez.
Bu bir kısayol değil — desenlerin **kontrol akışını** modelin keyfinden ayırıp
karşılaştırmayı tekrarlanabilir kılan bilinçli bir tasarım kararı.

Gerçek modelle koşmak için `.env-example`'daki değişkenlerden birini ver:
`OPENAI_API_KEY` ya da `OPENROUTER_API_KEY` (ikincisi ücretsiz key ile de çalışır).

---

## Ölçülen sonuç

Aynı görev: *"'gelir' serisini çek, istatistiklerini hesapla, kısa rapor yaz."*
Aynı ajanlar. Değişen tek şey: kimin ne zaman konuşacağına kim karar veriyor.

| desen | mesaj | LLM | tool | token | durma nedeni |
|---|---:|---:|---:|---:|---|
| RoundRobinGroupChat | 9 | 6 | 2 | 274 | `RAPOR_TAMAM` |
| **SelectorGroupChat** | **8** | **5** | 2 | **204** | `RAPOR_TAMAM` |
| Swarm | 14 | 7 | 4 | **334** | `RAPOR_TAMAM` |
| GraphFlow | 11 | 7 | 3 | 270 | grafik tamamlandı |
| autogen_core aktör | 9 | 0 | 0 | 0 | runtime idle |

**En pahalı ile en ucuz arasında %63.7 fark var.** Ödenen şey zekâ değil,
*yönlendirme özerkliği*:

- **Selector** yönlendirmeyi bir Python fonksiyonuyla yapar → 0 ek token, gereksiz
  ajanı (Eleştirmen) atlar.
- **Swarm**'da yönlendirmeye ajanın kendisi karar verir; her devir bir tool çağrısı
  + bir LLM turu harcar ve o tur hiç iş üretmez.
- **RoundRobin** kimseyi atlayamaz: Eleştirmen'in söyleyecek bir şeyi olmasa da sıra
  ona uğrar.

Bu tablo `kiyas_sonuc.json`'a da yazılır.

---

## Dosyalar

| Dosya | Ne gösteriyor |
|---|---|
| `motor.py` | Model istemcisi (gerçek/replay) + ölçüm altyapısı |
| `araclar.py` | Deterministik tool'lar — `FunctionTool` ve düz fonksiyon, iki kayıt yolu |
| `desen_1_roundrobin.py` | Sabit sıra. Harness'ın beyni yok |
| `desen_2_selector.py` | `selector_func` ile deterministik yönlendirme |
| `desen_3_swarm.py` | `Handoff` ile ajanın kendi kendine devri |
| `desen_4_graphflow.py` | `DiGraph` + **paralel dal ve join bariyeri** |
| `desen_5_core_aktor.py` | `autogen_core`: aktör modeli, pub/sub, RPC, hata deneyi |
| `kiyas.py` | Hepsini koşar, tabloyu ve JSON'u üretir |

---

## Desen 5 neden en önemlisi

İlk dört desen AgentChat katmanında — tutorial'ların bittiği yer. `desen_5` bir
katman aşağıda, `autogen_core`'da: **v0.4'te AutoGen'in baştan yazılma sebebi** olan
aktör modeli. Orada LLM yok; mesele zekâ değil, çalışma zamanı.

Ve orada POC'un tek **gerçek bulgusu** var. "Aktör modeli hata izolasyonu verir"
iddiası test edildi — üç deney:

- **Deney A** — çöken bir ajan sağlam iki ajanla aynı topic'e aboneyken:
  `stop_when_idle()` sağlam ajanlar bitmeden dönüyor. Sebep, `_process_publish`
  içindeki `asyncio.gather` bir handler exception fırlatınca hemen dönüyor ve
  hemen ardından `task_done()` çağrılıyor → kuyruk, kardeş handler'lar hâlâ
  çalışırken "boşaldı" sayılıyor. **Senkronizasyon bariyeri kırılıyor.**
- **Deney B** — aynı iş, çöken ajan olmayan bir topic'te: sonuçlar eksiksiz.
- **Deney C** — Deney A'daki yayından hemen sonra `close()`: yarım kalan handler'lar
  iptal ediliyor, iki sonuç da **kalıcı olarak kayboluyor**.

Sonuç: aktör modeli **runtime'ı korur, veriyi korumaz**. Kısmi başarısızlık
sessizdir — ne exception yükselir ne "eksik sonuç" uyarısı çıkar, yalnızca log'da
bir satır kalır. Doğrulama katmanını uygulama yazarı eklemek zorundadır.

Bu tam olarak [MAST taksonomisinin](https://arxiv.org/abs/2503.13657) *system design*
ve *task verification* kümelerine giren bir hata. Yani POC, kaynak haritasındaki
tezi laboratuvar koşullarında yeniden üretiyor: **çok-ajan sistemlerde hatanın
kaynağı model kalitesi değil, harness tasarımı.**

---

## Sınırlar (bilerek kapsam dışı)

Kod yürütme (`DockerCommandLineCodeExecutor`), bellek (`Memory`), MCP
(`McpWorkbench`), state kaydet/yükle, dağıtık gRPC runtime ve Magentic-One bu
POC'ta yok. Kaynak haritasının §5'indeki proje planında bunlar sonraki adımlar;
buradaki `motor.py` + `kiyas.py` iskeleti hepsini aynı metriklerle ölçecek şekilde
tasarlandı — yeni bir desen eklemek `calistir()` yazıp `kiyas.DESENLER`'e bir satır
eklemekten ibaret.
