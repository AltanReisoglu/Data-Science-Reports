# cua_lab — Computer-Use Agent + Seçilebilir Güvenilirlik Stratejileri

Bir computer-use ajanı ve onun etrafına takılabilen **17 farklı güvenilirlik zihniyeti**.
Amaç "en iyi guardrail"i bulmak değil; hangi zihniyetin hangi hatayı yakaladığını,
hangisini kaçırdığını aynı görev üzerinde ölçmek.

```bash
python3 -m cua_lab.cli list-strategies
python3 -m cua_lab.cli run --strategy openhands-stuck --scenario dead_button
python3 -m cua_lab.cli run --strategy verify-gate,arize-control --scenario healthy --model liar
python3 -m cua_lab.cli compare --scenario dead_button --strategies all
python3 -m cua_lab.cli replay <trace.jsonl>
```

Bağımlılık yok, API anahtarı yok. Python 3.10+.

## Neden

Anthropic'in referans computer-use döngüsünde (`computer-use-demo/loop.py`) **hiçbir
kontrol yok**: `while True`, tek çıkış modelin araç çağırmayı bırakması. Tur sayacı yok,
döngü tespiti yok, bütçe yok. `none` stratejisi bu davranışı taban çizgisi olarak koruyor —
diğer stratejilerle farkı ölçmek için.

Ayrıntı: [`docs/computer_use_zihniyet.md`](docs/computer_use_zihniyet.md)

## Hangi zihniyet

On yedi zihniyetin tamamı **en basitten en karmaşığa** altı seviyede anlatılıyor —
sayaç → pencere → dünya → şekil → kademe → karar. Sıfırdan başlıyorsanız 1 numaradan
okumaya başlayın; her seviye bir öncekinin kör noktasını kapatıyor.

[`docs/zihniyetler.md`](docs/zihniyetler.md) — anlatım + her madde için teknik karşılığı
[`docs/stratejiler.md`](docs/stratejiler.md) — referans tablosu

## Ölçülen fark

Aynı görev, aynı bozuk ortam (`dead_button` — tıklanan buton hiçbir şey yapmıyor):

```
strateji            durum               sebep                adim   token       $
none                CEILING             hard_ceiling          300  105000   1.050
openhands-stuck     STUCK               cycle_k2                3    1050   0.011
strands-entropy     STUCK               low_diversity           6    2100   0.021
loopguard-dignity   NEEDS_INPUT         abstain_need_input      7    2450   0.025
agentbudget-dollar  BUDGET_EXHAUSTED    budget_steps           12    3850   0.038
arize-control       BUDGET_EXHAUSTED    max_steps              13    4200   0.042
galileo-breaker     CEILING             hard_ceiling          300  105000   1.050   <- kor nokta
verify-gate         CEILING             hard_ceiling          300  105000   1.050   <- kor nokta
```

Sekiz zihniyet, **sekiz farklı sonuç** — tablo bu yüzden anlamlı. Ve son iki satır
kasıtlı: `galileo-breaker` hata oranına bakıyor, `dead_button`da hata yok;
`verify-gate` bitirme iddiasını bekliyor, ajan hiç "bitirdim" demiyor. **Kör noktaları
tabloda görünüyor** — "tek katman yetmez"in kanıtı bu iki satır.

Her zihniyetin kendi yakaladığı ayrı bir senaryo var:

```
broken_tool + patient   galileo-breaker  DEGRADED  tool_circuit_open   (arac kalici bozuk)
healthy     + liar      verify-gate      DEGRADED  verify_failed       (yapmadan "bitirdim")
dead_button + alternating  strands-entropy  STUCK  low_diversity       (A-B-A-B)
```

Sağlıklı koşumda (`healthy`), **meşru retry'da** (`flaky`) ve `silent_success`'te
**sekiz sütunun sekizi de `none` ile birebir aynı** — aynı adım, aynı token.
Guardrail çalışan bir koşuma tek token bindirmiyor. Bir dedektörün ikinci sınavı budur:
yakalamak kadar, yakalamaması gerekeni rahat bırakmak. `tests/test_faz2.py` bunu
her strateji × her meşru senaryo için kilitliyor.

## Yapı

| Yol | Ne |
|---|---|
| `cua_lab/events.py` | 17 eylemlik uzay (`computer_20251124`), olay modeli |
| `cua_lab/detect/guardrails.py` | Ortak altyapı: 5 dedektör + 5 eksenli bütçe |
| `cua_lab/sandbox/fake.py` | Sentetik ortam, 4 senaryo, sıfır bağımlılık |
| `cua_lab/loop.py` | ReAct döngüsü, 8 strateji kancası |
| `cua_lab/strategies/` | `src/` = ben_ekledim türevli · `harness/` = üretim harness'ı türevli |
| `cua_lab/trace.py` | İterasyon başına bir span, JSONL |

**Ortak framework, farklı zihniyet:** dedektörler ve bütçe sayaçları tek yerde; stratejiler
onları farklı biçimlerde kullanıyor. Ayrım mekanizmada değil, ne zaman ve nasıl müdahale
edildiğinde.

## Senaryolar

| Senaryo | Ortam ne yapıyor | Test ettiği şey |
|---|---|---|
| `healthy` | normal | yanlış pozitif kontrolü |
| `dead_button` | buton tıklanıyor, hiçbir şey olmuyor, hata da vermiyor | sessiz durgunluk |
| `flaky` | ilk iki denemede hata, sonra çalışıyor | meşru retry döngü sayılmamalı |
| `silent_success` | aynı ama koşum OK biter | "hakkında ticket açılmayan hata" |

Hata desenleri **ortamdan** geliyor, modelden değil — model gerçekten sıkışıyor.

## Durum

- [x] Faz 0 — computer-use zihniyet çalışması
- [x] Faz 1 — çekirdek + `none` + `openhands-stuck` + CLI
- [ ] Faz 2 — A ailesi (ben_ekledim, 11 strateji)
- [ ] Faz 3 — B ailesi (harness, kalan 5) + `compare` genişletme
- [ ] Faz 4 — ekran durgunluğu dedektörü + testler
- [ ] Faz 5 — Docker sandbox
- [ ] Faz 6 — HF Inference API
