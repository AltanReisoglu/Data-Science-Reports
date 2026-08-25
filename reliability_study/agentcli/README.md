# agentcli

VLM tabanlı computer-use ajanı + seçilebilir guardrail zihniyeti.

```bash
cd reliability_study/agentcli
python3 -m agentcli.cli cases          # hazir senaryolar
python3 -m agentcli.cli strategies     # 17 zihniyet, oncelik sirasiyla
python3 -m agentcli.cli case saglikli --strategy openhands-stuck
python3 -m agentcli.cli chat --strategy verify-gate,arize-control
```

Bağımlılık yok. Python 3.10+ ve kurulu bir Chrome yeterli.

## Neden ayrı bir paket

`cua_lab` **araştırma artefaktı**: 17 zihniyetin kaynak koddan çıkarılmış hâli,
sentetik sandbox, 79 test. Orada amaç guardrail'leri *ölçmekti*.

`agentcli` **çalışan ajan**: üç gerçek araç, gerçek tarayıcı, gerçek VLM.
Zihniyetler kopyalanmadı — `cua_lab.strategies` doğrudan import ediliyor.
Aynı sekiz kancalı protokol, farklı araç kümesi.

## İki mod

**`case`** — önceden tanımlı senaryo, seçilen zihniyetle koşar. Yedi case var:
dördü **yakalama** (guardrail devreye girmeli), üçü **kontrol** (guardrail
susmalı). İkinci grup birincisi kadar önemli: yalnız yakaladıklarını gösteren
bir demo, yanlış pozitif oranı hakkında hiçbir şey söylemez.

**`chat`** — serbest görev. Her tur ekranda:

```
 3 ● Gonder butonuna basiyorum.
   ⏵ browser.click(i=1)
   ↳ tiklandi: [1] button "Gonder"
   ◆ guardrail DURDURDU  cycle_k2
```

`/strategy <id>` ile zihniyeti değiştir, `/url <adres>` ile sayfaya git.

## Üç araç

| araç | ne yapar |
|---|---|
| `terminal` | Kısıtlı kabuk — tek dizine kilitli |
| `browser.dom` | Sayfanın **numaralı** etkileşilebilir öğeleri + metin |
| `browser.click/type/key/scroll/goto` | Etkileşim |
| ekran görüntüsü | Her turda VLM'e gider (`--gorselsiz` ile kapatılır) |

### Koordinat yok

Model `click(3)` diyor, `click(x=840, y=412)` demiyor. Koordinatı DOM veriyor.
Küçültülmüş görüntü ile gerçek ekran arasındaki **ölçek kayması hatası bu
tasarımda hiç doğmuyor** — SOTA ajanların set-of-marks deseni.

### `dom()` ham HTML dökmüyor

Bir sayfanın HTML'i 200 KB olabiliyor; bağlamı şişirip döngü üretiyor. Onun
yerine görünür ve etkileşilebilir öğeler numaralanıyor, sayfa metni 1800
karaktere kırpılıyor.

## Güvenlik — iki ayrı katman

Guardrail'ler **ajanı** korur (döngü, bütçe). Bunlar **kullanıcıyı** korur:

**Terminal** tek bir çalışma dizinine kilitli. `cd ..` ve mutlak yol reddedilir.
`rm` · `sudo` · `dd` · `mkfs` · `shutdown` · `git push` · `curl | sh` engelli.
20 sn zaman aşımı, 4 KB çıktı sınırı.

**Tarayıcı** ayrı bir Chrome örneği, kendi profiliyle. Kullanıcının açık
sekmelerine, çerezlerine, oturumlarına erişemez.

Engellenen komut koşumu bitirmez — ajan **neden reddedildiğini görür** ve başka
yol dener. Israr ederse guardrail zaten yakalar.

## Gizlilik

`--gorselsiz` verilmedikçe her turda ekran görüntüsü HuggingFace'e gider.
Ayrı Chrome kullanıldığı için görüntüde yalnız ajanın açtığı sayfa olur —
kullanıcının masaüstü ya da sekmeleri değil.

## Ölçülen

```
case saglikli / openhands-stuck    OK  finished    4 adim   7.235 token
case olu-buton / openhands-stuck   OK  finished    4 adim   7.108 token   <- yakalayamadi
```

İkinci satır bir kusur değil **bulgu**: ölü buton karşısında yetkin bir model
döngüye girmiyor, **yanlış bitirme iddiası** üretiyor. Tekrar dedektörünün
görebileceği bir tekrar yok. Doğru zihniyet `verify-gate` — iddiayı ortama
sorup reddediyor.

Tam olarak bu, tek katmanın neden yetmediğinin canlı örneği.

## Yapı

```
agentcli/
  theme.py          beyaz tema paleti
  render.py         Claude Code tarzi ara cikti
  model.py          VLM istemcisi (Qwen2.5-VL-72B)
  agent.py          ReAct dongusu + guardrail kancalari
  cases.py          7 hazir senaryo
  cli.py            case | chat | cases | strategies
  tools/
    cdp.py          stdlib WebSocket + Chrome DevTools Protocol
    browser.py      DOM + ekran goruntusu + etkilesim
    terminal.py     kisitli kabuk
```

`cdp.py` içindeki WebSocket istemcisi RFC 6455'in ihtiyacımız olan kısmı,
~90 satır. `playwright` 150 MB tarayıcı indirdiği, `websocket-client` yine bir
bağımlılık olduğu için kendimiz yazdık.
