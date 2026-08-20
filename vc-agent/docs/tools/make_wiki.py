#!/usr/bin/env python
"""Şirket wiki'si: tek Markdown, içinde çizilmiş şemalar.

    python docs/tools/make_wiki.py
    → docs/25-atlas-wiki.md              (tek dosya, gömülü SVG'ler)
    → docs/diagrams/wiki/*.excalidraw    (aynı şemalar, düzenlenebilir)

### Neden tek dosya

Bu depodaki yirmi dört belge birbirine bağlı ve sırayla okunuyor. Wiki'nin
okuyucusu farklı: **arayarak geliyor.** Bir soruyla açıyor, cevabı alıp
kapatıyor, ve büyük ihtimalle bir daha açmıyor. Yirmi dosyaya bölünmüş bir
wiki'de o kişi doğru dosyayı bulamıyor; tek dosyada `Ctrl+F` yetiyor.

### Neden gömülü SVG, neden ayrıca .excalidraw

Markdown `.excalidraw` dosyasını **render etmiyor** — Confluence de, GitHub de,
Obsidian'ın çoğu kurulumu da. Bir wiki'de görünmeyen şema, olmayan şemadır.
O yüzden çizimler `<svg>` olarak gömülü: her yerde açılıyor, dış dosyaya
bağımlı değil, ve elle çizilmiş görünüyor — çünkü bu bir düşünme aracı, bitmiş
bir ürün değil.

Ama gömülü SVG **düzenlenemiyor.** Birinin bir kutuyu değiştirmesi gerektiğinde
kaynağa ihtiyacı var. O yüzden aynı şemalar `.excalidraw` olarak da yazılıyor
ve wiki her şemanın altından onlara bağ veriyor: okumak için SVG, değiştirmek
için Excalidraw.

Şemalar `docs/diagrams/figures.py`'den geliyor — desteyle **aynı** çizimler.
Wiki'ye ayrı şema çizmek, aynı sistem hakkında iki ayrı resim demekti ve
ikisinden biri eskirdi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diagrams"))

import figures  # noqa: E402

SCENES = ROOT / "diagrams" / "wiki"

# Hangi belge nereye. Üçü aynı motoru kullanıyor ve şemaları paylaşıyor:
# `f_layers` iki wiki'de geçiyorsa aynı `.excalidraw` dosyasına bağlanıyor,
# ikinci bir kopya üretilmiyor.
WIKIS: list[tuple[str, str]] = [
    ("25-atlas-wiki.md", "atlas"),
    ("26-autogen-maf-wiki.md", "autogen_maf"),
    ("27-openclaw-wiki.md", "openclaw"),
]


def svg(name: str, caption: str) -> str:
    """Bir şemayı gömülü SVG olarak yaz, altına kaynak bağını koy.

    Genişlik `100%` DEĞİL: bazı wiki motorları `<svg>`'yi blok yapmıyor ve
    şema paragrafın içine biniyor. `viewBox` zaten oranı taşıyor.
    """
    body = getattr(figures, name)()
    link = f"diagrams/wiki/{name}.excalidraw"
    return (
        f'<div align="center">\n{body}\n</div>\n\n'
        f"<sub>▲ {caption} · düzenlemek için: [`{name}.excalidraw`]({link}) "
        f"→ excalidraw.com'a sürükle</sub>\n"
    )


def scene_from_svg(name: str) -> dict:
    """SVG'yi Excalidraw sahnesine çevirmek yerine, sahneyi **gömülü** taşı.

    Excalidraw `image` elemanı bir dosyayı `files` sözlüğünde data-URI olarak
    tutuyor. Şemayı eleman eleman yeniden üretmek (her kutu için ~20 alan)
    çizimin iki ayrı yerde yaşaması demekti; burada tek kaynak `figures.py`
    kalıyor ve Excalidraw dosyası onun taşıyıcısı oluyor.

    Sonuç: açılıyor, taşınıyor, yanına yeni kutu çizilebiliyor. Var olan
    kutuların *içi* düzenlenemiyor — dürüst sınır, ve wiki bunu yazıyor.
    """
    import base64

    body = getattr(figures, name)()
    data = base64.b64encode(body.encode("utf-8")).decode("ascii")
    fid = f"fig-{name}"
    # viewBox'tan doğal ölçü; sahne o oranda açılsın.
    import re

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', body)
    w, h = (float(m.group(1)), float(m.group(2))) if m else (600.0, 300.0)
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "vc-agent/docs/tools/make_wiki.py",
        "elements": [{
            "type": "image", "id": fid, "fileId": fid,
            "x": 0, "y": 0, "width": w, "height": h,
            "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": 1, "version": 1, "versionNonce": 1,
            "isDeleted": False, "boundElements": None, "updated": 1,
            "link": None, "locked": False, "status": "saved", "scale": [1, 1],
        }],
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {fid: {"mimeType": "image/svg+xml", "id": fid,
                        "dataURL": f"data:image/svg+xml;base64,{data}",
                        "created": 1, "lastRetrieved": 1}},
    }


def anchors(text: str) -> str:
    """`## 1 · Başlık` başlıklarının önüne AÇIK çapa koy, ve içindekileri ona bağla.

    Otomatik çapa üreten her motorun kuralı farklı: GitHub `· ` işaretini atıp
    iki boşluk bıraktığı için `#1--sözlük` üretiyor, python-markdown boşlukları
    tek tireye indirip `#1-sözlük`, Confluence büsbütün başka bir şey. Ölçüldü:
    üç wiki'de 40 bağın **hepsi** en az bir motorda kırıktı.

    Açık `<a id="s1">` her üçünde de aynı çalışıyor, ve içindekiler artık
    başlığın metnine değil **sırasına** bağlı — başlık yeniden yazıldığında bağ
    kırılmıyor.
    """
    import re

    n = 0
    out = []
    for line in text.split("\n"):
        m = re.match(r"^## (\d+) · ", line)
        if m:
            n = int(m.group(1))
            out.append(f'<a id="s{n}"></a>')
        out.append(line)
    return "\n".join(out)


def used_figures(text: str) -> list[str]:
    """Metinde gerçekten geçen şemalar — elle tutulan bir listeden değil.

    İlk hâlinde `FIGURES` diye ayrı bir liste vardı ve `f_gateway` orada olup
    metinde hiç kullanılmıyordu: on iki `.excalidraw` üretiliyor, on biri
    bağlanıyordu. İki yerde tutulan bir liste, ikisinden biri eskiyene kadar
    doğru görünüyor — ve eskidiğini kimse fark etmiyor çünkü fazladan dosya
    hata vermiyor.
    """
    import re

    seen: list[str] = []
    for name in re.findall(r"wiki/(f_[a-z_]+)\.excalidraw", text):
        if name not in seen:
            seen.append(name)
    return seen


def atlas() -> str:
    """Wiki metni. Her bölüm bir soruya cevap veriyor, sırayla değil ARANARAK."""
    return f"""# Atlas — ajan altyapısı wiki'si

> **Bu ne:** KKB'de bir ajan sistemi kurarken bilinmesi gerekenler. Tek dosya,
> arayarak okunmak için. `Ctrl+F` ile gel, cevabı al, kapat.
>
> **Kaynak:** `vc-agent` deposu · 484 test · her sayı ölçüldü.
> Etiketler: **[ölçüldü]** koşturuldu · **[kaynak]** birincil metinden ·
> **[teyitsiz]** okundu, koşturulmadı.
>
> **Şemalar:** okumak için gömülü, değiştirmek için her birinin altında
> `.excalidraw` bağı var — dosyayı [excalidraw.com](https://excalidraw.com)'a
> sürüklemek yetiyor.

---

## İçindekiler

1. [Sözlük — beş terim](#s1)
2. [AutoGen: üç katman](#s2)
3. [Aktör modeli: ajanlar nasıl konuşuyor](#s3)
4. [Tool döngüsü ve sessiz varsayılanlar](#s4)
5. [Workbench: tool'ların tek kapısı](#s5)
6. [Onay kapısı](#s6)
7. [Takımlar ve faturaları](#s7)
8. [Kod yürütme ve Docker](#s8)
9. [Zamanlayıcı](#s9)
10. [OpenClaw'dan alınanlar](#s10)
11. [Denetim: iki kayıt hattı](#s11)
12. [Çerçeve seçimi](#s12)
13. [Bilinen sınırlar](#s13)

---

## 1 · Sözlük

Beş terim; wiki'nin geri kalanı bunları kullanıyor.

| Terim | Ne demek |
|---|---|
| **Ajan** | Bir model + talimat + tool listesi + hafıza. Nesne olarak bir Python sınıfı. |
| **Tool** | Ajanın çağırabildiği fonksiyon. Model fonksiyonu görmüyor, **tarifini** görüyor. |
| **Runtime** | Ajanlar arası mesajı taşıyan postane. Ajan ajanı çağırmıyor; runtime'a mesaj veriyor. |
| **Workbench** | Tool listesi değil, tool **kaynağı**. "Elimde ne var" diye her turda sorulabiliyor. |
| **Harness** | Dil modelini iş yapabilen bir ajana çeviren runtime iskelesi — oturum, onay, bellek, zamanlama. |

---

## 2 · AutoGen: üç katman

{svg("f_layers", "AutoGen'in üç katmanı")}

* **`autogen_core`** — aktör modeli. Kimlik, runtime, topic, abonelik.
* **`autogen_agentchat`** — günlük iş. Hazır ajan, beş takım tipi, on bir sonlandırma koşulu.
* **`autogen_ext`** — dış dünya. Model istemcileri, MCP, kod yürütücüler.

**Kural:** yukarıdan başla. AgentChat'in çözdüğü bir problemi core'da yeniden
çözmek, aynı işi daha az testle yapmak demek. Aşağı inmek zorunda değilsin ama
**inebildiğini bilmek** bir güvence — bu projede paralel dal kaybını AgentChat'te
çözemedik, core'a inip çözdük.

---

## 3 · Aktör modeli

{svg("f_actor", "Ajan ajanı çağırmıyor — runtime'a mesaj veriyor")}

Bir ajan başka bir ajanın nesnesini elinde tutmuyor. Runtime'a mesaj veriyor,
teslimatı runtime yapıyor. Bunun bedeli var — araya bir katman giriyor ve
*"kim kimi çağırdı"* sorusunun cevabı yığın izinde görünmüyor. Karşılığında
üç şey kazanıyorsun: yeni ajan eklemek çağıran kodu **değiştirmiyor**, bütün
mesajlar tek noktadan geçtiği için müdahale ve ölçüm oraya takılıyor, ve aynı
sınıftan istediğin kadar örnek bedava.

### İki iletişim biçimi — fark adresleme değil, **hata**

| | Doğrudan (`send_message`) | Yayın (`publish_message`) |
|---|---|---|
| Alıcı | tek adres | topic'e abone olan herkes |
| Dönüş değeri | **var** | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Son satır bir tasarım kararı: bir sonucu bekleyeceksen doğrudan, bir olayı
duyuracaksan yayın. Karıştırırsan hata sessizce kaybolur.

---

## 4 · Tool döngüsü

{svg("f_tool_loop", "Model tool ister · kapı · çalıştır · sonucu gör · döngü")}

### Sessiz varsayılanlar — en pahalı tuzak

Ajan bir tool çağırdıktan sonra **kaç kez daha** dönebilir? Hiçbir çerçeve aynı
cevabı vermiyor, ve hiçbiri bunu öne çıkarmıyor. Hepsi kurulu paketten
okundu **[ölçüldü]**:

| Çerçeve | Alan | Varsayılan |
|---|---|---:|
| **AutoGen** | `max_tool_iterations` | **1** |
| OpenAI Agents SDK | `Runner.run(max_turns=)` | 10 |
| CrewAI | `Agent.max_iter` | 25 |
| **MAF** | `DEFAULT_MAX_ITERATIONS` | **40** |
| LangGraph | `recursion_limit` | 10007 |
| Google ADK | `LoopAgent.max_iterations` | **sınırsız** |

**AutoGen'de varsayılan 1:** ajan tool'u çağırır, sonucu görür ve **durur** —
cevabı hiç yazmaz. Hata da vermez.

> Tehlike iki uçta da aynı: **varsayılanı yazmadan koşturmak.** Bir uçta ajan
> sessizce hiçbir şey yapmıyor, öbür uçta sessizce durmuyor.

### Diğer sessiz varsayılanlar

* `model_context` verilmezse ajanın **belleği yok** — ve hata vermiyor.
* Sonlandırma koşulu yoksa takım **sonsuza kadar** konuşuyor; fatura gerçek.
* `description` boş bırakılan ajan, `SelectorGroupChat`'te **kör** seçiliyor.

---

## 5 · Workbench

{svg("f_workbench_component", "Üç kaynak, tek arayüz")}

`tools=[...]` bir **liste**, `workbench=` bir **kaynak**. Liste ajan yazılırken
donuyor; kaynak her turda sorulabiliyor. İkisi birlikte kullanılamıyor —
`ValueError: Tools cannot be used with a workbench.`

**Her turda ne oluyor:**

```
wb.list_tools()  →  JSON şemalar  →  model çağrısına `tools=` diye gider
```

Model fonksiyonu görmüyor; **adını, tarifini ve parametre şemasını** görüyor.
Üç sonuç:

1. **Docstring gerçekten arayüz.** Modelin o tool'a *ne zaman* uzanacağına karar
   verdiği tek metin o.
2. **Şemalar her turda ödeniyor.** 17 tool = her istekte 17 şema.
3. **Bir tool'u listeden çıkarmak** prompt'u ucuzlatıyor — *kapılamak* ile
   *filtrelemek* ayrı kararlar.

**Neden kapıyı buraya koyduk:** workbench, yerel bir Python fonksiyonuyla uzak
bir MCP tool'unu **aynı gören tek yer**. Ve kural, ajan yazılırken **var olmayan**
tool'lar için de geçerli — "şu isimler tehlikeli" listesi tam burada başarısız
olurdu.

---

## 6 · Onay kapısı

{svg("f_gate", "Çağrı geçmeden önce duran tek nokta")}

### Üç kural

**① Engellenen çağrı hata *döndürüyor*, fırlatmıyor.** Ajan reddedildiğini
öğreniyor, söyleyebiliyor, başka yol deneyebiliyor. İstisna turu bitirir ve
insana hiçbir şey anlatmazdı.

**② Onay bir kez tüketiliyor.** İmza `(tool, argümanlar)` üstünde. Aynı çağrı
ikinci kez geldiğinde **yeniden soruluyor**. "Bir daha sorma" bir kolaylık
kararıdır ve düzenlenmiş bir kurumda varsayılanı açık olmamalıdır.

**③ Bozulan bekçi kapanır, açılmaz.** Kanca kendi istisnasında `block: True`
döndürüyor.

### Kapılamak ≠ filtrelemek

| | Ne yapar | Ne zaman doğru |
|---|---|---|
| **Kapılamak** | tool görünür kalır, çağrı reddedilir | ajan *"mesaj atardım ama onayınız lazım"* diyebilir |
| **Filtrelemek** | `list_tools`'tan çıkar, prompt'a hiç girmez | prompt maliyeti · meşru kullanımı olmayan tool |

Filtrelenmiş tool **adıyla çağrılsa da reddediliyor** — *liste bir ipucudur,
zorlama noktası değil.*

---

## 7 · Takımlar

{svg("f_teams", "Beş takım tipi — değişen tek şey: sırayı kim belirliyor")}

Aynı görev, aynı ajanlar, yalnız orkestrasyon değişiyor **[ölçüldü]**:

| Desen | Sırayı kim belirliyor | Mesaj | LLM | Tool | Token |
|---|---|---:|---:|---:|---:|
| **SelectorGroupChat** | model her turda seçiyor | 8 | 5 | 2 | **204** |
| GraphFlow | önceden çizilmiş DAG | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | sırayla, kararsız | 9 | 6 | 2 | 274 |
| **Swarm** (handoff) | ajanın kendisi devrediyor | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen şey zekâ değil **yönlendirme özerkliği**: ajanlara
"kime devredeceğine sen karar ver" dediğin an fatura artıyor, çünkü her devir
bir tur ve her tur bir model çağrısı.

> Kıyasa çevirisi: **Agents SDK'nın tek modeli olan handoff, AutoGen'in en
> pahalı desenidir.** Tek desenli bir çerçeve seçmek, o desenin faturasını da
> seçmektir.

---

## 8 · Kod yürütme

{svg("f_code_executors", "Yerel yürütücü ve konteyner")}

### Rol: yirmi ikinci tool değil, **kaçış kapağı**

Model önce mevcut tool'lara bakıyor; sorulanı karşılayan bir tool **yoksa**
Python yazıp çalıştırıyor. Ayrım tarifle zorlanıyor: tarif *"kod çalıştırır"*
deseydi ajan her hesabı yeniden icat eder, yirmi bir tool boşa çalışırdı.

### Ömür: konteyner **sürece** ait, çağrıya değil

Sunucu açılırken bir konteyner kalkıyor, kapanırken iniyor. Çağrı başına
konteyner kaldırmak 2–3 saniye ve bunun tamamı kullanıcının beklediği süreye
eklenirdi.

**Bedeli:** konteyner turlar arasında **durum taşıyor**. İzolasyon konteyner ile
host arasında; tur ile tur arasında değil.

### Güvenlik — ölçüldü, ve iyi görünmüyor

| | Değer |
|---|---|
| kullanıcı | **root** (uid=0) |
| ağ | **bridge** — dışarı çıkıyor (pypi.org'a `200` alındı) |
| salt okunur kök | hayır |
| bellek / CPU / PID sınırı | **yok** |
| düşürülen yetki | **hiçbiri** |
| ayrıcalıklı | hayır ✔ |

Hiçbiri tercih değil: `DockerCommandLineCodeExecutor`'da bu parametrelerin
**hiçbiri yok**.

**Buna karşılık:** varsayılan kapalı · her koşuda insan onayı · onay kartı ağ
erişimini açıkça yazıyor · onay **kodun imzasına** bağlı · 60 sn zaman aşımı.

> Gerçek savunma sandbox değil, **kapı**. Bu wiki'de *"sandbox güvenli"* cümlesi
> kurulmuyor.

### Onay neden saklanan metni koşturuyor

Kapının reddi turu bitiriyor. Onay o turu geri getiremiyor, ve modelden kodu
yeniden istemek işe yaramıyor — **ölçüldü: aynı soru iki farklı program üretti**
(imzalar `029f4d1f…` ve `107fdfd1…`). Onaylananla çalışanın aynı olmasının tek
yolu, çalıştırılacak olanın **onaylanan metin** olması.

---

## 9 · Zamanlayıcı

{svg("f_task_stack", "Zamanlama yığını")}

**AutoGen'de zamanlama diye bir kavram yok** — ve bu bir eksiklik değil, bir
kütüphane saat tutmaz.

Bizde iki katman var, biri bağlı:

* **Çevirmen (bağlı).** Türkçe "ne zaman" ifadesini cron şekline çeviriyor.
  Üç biçim kabul ediyor — `her gün 09:00` · `30dk` · `20dk sonra` — ve
  dördüncüsünü **tahmin etmiyor**, sözdizimini yazıp reddediyor.
* **Yerli zamanlayıcı (yazıldı, bağlanmadı).** 322 satır, 19 test.

### Üç bilinçli kısıt

| Karar | Neden |
|---|---|
| Payload hep `agentTurn` | `command`/`script` de var ama **ikisi de kabuk**; kabuk kararı onay kapısına ait, gece 3'te koşan bir iş tanımına değil |
| `sessionTarget: isolated` | Zamanlanmış koşu birinin konuşmasını ne miras almalı ne kirletmeli |
| `to` asla varsayılan değil | Adres tahmin etmek, yabancıya mektup atmak |

**Kapı yazılanı imzalıyor, çözülmüş zamanı değil.** `"20dk sonra"` her
ayrıştırmada başka bir damga veriyor; sonucun üstündeki imza hiç tutmazdı.

### Dürüst sınır

Zamanlama yalnız OpenClaw'ın Gateway'i koşarken çalışıyor. Sessizce ateşlemeyi
bırakmış bir iş, bir zamanlayıcının en kötü arızası — o yüzden liste, Gateway'e
ulaşılamamasını **boş liste değil, kendi durumu** olarak raporluyor.

---

## 10 · OpenClaw'dan alınanlar

{svg("f_three_axes", "Üç kontrol ekseni — karıştırmak en yaygın hata")}

"İzin" tek kavram değil, **üç ayrı soru**:

| Eksen | Soru |
|---|---|
| **Sandbox** | Tool **nerede** koşuyor? |
| **Tool policy** | **Hangi** tool çağrılabilir? |
| **Elevated** | Kutunun **dışına çıkış** var mı? |

Kurallar: `deny` her zaman kazanır · `allow` doluysa listede olmayan her şey
bloklu · tool policy sert duraktır.

**Ve OpenClaw'ın kendi belgesindeki uyarı:**

> *"Tool policy tool'u **adına göre** filtreler; `exec` içindeki yan etkileri
> incelemez. `exec` serbestse, `write`/`edit`'i reddetmek shell komutlarını
> salt-okunur yapmaz."*

Yani **"yazma tool'unu kapattık, artık read-only" cümlesi yanlıştır.**

### Taşınacak fikir: rol bir tool listesi değil, **grup adı**

OpenClaw'da 13 tool grubu var (`group:fs`, `group:runtime`, `group:web`…).
KKB'de bu `group:musteri-verisi`, `group:kredi-sorgu`, `group:rapor`,
`group:dis-erisim` olur. Yeni bir tool eklendiğinde **40 rol dosyası
güncellenmiyor**.

### Diğer alınanlar

* **Onay komuta değil, plana bağlanır** — donmuş plan.
* **Dış içerik veri, talimat değil.**
* **Kademeli açığa çıkarma:** prompt'a yalnız bir satırlık tarif giriyor,
  gövde ancak seçilince ödeniyor.

---

## 11 · Denetim

{svg("f_two_ledgers", "İki kayıt hattı — aynı şey değiller")}

**Uyum kaydı** ile **hata ayıklama kaydı** aynı şey değildir:

| | Uyum kaydı | Hata ayıklama kaydı |
|---|---|---|
| Değişmez mi | **evet** | hayır |
| Saklama süresi | var | kısa |
| Sır taşır mı | **asla** | taşıyabilir |
| Kim okur | denetçi | mühendis |

Tek hatla ikisini birden yapmaya çalışmak **ikisini de bozar**: ya denetim
kaydına sır sızar, ya hata ayıklama kaydı gereksiz yere ömür boyu saklanır.

---

## 12 · Çerçeve seçimi

{svg("f_atlas", "Üç ayrı ilişki")}

### Bakım modu bir söylenti değil — ölçüldü

| Paket | Son sürüm | Kaç gün önce |
|---|---|---:|
| **autogen-agentchat** | 0.7.5 | **323** |
| semantic-kernel | 1.44.1 | 13 |
| langgraph | 1.2.11 | 8 |
| agent-framework (MAF) | 1.14.0 | 5 |
| crewai | 1.15.16 | 5 |
| google-adk | 2.7.1 | 2 |
| openai-agents | 0.22.0 | **0** |

Rakiplerin hepsi son iki hafta içinde sürüm çıkardı; AutoGen on bir ay önce.

### Ama MAF'a bugün geçmek de bedava değil

* GA'dan sonra **iki ayda 15 kırıcı değişiklik** — Microsoft'un kendi
  işaretlemesiyle **[kaynak]**
* 36 paketin **8'i** kararlı; harness, FIDES, beceriler hepsi `experimental`
* **Dağıtık runtime yok** — ve LangGraph, CrewAI, Agents SDK, ADK'da da yok

### Kararın dayanağı: motor değiştirilebilir

54 modülün **17'si** AutoGen içe aktarıyor. Kodun **%72,5'i** altında hangi
motorun döndüğünü bilmiyor **[ölçüldü]**. Ekrandaki MAF düğmesi bunun kanıtı.

> **Üç ayrı ilişki:** AutoGen'i **gömüyoruz** (motor, ince arayüz arkasında) ·
> OpenClaw'ı **öğreniyoruz** (karar kuralları, kodu değil) · OpenClaw'ı
> mühendislikte **kullanmaya devam ediyoruz**.

---

## 13 · Bilinen sınırlar

Bu wiki'nin en önemli bölümü. Her sayının ölçüldüğünü söyleyen bir belge,
ölçmediklerini de sayabilmeli.

| Ne | Durum | Neden |
|---|---|---|
| Kod konteynerinin ağ izolasyonu | **bilinen açık** | Yukarı akış parametre sunmuyor. Konteyner izole, ama ağı var. |
| Prompt enjeksiyonu | **izlenmiyor** | Kapı tool adına ve imzasına bakıyor, verinin nereden geldiğine değil. Tarama sonucuna gömülü talimat kapıdan geçer. |
| Zamanlayıcı | **devredilmiş** | Yerli karşılığı yazıldı ve testli, bağlanmadı. |
| MAF kipi | **dar** | Beş API yüzeyi. Kıyas yüzeyi, ikinci boru hattı değil. Tool çağrılan turda cevap metni boş dönüyor. |
| LangGraph / CrewAI davranışı | **[teyitsiz]** | Kuruldular, sembolleri tarandı, **koşturulmadılar**. "Var" demek "çalışıyor" demek değil. |
| Lobster (OpenClaw eklentisi) | **[teyitsiz]** | Resmî eklenti, çekirdekte değil, kurmadık. |

---

<sub>Üretim: `python docs/tools/make_wiki.py` · şemalar `docs/diagrams/figures.py`
(desteyle aynı çizimler) · düzenlenebilir kaynaklar `docs/diagrams/wiki/`</sub>
"""


if __name__ == "__main__":
    import wiki_pages  # noqa: E402 — bölüm metinleri, bu dosyayı şişirmesin

    wiki_pages.bind(svg, figures)

    texts = {}
    for filename, key in WIKIS:
        texts[filename] = atlas() if key == "atlas" else getattr(wiki_pages, key)()

    # Şemalar paylaşılıyor: üç belgede geçen her ad tek dosyaya bağlanıyor.
    names: list[str] = []
    for t in texts.values():
        for n in used_figures(t):
            if n not in names:
                names.append(n)

    SCENES.mkdir(parents=True, exist_ok=True)
    for stale in SCENES.glob("*.excalidraw"):
        if stale.stem not in names:
            stale.unlink()          # artık bağlanmayan sahne kalmasın
    for name in names:
        (SCENES / f"{name}.excalidraw").write_text(
            json.dumps(scene_from_svg(name), ensure_ascii=False), encoding="utf-8")

    for filename, _ in WIKIS:
        text = texts[filename]
        text = anchors(text)
        (ROOT / filename).write_text(text, encoding="utf-8")
        embedded, linked = text.count("<svg"), len(used_figures(text))
        assert embedded == linked, f"{filename}: {embedded} svg, {linked} bağ"
        print(f"{filename:26s} {len(text.splitlines()):>4} satır  "
              f"{embedded:>2} şema  {len(text)/1024:>4.0f} KB")
    print(f"{'diagrams/wiki':26s} {len(names):>4} .excalidraw (hepsi bağlı)")
