#!/usr/bin/env python
"""İki hatırlatma kartı: AutoGen ve OpenClaw, ikişer A4 sayfa.

    python docs/diagrams/make_kart.py
    → docs/pdf/kart-autogen.html   (+ .pdf, 2 sayfa)
    → docs/pdf/kart-openclaw.html  (+ .pdf, 2 sayfa)

**Bu bir deste değil.** Desteye izleyici bakar; bu karta konuşan bakar, ve
konuşurken bakar. Tasarım tamamen o kısıttan çıkıyor:

* **Söylenecek cümle büyük, dayanağı küçük.** Göz önce koyu satırı bulmalı;
  altındaki ince satır ancak biri üstüne gelirse okunacak.
* **Blok başlıkları konuyu değil, *anı* söylüyor** — "EN PAHALI TUZAK", "İLK 90
  SANİYEDE SÖYLE", "KURUMSAL İZLEYİCİNİN SLAYTI". Kartı açtığında aradığın şey
  konu başlığı değil, o an ne söyleyeceğin.
* **Şekil yok.** Şekiller destede. Karta şekil koymak gözün tarama süresini
  uzatır, ve kartın tek işi o süreyi kısaltmak.
* **Sayfa sayısı sabit: iki.** Üçüncü sayfa çevrilmez, o yüzden var olmamalı.
  `tasma_olc.py` bunu ölçüyor — taşarsa içerik sessizce kesilir.

Sayfa geometrisi `make_slides.py`'den ayrı: o 297×167 yatay, bu 210×297 dikey.
Ortak olan tek şey ölçüm disiplini.
"""

from __future__ import annotations

from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent.parent / "pdf"

CSS = """
:root{
  --ink:#16191c; --ink2:#3f474d; --ink3:#767f86;
  --rule:#c9d0d4; --rule2:#e6eaec; --panel:#f4f6f7;
  --accent:#0d5f6b; --accent-bg:#e3eef0;
  --warn:#7a4a10; --warn-bg:#f6efe4;
  --mono:"DejaVu Sans Mono","Liberation Mono",monospace;
  --sans:"DejaVu Sans","Liberation Sans",Arial,sans-serif;
  --serif:"DejaVu Serif","Liberation Serif",Georgia,serif;
  --W:210mm; --H:297mm;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#2a2c28;color:var(--ink)}
body{font-family:var(--sans);line-height:1.4}

.page{
  width:var(--W); height:var(--H); background:#fff; overflow:hidden;
  padding:12mm 13mm 10mm; position:relative;
}

.hd{
  display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:2px solid var(--ink);padding-bottom:1.8mm;margin-bottom:3.4mm;
}
.hd .t{font-family:var(--mono);font-size:14pt;font-weight:700;letter-spacing:-.02em}
.hd .n{font-family:var(--mono);font-size:8.4pt;color:var(--ink3);letter-spacing:.1em;
       text-transform:uppercase}

.b{margin:0 0 3.6mm;padding:0 0 3.4mm;border-bottom:1px solid var(--rule2)}
.b:last-child{border-bottom:0;margin-bottom:0}
.b .lbl{
  font-family:var(--mono);font-size:8pt;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent);margin-bottom:1.2mm;
}
.b.w .lbl{color:var(--warn)}
.b .say{
  font-family:var(--serif);font-size:13.4pt;line-height:1.3;color:var(--ink);margin:0;
}
.b .say + .say{margin-top:1.3mm}
.b .why{font-size:9.8pt;line-height:1.36;color:var(--ink2);margin:1.3mm 0 0}
.b .why + .why{margin-top:1mm}
.b .why b{color:var(--ink)}
code{font-family:var(--mono);font-size:.88em;background:var(--panel);padding:.05em .25em}

.b.hi{background:var(--accent-bg);padding:2.4mm 2.8mm;border-left:2.5px solid var(--accent);
      border-bottom:0}
.b.w{background:var(--warn-bg);padding:2.4mm 2.8mm;border-left:2.5px solid var(--warn);
     border-bottom:0}

.nums{display:flex;flex-wrap:wrap;gap:0 4.6mm;margin:1.4mm 0 0}
.nums div{font-family:var(--mono);font-size:9.4pt;color:var(--ink2)}
.nums b{color:var(--ink);font-size:11pt}

table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:9.6pt;
      margin:1.4mm 0 0}
td{padding:.7mm 0;border-bottom:1px solid var(--rule2);color:var(--ink2)}
td.n{text-align:right;color:var(--ink);font-weight:700;width:16mm}
tr:last-child td{border-bottom:0}

.ft{position:absolute;left:13mm;right:13mm;bottom:5.5mm;
    font-family:var(--mono);font-size:7.4pt;color:var(--ink3);
    border-top:1px solid var(--rule);padding-top:1.1mm;
    display:flex;justify-content:space-between}

@media screen{
  body{padding:16px}
  .page{margin:0 auto 16px;box-shadow:0 2px 14px rgba(0,0,0,.4)}
}
@media print{
  @page{size:210mm 297mm;margin:0}
  html,body{background:#fff}
  .page{page-break-after:always;box-shadow:none;margin:0}
  .page:last-child{page-break-after:auto}
}
"""


def blok(lbl: str, *parts: str, kind: str = "") -> str:
    """Bir blok. `kind` her zaman en sonda — anahtar kelime argümanı."""
    cls = f"b {kind}".strip()
    return f'<div class="{cls}"><div class="lbl">{lbl}</div>' + "".join(parts) + "</div>"


def s(text: str) -> str:
    """Söylenecek cümle — büyük punto."""
    return f'<p class="say">{text}</p>'


def w(text: str) -> str:
    """Dayanak — küçük punto, ancak sorulursa okunur."""
    return f'<p class="why">{text}</p>'


def sayfa(baslik: str, no: str, govde: str, alt_sol: str, alt_sag: str) -> str:
    return (
        f'<div class="page"><div class="hd"><span class="t">{baslik}</span>'
        f'<span class="n">{no}</span></div>{govde}'
        f'<div class="ft"><span>{alt_sol}</span><span>{alt_sag}</span></div></div>'
    )


def kart(title: str, pages: list[str], out: str) -> Path:
    html = (
        '<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{CSS}</style></head><body>\n"
        + "".join(pages) + "\n</body></html>\n"
    )
    path = PDF_DIR / out
    path.write_text(html, encoding="utf-8")
    print(f"  {path.name}  ·  {len(pages)} sayfa  ·  {len(html)/1024:.0f} KB")
    return path


# ══════════════════════════════════════════════════ KART 1 — AutoGen

A1 = "".join([
    blok(
        "aç — omurga cümlesi",
        s("“AutoGen bize bir motor veriyor ama kontrol düzlemi vermiyor.”"),
        w("Üç katman: <b>core</b> aktör modeli · <b>agentchat</b> günlük iş · "
          "<b>ext</b> dış dünya. Kural: yukarıdan başla."),
        kind="hi"),

    blok(
        "ilk 90 saniyede söyle — sonra söylersen gizlemiş olursun",
        s("“AutoGen bakım modunda. Yeni özellik almayacak.”"),
        w("README'nin kendi cümlesi. Son sürüm <code>python-v0.7.5</code>. Halef "
          "<b>microsoft/agent-framework</b>, Nisan 2026'da 1.0 GA. Bizim maruziyet "
          "ölçülü — 2. sayfanın altına bak.")),

    blok(
        "en pahali tuzak — tek bir şey hatirlatacaksan bu",
        s("“Model tool'u çağırıyor, sonuç dönüyor, ve tur orada bitiyor. Model o "
          "sonucu hiç görmüyor.”"),
        w("<code>max_tool_iterations</code> varsayılanı <b>1</b>. Hiçbir hata "
          "çıkmıyor: loga bakarsın, tool çağrılmış, sonuç dönmüş, cevap yine "
          "yanlış. Eksik olan çağrı değil, <b>ikinci model turu</b>.")),

    blok(
        "sessiz bellek kaybi",
        s("“Source'u her istekte değiştirirseniz her istekte yeni bir ajan "
          "doğuyor. Öncekinin belleği silinmiyor — ulaşılamaz hale geliyor.”"),
        w("<code>topic.source → agent.key</code>, dönüşümsüz (core kılavuzu "
          "05:670). Sistem çalışıyor, ajanlar cevap veriyor, hiçbiri bir öncekini "
          "hatırlamıyor.")),

    blok(
        "sessiz dal kaybi",
        s("“Bir dal sessizce ölürse toplayıcı sonsuza kadar bekliyor.”"),
        w("<code>publish_message</code> hata fırlatmıyor, yalnız logluyor. Sıfır "
          "abone de geçerli sonuç. <b>Çözüm tek satır:</b> sayacı "
          "<code>finally</code> bloğunda artır.")),

    blok(
        "maliyet — sayiyi ezberle",
        s("“Aynı görev, aynı ajanlar, aynı model. Yalnız orkestrasyon deseni "
          "değişiyor: 204 tokenden 334'e. Yüzde 63,7.”"),
        '<table>'
        '<tr><td>SelectorGroupChat</td><td class="n">204</td></tr>'
        '<tr><td>GraphFlow</td><td class="n">270</td></tr>'
        '<tr><td>RoundRobinGroupChat</td><td class="n">274</td></tr>'
        '<tr><td>Swarm · handoff</td><td class="n">334</td></tr></table>',
        w("<code>poc/kiyas.py</code> · Ödenen şey zekâ değil <b>yönlendirme "
          "özerkliği</b>. Agents SDK'nın tek modeli olan handoff, AutoGen'in en "
          "pahalı deseni."),
        kind="hi"),

    blok(
        "bunu bir kez yanliş yazdim — eski slayti gören olabilir",
        w("“Turun %63,7'si sistem prompt'udur” diye yazmıştım. <b>Değil.</b> Bu "
          "orkestrasyon deseni farkı, bağlam kompozisyonu değil."),
        kind="w"),
])

A2 = "".join([
    blok(
        "vermediği üç şey — “hiç yok” deme; üçünün de yakini var, üçü de eksik",
        w("<b>Kapı</b> · mesaj katmanında duruyor. Bir mesajı düşürebiliyor ama "
          "“bu ajan şu komutu şu argümanlarla çalıştırmak istiyor”u görmüyor, ve "
          "reddi ajana <b>gerekçesiz</b> dönüyor."),
        w("<b>Onay</b> · <b>var</b>. <code>CodeExecutorAgent</code> içinde, "
          "deneysel, yalnız kod çalıştırmaya özel. Verilmezse sadece uyarı yazıp "
          "çalıştırıyor. Ve onaylı bir ajan <b>yapılandırmaya yazılamıyor</b> — "
          "kod <code>ValueError</code> fırlatıyor."),
        w("<b>Denetim kaydı</b> · Python'ın <code>logging</code>'i. Teslim "
          "garantisi yok, kimlik yok, kurcalama kanıtı yok. Ve "
          "<code>LLMCallEvent</code> modelin <b>bütün mesajlarını</b> içine "
          "koyuyor."),
        kind="hi"),

    blok(
        "perdeyi bağlayan kanca — durakla, sonra söyle",
        s("“Bunu aklınızda tutun. Birazdan OpenClaw'ın denetim kaydını "
          "anlatacağım, ve onun sorunu tam tersi olacak: hiç içerik tutmuyor. "
          "İkisi de bizim istediğimiz şey değil.”")),

    blok(
        "compaction yok",
        s("“Dört bağlam sınıfının dördü de kırpıyor, özetlemiyor.”"),
        w("<code>Unbounded</code> · <code>Buffered</code> · "
          "<code>HeadAndTail</code> · <code>TokenLimited</code>. Özetleyerek "
          "sıkıştıran hazır uygulama <b>yok</b> — onu biz yazdık "
          "(<code>context_engine.py</code>).")),

    blok(
        "iki farkli cache — kariştirilmasi çok yaygin",
        w("<b>ChatCompletionCache</b> · sha256 <b>tam eşleşmeli</b> cevap "
          "önbelleği. Aynı istek birebir tekrarlanırsa cevabı döndürüyor."),
        w("<b>Sağlayıcı prompt cache</b> · <b>önek</b> tabanlı, baştan ilk farklı "
          "bayta kadar. Değişkeni başa koyarsan arkası düşer — tek bir tarih "
          "damgası sistem prompt'unu yakar."),
        w("<code>RequestUsage</code>'da <code>cached_tokens</code> alanı "
          "<b>yok</b>: isabeti AutoGen'in ölçüm nesnesinden okuyamazsın.")),

    blok(
        "sekiz desen — “dokuz” diyen kaynak varsa o tasnif yazarin",
        w("Concurrent Agents · Sequential Workflow · Group Chat · Handoffs · "
          "Mixture of Agents · Multi-Agent Debate · Reflection · Code Execution"),
        w("Yedisi orkestrasyon, sonuncusu bir <b>yetenek</b>. Kılavuzun kendi "
          "bölümlemesi, 05:3206. <b>Ben de bir kez dokuz yazdım, düzelttim</b> — "
          "bunu söylemek geri kalan sayıları güçlendiriyor.")),

    blok(
        "dört sessiz varsayilan — hiçbiri hata vermiyor",
        w("<code>max_tool_iterations=1</code> tool sonucunu modelden saklıyor · "
          "<code>model_context</code> verilmezse ajanın belleği hiç olmuyor · "
          "<code>model_client_stream=False</code> ile token akışı çıkmıyor · "
          "sonlandırma koşulu yoksa takım tavansız koşuyor.")),

    blok(
        "bizde ne var — sorulmadan söyle",
        '<div class="nums">'
        '<div><b>12.892</b> satır</div>'
        '<div><b>4.496</b> autogen’e dokunan</div>'
        '<div><b>8.396</b> dokunmayan</div>'
        '<div><b>381</b> test, hepsi geçiyor</div></div>',
        w("Kontrol düzlemini taşıyan altı gateway modülünde <b>sıfır</b> autogen "
          "importu. Motor değişirse o altı modül yerinde kalır."),
        kind="hi"),
])

# ══════════════════════════════════════════════════ KART 2 — OpenClaw

B1 = "".join([
    blok(
        "aç — omurga cümlesi",
        s("“Ajan runtime'ı kimliği, yetkiyi ve denetimi bilmiyor. Hepsi ajan "
          "döngüsünün dışında — ve tam bu yüzden değiştirilebilir.”"),
        w("Solda kanallar, sağda yetenekler, ortada <b>Gateway</b>. Ajan motorunu "
          "söküp yerine başkasını koyabilirsin; kontrol düzlemi yerinde kalır."),
        kind="hi"),

    blok(
        "üç kontrol ekseni — en yaygin yapilandirma hatasi",
        w("<b>sandbox</b> nerede koşuyor · <b>tool policy</b> hangisi "
          "çağrılabiliyor · <b>elevated</b> kutudan çıkış var mı"),
        s("“‘Write'ı kapattık, artık salt-okunur' cümlesi yanlıştır.”"),
        w("Tool politikası tool'u yalnız <b>adına</b> göre filtreliyor, "
          "<code>exec</code>'in içindeki yan etkiye bakmıyor. Kabuk serbestse "
          "yazmak zaten mümkün.")),

    blok(
        "kurumsal izleyicinin slayti — burada yavaşla",
        s("“Onayladığınız şey bir cümle değil, <b>donmuş bir plan</b>: çalışma "
          "dizini, tam argüman listesi, sabitlenmiş yol.”"),
        w("Onaydan sonra <b>saklanan planı</b> çalıştırıyor, çağıranın sonradan "
          "gönderdiğini değil. Onay bir dosyaya bağlıysa ve dosya değiştiyse "
          "koşuyu <b>reddediyor</b>."),
        kind="hi"),

    blok(
        "diş içerik — prompt injection",
        s("“Sınırın kimliği rastgele. Sabit olsaydı içerik kendi kapanış etiketini "
          "yazıp kutudan çıkardı.”"),
        w("Şüpheli desenler yalnızca <b>loglanıyor</b>, engellenmiyor — çünkü "
          "desen eşleştirmeyle injection engellenemez. Bunu açıkça söylemeleri "
          "güvenilirlik işareti; aksini iddia eden ürünler var.")),

    blok(
        "belleğin güvenlik siniri",
        s("“Savunma ‘kötü belleği sonradan bul' değil, ‘kötü bellek terfi "
          "edemesin'.”"),
        w("Köken sınıfı <b>kapalı bir küme</b> ve SQLite sütununda — modelin "
          "düzyazıyla yazamayacağı yerde. Kökeni belirlenemeyen dışsal içerik "
          "<code>untrusted</code>, <b>asla owner değil</b>.")),

    blok(
        "kademeli açiğa çikarma — %93 tasarruf",
        w("74 skill kurulu ama prompt'a yalnız <b>indeksleri</b> giriyor: ad ve "
          "tek satır açıklama. Gövde talep üzerine çekiliyor. Kazanç yalnız token "
          "değil <b>isabet</b> — model 74 talimatı aynı anda görmüyor.")),

    blok(
        "üç eksen — “iş başarisiz oldu” demenin bedeli",
        s("“Yürütme, teslimat ve nihai sonuç <b>ayrı tipler</b>. Karıştırmak "
          "mümkün değil — tip sistemi hatırlatıyor.”"),
        w("<b>Somut senaryo:</b> alt-ajan işini bitirdi, ama sonucu vereceği oturum "
          "kapanmıştı. Bu iş <code>blocked</code>, <code>failed</code> <b>değil</b>. "
          "<code>failed</code> deseydin biri gelip yapılmış işi ikinci kez "
          "koştururdu; <code>blocked</code> diyorsan çözüm yeniden koşturmak değil, "
          "teslimatı düzeltmek.")),
])

B2 = "".join([
    blok(
        "kkb için en önemli blok — en çok zamani buraya ayir",
        s("“Denetim kaydı içerik tutmuyor. Ve sınırını kendisi söylüyor: "
          "<i>bu korelasyondur, anonimleştirme değildir</i>.”"),
        s("“Kayıt best-effort: kuyruk dolarsa satır düşüyor, koşu devam ediyor.”"),
        s("“Bizim için tersi gerekiyor: kayıpsız, senkron, fail-closed. "
          "Yazılamıyorsa koşu düşmeli.”"),
        w("<b>KİLİT CÜMLE ·</b> Alacağımız şey mekanizma değil, <b>ayrım</b>: "
          "operasyonel hat ile uyum hattı farklı garantiler ister, ve tek bir log "
          "ikisini birden karşılayamaz."),
        kind="hi"),

    blok(
        "durable execution — “dayanikli” kelimesi yanliş anlaşiliyor",
        s("“Dayanıklı <i>durum</i> var, dayanıklı <i>yürütme</i> değil.”"),
        s("“Kurtarma bir replay değil — modele yazılan bir cümle: ‘önceki turun "
          "kesildi, mevcut transkriptten devam et.'”"),
        w("Deterministik replay yok, tamamlanmış adımların memoizasyonu yok. Yan "
          "etkili bir tool çağrıldıktan sonra çökülürse ikinci kez çağrılmasını "
          "<b>mekanik olarak engelleyen hiçbir şey yok</b>; tek koruma modelin "
          "transkripti okuması.")),

    blok(
        "zamanlayici çekirdeği — yalniz sorulursa aç",
        w("Tek kendini yeniden kuran timer, tavan <b>60 sn</b> · her tikte diskten "
          "tazele · vadesi geleni <code>queuedAtMs</code> damgası <b>ve</b> "
          "rezervasyon kimliğiyle rezerve et, <i>sonra</i> koş · "
          "<code>p-map</code> havuzu, eşzamanlılık sabit <b>8</b> (ayar emekli "
          "edilmiş) · backoff 30sn → 60sn → 5dk → 15dk → 60dk")),

    blok(
        "sayilar",
        '<div class="nums">'
        '<div><b>51</b> tool kaynakta</div>'
        '<div><b>44</b> canlı kurulumda</div>'
        '<div><b>22</b> paket</div>'
        '<div><b>5</b> zamanlama türü</div>'
        '<div><b>8</b> kapsam</div>'
        '<div><b>5</b> bellek katmanı</div></div>',
        w("Zamanlama türleri: <code>at</code> · <code>every</code> · "
          "<code>cron</code> · <code>on-exit</code> · <code>stream</code>. Son "
          "ikisi zamana <b>hiç</b> bakmıyor, onlar olay kaynağı.")),

    blok(
        "kapaniş — bu cümleyi ezberle",
        s("“Mekanizmaları al, güven modelini yeniden kur.”"),
        w("OpenClaw <b>tek bir güvenilen operatörün</b> etrafında tasarlanmış. "
          "Belgelerindeki bütün “bu bir güvenlik sınırı değildir” cümleleri buradan "
          "geliyor. Birbirine güvenmeyen departmanların olduğu bir kurumda aynı "
          "cümleler birer <b>açık</b> olur."),
        w("<b>Ölçtüğüm somut örnek:</b> bu makinede exec politikası "
          "<code>mode=full · security=full · ask=off</code>. Ajanın kabuk erişimi "
          "var ve onay sormuyor."),
        kind="hi"),

    blok(
        "“atlas olarak kuralim mi” sorusu gelirse",
        w("Üç ayrı ilişki: AutoGen'i <b>gömüyoruz</b> · OpenClaw'ı "
          "<b>öğreniyoruz</b> · OpenClaw'ı mühendislik takımında <b>araç olarak "
          "kullanıyoruz</b>. Atlas'ın yerine OpenClaw <b>kurmuyoruz</b>."),
        kind="w"),
])


if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print("hatırlatma kartları:")
    kart("AutoGen — hatırlatma kartı", [
        sayfa("AutoGen · hatırlatma kartı", "1 / 2", A1,
              "deste: hap-autogen.pdf · 37 slayt", "19 Ağustos 2026"),
        sayfa("AutoGen · hatırlatma kartı", "2 / 2", A2,
              "ölçüm: poc/kiyas.py · compare_fanin.py", "autogen 0.7.5"),
    ], "kart-autogen.html")
    kart("OpenClaw — hatırlatma kartı", [
        sayfa("OpenClaw · hatırlatma kartı", "1 / 2", B1,
              "deste: hap-openclaw.pdf · 19 slayt · niş 17", "19 Ağustos 2026"),
        sayfa("OpenClaw · hatırlatma kartı", "2 / 2", B2,
              "docs/16 · docs/17 · docs/18", "openclaw @ 01cc7106"),
    ], "kart-openclaw.html")
