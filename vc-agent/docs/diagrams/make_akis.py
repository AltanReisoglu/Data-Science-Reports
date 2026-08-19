#!/usr/bin/env python
"""Sunum akış kartı: `docs/19-sunum-metni.md`'nin kürsüde bakılan hâli.

    python docs/diagrams/make_akis.py
    → docs/pdf/sunum-akis.html  (+ .pdf, 3 sayfa)

### Neden ayrı bir şey

`docs/19` **hazırlanmak** için: 717 satır, gerekçeleriyle. Bu kart **konuşurken**
bakmak için, ve o iki iş aynı belgeyle yapılamaz. Kürsüde kimse paragraf okumaz;
gözün bulması gereken şey sıradaki cümle.

O yüzden bu bir özet değil, bir **sıra çizelgesi**. Zaman soldan sağa akıyor,
her satır bir slayt, ve satırın taşıdığı tek şey söylenecek cümle.

### Neyin kaldığına dair kural

Uzun metinde slayt başına iki-üç cümle var. Buraya **taşıyıcı olan** bir tanesi
giriyor, artı doğaçlanamayacak olan: rakamlar, sınıf adları, satır atıfları.
Gerekçeler girmiyor — onları zaten biliyorsun, bilmediğin şey sıradaki cümle.

İki istisna, çünkü ikisi de tek cümleyle anlatılamıyor ve ikisi de sunumun
omurgası: `[S9]`'un ölçümü ve `[S18]`'in denetim ayrımı.

### Kısa sürüm işareti

Soldaki ▪ "kısa sürümde de kalır" demek. Süre daralınca bakılacak tek şey bu:
işaretsiz satır atlanabilir, işaretli satır atlanamaz.
"""

from __future__ import annotations

from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent.parent / "pdf"

CSS = """
:root{
  --ink:#14181b; --ink2:#3c454b; --ink3:#727c83;
  --rule:#ccd3d7; --rule2:#e8ecee; --panel:#f2f5f6;
  --accent:#0c5f6a; --accent-bg:#e2eef0;
  --warn:#7c4a10; --warn-bg:#f7f0e5;
  --mono:"DejaVu Sans Mono","Liberation Mono",monospace;
  --sans:"DejaVu Sans","Liberation Sans",Arial,sans-serif;
  --serif:"DejaVu Serif","Liberation Serif",Georgia,serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#2a2c28;color:var(--ink)}
body{font-family:var(--sans);line-height:1.3}

.page{width:210mm;height:297mm;background:#fff;overflow:hidden;
      padding:10mm 11mm 8mm;position:relative}

.hd{display:flex;justify-content:space-between;align-items:baseline;
    border-bottom:2px solid var(--ink);padding-bottom:1.4mm;margin-bottom:2.6mm}
.hd .t{font-family:var(--mono);font-size:11.4pt;font-weight:700;letter-spacing:-.02em}
.hd .n{font-family:var(--mono);font-size:7.4pt;color:var(--ink3);
       letter-spacing:.1em;text-transform:uppercase}

/* ── perde başlığı ─────────────────────────────── */
.act{display:flex;justify-content:space-between;align-items:baseline;
     background:var(--ink);color:#fff;padding:1mm 2.2mm;margin:2.1mm 0 1.5mm;
     font-family:var(--mono);font-size:8.4pt;font-weight:700;
     letter-spacing:.08em;text-transform:uppercase}
.act:first-child{margin-top:0}
.act .dk{font-weight:400;opacity:.75;letter-spacing:.04em}

/* ── satır: sol ray + gövde ────────────────────── */
.r{display:grid;grid-template-columns:15mm 1fr;gap:2.4mm;
   padding:.95mm 0;border-bottom:1px solid var(--rule2)}
.r:last-child{border-bottom:0}
.r .k{font-family:var(--mono);font-size:7.6pt;color:var(--ink3);
      padding-top:.4mm;white-space:nowrap}
.r .k b{color:var(--accent);font-weight:700}
.r .say{font-family:var(--serif);font-size:9.6pt;line-height:1.26;margin:0}
.r .why{font-size:7.6pt;line-height:1.28;color:var(--ink2);margin:.7mm 0 0}
.r .why b{color:var(--ink)}
.r.hi{background:var(--accent-bg);border-bottom:0;
      padding:1.6mm 2.2mm;margin:.8mm 0;border-left:2px solid var(--accent)}
.r.w{background:var(--warn-bg);border-bottom:0;
     padding:1.6mm 2.2mm;margin:.8mm 0;border-left:2px solid var(--warn)}
code{font-family:var(--mono);font-size:.9em;background:var(--panel);padding:0 .2em}

/* ── akış merdiveni ────────────────────────────── */
.flow{display:grid;grid-template-columns:5mm 17mm 1fr;gap:.6mm 2mm;
      font-size:7.8pt;line-height:1.26;margin:1.4mm 0 0}
.flow .n{font-family:var(--mono);color:var(--ink3);text-align:right}
.flow .l{font-family:var(--mono);font-size:6.8pt;font-weight:700;
         letter-spacing:.05em;padding-top:.3mm}
.flow .l.g{color:#c2560a} .flow .l.c{color:#5b3ac0} .flow .l.a{color:#1868bd}
.flow .d{color:var(--ink2)}
.flow .d b{color:var(--ink)}

/* ── tablo ─────────────────────────────────────── */
table{border-collapse:collapse;width:100%;font-size:7.8pt;margin:1.2mm 0 0}
th{font-family:var(--mono);font-size:6.8pt;letter-spacing:.06em;
   text-transform:uppercase;color:var(--ink3);text-align:left;
   padding:0 2mm 1mm 0;border-bottom:1px solid var(--rule)}
td{padding:.9mm 2mm .9mm 0;border-bottom:1px solid var(--rule2);
   color:var(--ink2);vertical-align:top}
td b{color:var(--ink)}
tr:last-child td{border-bottom:0}

ul{margin:1mm 0 0;padding-left:4mm;font-size:7.8pt;line-height:1.35;color:var(--ink2)}
li{margin:0 0 .6mm}
li b{color:var(--ink)}

.ft{position:absolute;left:11mm;right:11mm;bottom:4.5mm;
    font-family:var(--mono);font-size:6.4pt;color:var(--ink3);
    border-top:1px solid var(--rule);padding-top:.9mm;
    display:flex;justify-content:space-between}

@media screen{body{padding:16px}.page{margin:0 auto 16px;box-shadow:0 2px 14px rgba(0,0,0,.4)}}
@media print{
  @page{size:210mm 297mm;margin:0}
  html,body{background:#fff}
  .page{page-break-after:always;box-shadow:none;margin:0}
  .page:last-child{page-break-after:auto}
}
"""


def act(title: str, note: str = "") -> str:
    return f'<div class="act"><span>{title}</span><span class="dk">{note}</span></div>'


def r(key: str, say: str, why: str = "", *, kind: str = "", kisa: bool = False) -> str:
    """Bir satır. `kisa=True` → kısa sürümde de kalır (▪ işareti)."""
    mark = "<b>▪</b> " if kisa else "&nbsp;&nbsp; "
    body = f'<p class="say">{say}</p>' if say else ""
    if why:
        body += f'<p class="why">{why}</p>'
    cls = f"r {kind}".strip()
    return f'<div class="{cls}"><div class="k">{mark}{key}</div><div>{body}</div></div>'


def blok(html: str) -> str:
    return f'<div class="r" style="grid-template-columns:1fr"><div>{html}</div></div>'


def page(title: str, no: str, body: str, fl: str, fr: str) -> str:
    return (f'<div class="page"><div class="hd"><span class="t">{title}</span>'
            f'<span class="n">{no}</span></div>{body}'
            f'<div class="ft"><span>{fl}</span><span>{fr}</span></div></div>')


# ═══════════════════════════════════════════ SAYFA 1

P1 = "".join([
    act("Omurga — sunumun kanıtladığı tek cümle"),
    blok('<p class="say" style="font-size:9.8pt">AutoGen bize bir <b>motor</b> '
         'veriyor ama <b>kontrol düzlemi</b> vermiyor. OpenClaw kontrol düzlemini '
         'çözmüş ama <b>güven modeli</b> bizim kurumumuz için yanlış. Atlas'
         '&#39;ın alması gereken: AutoGen&#39;in motoru, OpenClaw&#39;ın karar '
         'kuralları, bizim güven modelimiz.</p>'
         '<p class="why">Sol sütundaki <b>▪</b> = kısa sürümde de kalır. '
         'İşaretsiz satır atlanabilir. Süre daralırsa slayt atla, cümle kısaltma.</p>'),

    act("Açılış", "2 dk · slayt yok"),
    r("", "“Üç aydır iki sistemi inceliyorum. Bugün size ne öğrendiğimi değil, "
          "<b>neyi ölçtüğümü</b> anlatacağım.”", kisa=True),
    r("", "“Baştan söyleyeyim: AutoGen <b>bakım modunda</b>, OpenClaw&#39;ı da olduğu "
          "gibi kurmamızı önermiyorum. Buna rağmen ikisini de anlatıyorum — ikisi de "
          "bizim problemlerimizi bizden önce çözmüş ve <b>nerede durduklarını "
          "yazmışlar</b>.”",
      "Kapanış: “Çıktı bir ürün önerisi değil, bir <b>değerlendirme çerçevesi</b>.” "
      "· 90 saniyede bitir, ayrıntı verme.", kisa=True),

    act("Perde 1 — Motor: AutoGen ne veriyor", "10 dk · hap-autogen.pdf"),
    r("S2", "“Üç katman: <b>core</b> aktör modeli, <b>agentchat</b> günlük iş, "
            "<b>ext</b> dış dünya. Kural: yukarıdan başla.”", kisa=True),
    r("S3", "“Ajan ajanı çağırmıyor, runtime&#39;a mesaj veriyor.”",
      "Bedel: “kim kimi çağırdı” yığın izinde yok. Karşılık: <b>bütün mesajlar tek "
      "noktadan geçiyor</b> — kapıyı kurabilmemizin sebebi."),
    r("S6", "“Source&#39;u her istekte değiştirirseniz her istekte <b>yeni bir ajan</b> "
            "doğuyor. Öncekinin belleği silinmiyor — ulaşılamaz oluyor.”",
      "<code>topic.source → agent.key</code> (05:670). Hata çıkmıyor: sistem "
      "çalışıyor, hiçbiri bir öncekini hatırlamıyor."),
    r("S7", "“Bir dal sessizce ölürse toplayıcı <b>sonsuza kadar bekliyor</b>.”",
      "<code>publish_message</code> hata fırlatmıyor, logluyor. Çözüm tek satır: "
      "sayacı <code>finally</code>&#39;de artır."),
    r("S9", "“Tool koşuyor, sonuç dönüyor, tur orada bitiyor. Cevabı model yazmıyor "
            "— kullanıcıya <b>ham tool çıktısı</b> gidiyor.”",
      "<b>ÖLÇÜM:</b> iki adımlı iş (id bul → detay çek), soru “çalışan sayısı kaç”, "
      "dönen cevap <code>{\"id\":\"KA-9931\"}</code>. İkinci tool <b>hiç çağrılmadı</b>, "
      "log&#39;da hata yok. <b>Sessiz olan ham çıktı değil, duran zincir.</b><br>"
      "İki ayrı anahtar: <code>max_tool_iterations</code>=1 zincirlemeyi · "
      "<code>reflect_on_tool_use</code>=False modelin cevabı yazmasını kapatıyor.",
      kind="hi", kisa=True),
    r("S10<br>S15", "“Aynı görev, aynı ajan, aynı model. Yalnız orkestrasyon deseni "
                    "değişiyor: <b>204 → 334 token. %63,7.</b>”",
      "Selector 204 · GraphFlow 270 · RoundRobin 274 · Swarm 334 "
      "(<code>poc/kiyas.py</code>). Ödenen zekâ değil <b>yönlendirme özerkliği</b>.<br>"
      "Cache: önbellek <b>önekten</b> çalışır, ilk farklı bayta kadar — başa "
      "değişken koyarsan arkası düşer."),
    r("S14", "“Resmî kılavuzda <b>sekiz</b> desen var.”",
      "“Dokuz desen” diyen kaynak varsa o tasnif yazarın. <b>Ben de bir kez dokuz "
      "yazdım, düzelttim.</b>"),
    r("S17", "“Dört sessiz varsayılan, dördü de sistemi çalıştırıp sonucu bozuyor.”",
      "<code>max_tool_iterations=1</code> · <code>model_context</code> yoksa bellek "
      "yok · <code>model_client_stream=False</code> · sonlandırma yoksa tavansız."),
    r("S18", "“Vermediği <b>kontrol düzlemi</b> — ve üçünün de yakını var, üçü de "
             "eksik.”",
      "<b>Kapı</b>: mesaj katmanında, tool çağrısını görmüyor, reddi gerekçesiz. · "
      "<b>Onay</b>: VAR ama <code>CodeExecutorAgent</code> içinde, deneysel, koda "
      "özel; onaylı ajan <b>yapılandırmaya yazılamıyor</b>. · <b>Denetim</b>: Python "
      "logging, teslim garantisi yok, modelin <b>bütün mesajlarını</b> içine koyuyor."
      "<br><b>KANCA →</b> “Bunu aklınızda tutun; OpenClaw&#39;ın sorunu tam tersi "
      "olacak: hiç içerik tutmuyor.”", kind="hi", kisa=True),

    # Rakamlar tek yerde: sunum sırasında bir sayı sorulduğunda sayfa
    # çevirmeden bulunmalı. Hepsi ölçüldü ya da birincil kaynaktan.
    act("Rakamlar — hepsi tek yerde", "sorulunca sayfa çevirme"),
    blok('<table>'
         '<tr><th>ne</th><th>sayı</th><th>nereden</th></tr>'
         '<tr><td>Desen maliyet farkı</td><td><b>%63,7</b> · 204→334 token</td>'
         '<td>ölçüldü · <code>poc/kiyas.py</code></td></tr>'
         '<tr><td>Resmî desen · AutoGen sürümü</td>'
         '<td><b>8</b> desen · <b>0.7.5</b>, bakım modu</td>'
         '<td>kaynak · <code>05:3206</code> · halef MAF</td></tr>'
         '<tr><td>OpenClaw: tool · paket · zamanlama · kapsam</td>'
         '<td><b>51</b>/44 · <b>22</b> · <b>5</b> · <b>8</b></td>'
         '<td>ölçüldü + kaynak · <code>@01cc7106</code></td></tr>'
         '<tr><td>Cron eşzamanlılık · timer tavanı</td>'
         '<td><b>8</b> sabit · <b>60 sn</b></td><td>kaynak · ayar emekli</td></tr>'
         '<tr><td>Bizim boru hattı</td>'
         '<td><b>12.892</b> satır · <b>4.496</b>/<b>8.396</b></td>'
         '<td>ölçüldü · autogen&#39;e dokunan/dokunmayan</td></tr>'
         '<tr><td>Test · lisans</td><td><b>395</b> geçiyor · ikisi de <b>MIT</b></td>'
         '<td>ölçüldü + kaynak</td></tr>'
         '<tr><td>Yerel model donanım tabanı</td><td>~<b>$30k</b></td>'
         '<td><b>teyitsiz</b> — piyasa fiyatı, ölçüm değil</td></tr>'
         '</table>'),
])

# ═══════════════════════════════════════════ SAYFA 2

P2 = "".join([
    act("Perde 2 — Kuşatma: OpenClaw nasıl çözmüş", "12 dk · hap-openclaw.pdf"),
    r("S2", "“Ajan runtime&#39;ı kimliği, yetkiyi ve denetimi <b>bilmiyor</b>. Hepsi "
            "ajan döngüsünün dışında — ve tam bu yüzden değiştirilebilir.”", kisa=True),
    r("S3", "“‘Write&#39;ı kapattık, artık salt-okunur&#39; cümlesi <b>yanlıştır</b>.”",
      "Tool politikası tool&#39;u yalnız <b>adına</b> göre filtreliyor, "
      "<code>exec</code>&#39;in içindeki yan etkiye bakmıyor. Üç eksen: sandbox / tool "
      "policy / elevated."),
    r("S5", "“Onayladığınız şey bir cümle değil, <b>donmuş bir plan</b>: çalışma "
            "dizini, tam argüman listesi, sabitlenmiş yol.”",
      "Onaydan sonra <b>saklanan planı</b> çalıştırıyor. Dosya değiştiyse koşuyu "
      "<b>reddediyor</b>. · Burada yavaşla.", kisa=True),
    r("S6", "“Sınırın kimliği <b>rastgele</b>. Sabit olsaydı içerik kendi kapanış "
            "etiketini yazıp kutudan çıkardı.”",
      "Şüpheli desenler yalnız <b>loglanıyor</b>, engellenmiyor — desen "
      "eşleştirmeyle injection engellenemez. Bunu söylemeleri güvenilirlik işareti."),
    r("S10", "“Savunma ‘kötü belleği sonradan bul&#39; değil, ‘<b>kötü bellek terfi "
             "edemesin</b>&#39;.”",
      "Köken sınıfı kapalı küme + SQLite sütununda — model düzyazıyla yazamıyor. "
      "Belirlenemeyen dışsal içerik <code>untrusted</code>, <b>asla owner değil</b>."),
    r("S12", "“Beş tetikleyici türü, ve son ikisi zamana <b>hiç</b> bakmıyor.”",
      "<code>at</code>·<code>every</code>·<code>cron</code>·<code>on-exit</code>·"
      "<code>stream</code>. İki zamanlayıcı: Automations izole+kayıtlı, Heartbeat "
      "bağlamlı+kayıtsız."),
    r("S16", "“Dayanıklı <b>durum</b> var, dayanıklı <b>yürütme</b> değil. Kurtarma "
             "bir replay değil — modele yazılan bir cümle.”",
      "“Önceki turun kesildi, transkriptten devam et.” Deterministik replay yok, "
      "memoizasyon yok. <b>Yan etkili tool ikinci kez çağrılabilir</b>; tek koruma "
      "modelin transkripti okuması."),
    r("S18", "“Denetim kaydı içerik tutmuyor. Ve sınırını kendisi söylüyor: "
             "<i>bu korelasyondur, anonimleştirme değildir</i>.”",
      "“Kayıt <b>best-effort</b>: kuyruk dolarsa satır düşer, koşu devam eder.” → "
      "“<b>Bizim için tersi gerekiyor: kayıpsız, senkron, fail-closed. Yazılamıyorsa "
      "koşu düşmeli.</b>”<br><b>KİLİT:</b> alacağımız mekanizma değil <b>ayrım</b> — "
      "operasyonel hat ile uyum hattı farklı garantiler ister, tek log ikisini "
      "karşılayamaz. · <b>En çok zamanı buraya ayır.</b>", kind="hi", kisa=True),
    r("S19", "“<b>Mekanizmaları al, güven modelini yeniden kur.</b>”",
      "OpenClaw tek bir <b>güvenilen operatörün</b> etrafında tasarlanmış — "
      "“bu bir güvenlik sınırı değildir” cümleleri oradan. Ölçüm: bu makinede "
      "<code>exec: mode=full · ask=off</code>.", kisa=True),

    act("Perde 3 — Bizde ne var", "5 dk"),
    r("", "“Bunlar okuduğum değil, <b>ölçtüğüm</b> sistemler. Ölçmek için bir boru "
          "hattı yazdım ve bugün çalışıyor.”",
      "<b>12.892</b> satır · <b>4.496</b> autogen&#39;e dokunan · <b>8.396</b> "
      "dokunmayan · <b>395</b> test geçiyor. Kontrol düzlemini taşıyan altı gateway "
      "modülünde <b>sıfır</b> autogen importu.", kisa=True),
    r("", "<b>Üç dürüst sınır — sen söyle, biri bulmasın:</b>",
      "① “İnce arayüz” tam tutmuyor: autogen <b>15 modüle</b> sızmış; değiştirilebilir "
      "olan gateway, motor değil. · ② Kendi zamanlayıcımız <b>yazılı ama bağlı değil</b> "
      "(<code>gateway/cron.py</code>, 322 satır) — kontrol düzlemi kararıyla tutarsız. · "
      "③ LangGraph/CrewAI karşılaştırmaları <b>koşturulmadı</b>.", kind="w", kisa=True),

    act("Kapanış ve istenen karar", "3 dk · slayt yok · EZBERLE"),
    r("", "“AutoGen&#39;i <b>gömüyoruz</b>. OpenClaw&#39;ı <b>öğreniyoruz</b> — karar "
          "kurallarını, kodunu değil. Ve mühendislik takımında <b>araç olarak "
          "kullanmaya</b> devam ediyoruz. Atlas olarak OpenClaw <b>kurmuyoruz</b>.”",
      kisa=True),
    r("", "“İstediğim bugün bir ürün kararı değil. <b>Faz 1 için onay</b>: onay kapısı, "
          "uyum kayıt hattı, tek dar kullanım. <b>30 gün, tek kişi.</b> Faz 1 bitince "
          "elimizde ölçülmüş bir şey olacak; kalan iki fazı o zaman konuşuruz — şimdi "
          "konuşursak tahmin etmiş oluruz.”",
      "<b>Kapanış hiçbir koşulda kısaltılmaz.</b> Karar söylenmezse sunum "
      "bilgilendirmeye döner.", kind="hi", kisa=True),
])

# ═══════════════════════════════════════════ SAYFA 3

P3 = "".join([
    act("Demo — üç perde", "3 dk · localhost:8000"),
    r("①", "<b>İzleme şeridi · 30 sn</b> — sıradan bir soru sor.",
      "<code>context → model → stream → done</code>, ~1 sn. “Her satırda gerçek "
      "sınıf adı, gerçek dosya, kılavuz satırı var.”"),
    r("②", "<b>Kapı ve ikinci tur · 60 sn</b> — asıl perde.",
      "Soruyu <b>ölçülmüş olanlardan seç</b>, yoksa model tool&#39;a hiç gitmez ve "
      "şerit dört satırda biter:<br><code>search_docs ile durable execution "
      "konusunda ne dediğimizi bul</code><br><code>scan_facts ile son taramanın "
      "özetini ver</code>"),
    r("③", "<b>Onay kapısı · 90 sn</b> — KKB perdesi.",
      "<code>/openclaw schedule her gün 05:00 | bana merhaba de</code> → kart çıkar, "
      "iş <b>kurulmaz</b>. Onayla, tekrar gönder, iş kurulur (<code>0 5 * * *</code>), "
      "listeyi göster, sil.<br>“Onay <b>tüketildi</b>. Aynı satırı tekrar gönderirsem "
      "yeniden sorar — onaylanan ‘bu tool&#39; değil, ‘bu tool bu argümanlarla, bir "
      "kez&#39;.”"),

    act("Akışı canlı anlatmak", "10 aşama · 2 dk · şerit kaydırmadan sığar"),
    blok('<div class="flow">'
         '<div class="n">1</div><div class="l g">GATEWAY</div>'
         '<div class="d"><b>Bağlam kuruluyor</b> — “Modele ne gideceğine karar '
         'veriliyor. AutoGen&#39;in sınıfı mesaj sayar, bu <b>token</b> sayan hâlimiz.” '
         '<i>16 tool · 12000 bütçe · üç workbench: biri yerel, ikisi MCP</i></div>'
         '<div class="n">2</div><div class="l c">CORE</div>'
         '<div class="d"><b>Model çağrısı</b> — “<code>create</code> değil '
         '<code>create_stream</code>: bu yüzden <code>LLMCallEvent</code> değil '
         '<code>LLMStreamEndEvent</code> yayılıyor. Yalnız ilkini dinleyen ölçüm '
         '<b>sıfır</b> görür.”</div>'
         '<div class="n">3</div><div class="l a">AGENTCHAT</div>'
         '<div class="d"><b>Model bir tool istedi</b> — “Seçmesinin tek sebebi '
         '<b>docstring</b>. Şema imzadan ve docstring&#39;den üretiliyor; yani '
         'docstring dokümantasyon değil <b>arayüz</b>.”</div>'
         '<div class="n">4</div><div class="l g">GATEWAY</div>'
         '<div class="d"><b>KAPI — burada dur, sunumun tezi</b> — “Bu satır bizim; '
         'AutoGen&#39;de karşılığı yok. Her tool çağrısı buradan geçiyor. '
         '<i>blocked:false · hooks:2</i> — biri engelleseydi ajana <b>gerekçesiyle</b> '
         'bir ret dönerdi, istisna değil. Kapı ajanın uyum göstermeyi seçmesine değil, '
         '<b>hattın kendisine</b> dayanıyor.”</div>'
         '<div class="n">5</div><div class="l c">CORE</div>'
         '<div class="d"><b>Tool koşuyor</b> — “<code>StaticWorkbench</code>, yani '
         'yerel fonksiyon. Uzak MCP olsaydı <code>McpWorkbench</code> yazardı, '
         '<b>gerisi aynı</b>.”</div>'
         '<div class="n">6</div><div class="l a">AGENTCHAT</div>'
         '<div class="d"><b>Sonuç döndü</b> — bağlama girdi, döngü modele dönüyor.</div>'
         '<div class="n">7</div><div class="l a">AGENTCHAT</div>'
         '<div class="d"><b>Döngü devam ediyor</b> — “<b>[S9]&#39;un ekrandaki '
         'kanıtı.</b> Altı yazıyor çünkü <b>biz yükselttik</b>. Varsayılan bir olsaydı '
         'bu satır hiç olmayacaktı: zincir burada dururdu ve kullanıcıya ham tool '
         'çıktısı giderdi.”</div>'
         '<div class="n">8</div><div class="l c">CORE</div>'
         '<div class="d"><b>Model çağrısı</b> — “Model şimdi tool&#39;un bulduğunu '
         '<b>görüyor</b> ve cevabı ona göre yazıyor.”</div>'
         '<div class="n">9</div><div class="l a">AGENTCHAT</div>'
         '<div class="d"><b>Token akışı</b> — “<code>model_client_stream</code> kapalı '
         'olsaydı cevap tek parça, model bitirdikten sonra düşerdi.”</div>'
         '<div class="n">10</div><div class="l a">AGENTCHAT</div>'
         '<div class="d"><b>Tur bitti</b> — <i>2 llm · 1 tool · ~20.000 token</i>. '
         '“Bu sayı ekranda çünkü <b>biz sayıyoruz</b> — ve %63,7&#39;lik desen farkı '
         'tam olarak bu sayacın ürettiği ölçüm.”</div>'
         '</div>'),
    r("", "<b>Kapanış — ekranı göstererek:</b> “On satır. İkisi GATEWAY, üçü CORE, "
          "beşi AGENTCHAT. Bütün sunum boyunca anlattığım ayrım burada duruyor: "
          "<b>motor onların, kuşatma bizim.</b> Ve dördüncü satır — kapı — "
          "AutoGen&#39;de olmayan tek şey.”", kind="hi", kisa=True),

    act("Demo çökerse — üçünün de dürüst cümlesi var"),
    blok('<table><tr><th>ne çökerse</th><th>ne söyle</th></tr>'
         '<tr><td><b>Canlı model yok</b></td><td>“Kuru modda koşuyorum: cevaplar '
         'önceden yazılmış. Ama <b>kontrol akışı gerçek</b>, ve göstermek istediğim '
         'o.”</td></tr>'
         '<tr><td><b>OpenClaw kapalı</b></td><td>③&#39;ü atla. “Zamanlama '
         'OpenClaw&#39;a devredilmiş ve Gateway ayakta değil. <b>Bu tam da az önce '
         'söylediğim risk:</b> <code>Linger=no</code>.”</td></tr>'
         '<tr><td><b>Panel açılmıyor</b></td><td>Üçünü de atla, ekran görüntülerine '
         'geç. <b>Sunumu demoya bağlama.</b></td></tr></table>'),

    act("Süre daralırsa — sırayla feda et"),
    blok('<ul>'
         '<li><b>1.</b> Perde 1&#39;in ortası: S3·S6·S7·S14·S17. '
         '<b>S9 ve S18 kalır</b> — biri en pahalı tuzağı, diğeri perde dönüşünü taşır.</li>'
         '<li><b>2.</b> Perde 2&#39;nin ortası: S3·S6·S10·S12. '
         '<b>S5·S18·S19 asla atılmaz</b> — onay, denetim, güven modeli.</li>'
         '<li><b>3.</b> Perde 3&#39;ü üç sayıya indir: 12.892 satır · altı modülde '
         'sıfır autogen · 395 test geçiyor.</li>'
         '<li><b>Kapanış hiçbir koşulda kısaltılmaz.</b></li></ul>'),

    act("Sunum öncesi kontrol", "10 dk kala"),
    blok('<ul>'
         '<li>☐ Üç PDF açık ve sırada · soru hazırlığı sayfası telefonda</li>'
         '<li>☐ <code>curl localhost:8000/api/health</code> → <code>ok</code> ve '
         '<code>live_llm</code> true</li>'
         '<li>☐ Bir soru sor: şeritte <b>on satır</b> çıkıyor mu? Dört çıkıyorsa '
         'model tool&#39;a gitmemiştir — ölçülmüş sorulardan seç</li>'
         '<li>☐ <code>openclaw gateway status</code> ayakta · '
         '<code>/openclaw schedule</code> listeliyor</li>'
         '<li>☐ <b>Test artıklarını sil</b> — önceki denemelerden kalan işler '
         'OpenClaw&#39;ın deposunda kalıcı</li>'
         '<li>☐ Kapanıştaki “istediğim şey” cümlesi ezberde</li></ul>'),
])


if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    html = (
        '<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        "<title>Sunum akış kartı</title>\n"
        f"<style>{CSS}</style></head><body>\n"
        + page("Sunum akışı · AutoGen → OpenClaw → Atlas", "1 / 3", P1,
               "tam metin: docs/19-sunum-metni.md", "19 Ağustos 2026")
        + page("Sunum akışı · Perde 2 ve kapanış", "2 / 3", P2,
               "desteler: hap-openclaw.pdf · niş 17", "19 Ağustos 2026")
        + page("Sunum akışı · demo ve acil durum", "3 / 3", P3,
               "localhost:8000 · 395 test geçiyor", "19 Ağustos 2026")
        + "\n</body></html>\n"
    )
    out = PDF_DIR / "sunum-akis.html"
    out.write_text(html, encoding="utf-8")
    print(f"  {out.name}  ·  3 sayfa  ·  {len(html)/1024:.0f} KB")
