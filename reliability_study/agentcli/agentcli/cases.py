"""
Önceden tanımlı case'ler.

Her case bir SAYFA + bir GÖREV + bir BEKLENTİ. Sayfalar `data:` URL'si — ağ
gerekmiyor, deterministik, ve bozukluğu ben tasarlıyorum.

İki grup:
  YAKALAMA  guardrail devreye girmeli
  KONTROL   guardrail SUSMALI — meşru koşum, kesilmemeli

İkinci grup birincisi kadar önemli: yalnız yakaladıklarını gösteren bir demo,
dedektörün yanlış pozitif oranı hakkında hiçbir şey söylemez.
"""

from __future__ import annotations

from dataclasses import dataclass

_STIL = ("<style>body{font:15px system-ui;padding:24px;max-width:520px}"
         "input,button{font:14px system-ui;padding:6px 10px;margin:4px 0;display:block}"
         "button{cursor:pointer}#log{color:#666;margin-top:12px}</style>")


def _sayfa(govde: str) -> str:
    return "data:text/html," + (_STIL + govde).replace("\n", "").replace("#", "%23")


@dataclass
class Case:
    ad: str
    grup: str            # "yakalama" | "kontrol"
    gorev: str
    url: str
    anlat: str
    bekleniyor: str
    kategori: str = "-"  # hangi ZIHNIYET KATEGORISINI sinaviyor
    kok: str | None = None   # bu case ozel bir calisma dizini istiyorsa
    masaustu: str | None = None   # sahte masaustu senaryosu (bkz. sahte.py)


CASES: dict[str, Case] = {}


def _ekle(c: Case) -> None:
    CASES[c.ad] = c


# ------------------------------------------------------------- YAKALAMA

_ekle(Case(
    ad="olu-buton", grup="yakalama", kategori="window",
    gorev="Formu doldur ve Gonder butonuna basarak gonder.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick='void 0'>Gonder</button><div id=log></div>"),
    anlat="Buton tıklanıyor, HATA VERMİYOR, hiçbir şey de olmuyor. "
          "En sinsi bozukluk: ajan başarısız olduğunu anlamıyor.",
    bekleniyor="Tekrar/ilerleme dedektörleri yakalar. Bütçe zihniyetleri limitte durur."))

_ekle(Case(
    ad="bozuk-arac", grup="yakalama", kategori="evidence",
    gorev="Gonder butonuna basarak formu gonder.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick=\"document.getElementById('log').innerText="
               "'HATA: servis yanit vermedi ('+(++window.n||(window.n=1))+')'\">Gonder</button>"
               "<div id=log></div>"),
    anlat="Araç KALICI bozuk — her tıklama hata döndürüyor. "
          "Suç ajanda değil, araç katmanında.",
    bekleniyor="galileo-breaker devre kesiciyi açar. Diğerleri tekrar deseni olarak yakalar."))

_ekle(Case(
    ad="sonsuz-onay", grup="yakalama", kategori="window",
    gorev="Formu gonder ve islemi tamamla.",
    url=_sayfa("<h2>Onay</h2><div id=log>Devam etmek icin onayla</div>"
               "<button onclick=\"document.getElementById('log').innerText="
               "'Lutfen bir kez daha onaylayin'\">Onayla</button>"
               "<button onclick=\"document.getElementById('log').innerText="
               "'Lutfen bir kez daha onaylayin'\">Devam</button>"),
    anlat="İki düğme arasında sonsuz onay döngüsü — A-B-A-B deseni. "
          "Ardışık tekrar sayan dedektörlerin kör noktası.",
    bekleniyor="strands-entropy tek kuralla, openhands-stuck çevrim taramasıyla yakalar."))

_ekle(Case(
    ad="yalanci-bitis", grup="yakalama", kategori="evidence",
    gorev="Formu doldur ve gonder.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick='void 0'>Gonder</button>"
               "<div id=log>durum: gonderilmedi</div>"),
    anlat="Ajan işi yapmadan 'bitirdim' diyebilir. Sayfa 'gonderilmedi' diyor "
          "ama model bunu görmezden gelirse koşum OK biter.",
    bekleniyor="verify-gate ve telemetry-repair iddiayı reddeder. Diğerleri OK sayar."))

_ekle(Case(
    ad="sonsuz-liste", grup="yakalama", kategori="budget",
    gorev="Listedeki BUTUN kayitlari gez ve toplam kayit sayisini soyle.",
    url=_sayfa("<h2>Kayitlar</h2><div id=log>sayfa 1 · kayit 1-10</div>"
               "<button onclick=\"window.s=(window.s||1)+1;"
               "document.getElementById('log').innerText="
               "'sayfa '+window.s+' · kayit '+(window.s*10-9)+'-'+(window.s*10)\">"
               "Sonraki sayfa</button>"),
    anlat="Sayfa ASLA bitmiyor: her tiklama yeni bir sayfa. Ne tekrar var "
          "(icerik her adimda degisiyor) ne hata var. Ilerleme dedektorleri "
          "kor: gercekten ilerliyor. Duran tek sey BUTCE.",
    bekleniyor="SAYAC ailesi limitte durur. PENCERE ailesi sessiz kalir — "
               "bu ailenin kor noktasinin en temiz gosterimi."))

_ekle(Case(
    ad="zorunlu-sira", grup="yakalama", kategori="shape",
    gorev="Formu gonder.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick=\"window.d=1;document.getElementById('log')"
               ".innerText='dogrulandi — simdi gonderebilirsiniz'\">Dogrula</button>"
               "<button onclick=\"document.getElementById('log').innerText="
               "window.d ? 'gonderildi' : 'once Dogrula'\">Gonder</button>"
               "<div id=log>durum: hazir</div>"),
    anlat="Islem bir SIRA dayatiyor: once Dogrula, sonra Gonder. Sirasi "
          "bozulunca sayfa sessizce reddediyor. Ajan Gonder'e basip duruyor.",
    bekleniyor="SEKIL ailesi (modexa) izinsiz gecisi gorur. autogen-static "
               "GOREMEZ — o yalniz statik graf dogruluyor, bu kor noktasi."))

_ekle(Case(
    ad="yanlis-yer", grup="yakalama", kategori="evidence",
    gorev="Masaustunde 'rapor' adinda bir klasor olustur.",
    url=_sayfa("<h2>Terminal gorevi</h2><div id=log>Bu gorev tarayici gerektirmiyor</div>"),
    anlat="Gorev masaustunu soyluyor ama kisitli kabuk baska bir dizine kilitli. "
          "Ajan `mkdir rapor` calistirip 'masaustunde olusturuldu' diyor. "
          "Is YAPILDI ama YANLIS YERDE — olculdu, gercek kosumda tam bu oldu.",
    bekleniyor="DUNYA ailesi (verify-gate/telemetry-repair) konum uyusmazligini "
               "yakalar. Digerleri OK sayar: hicbir tekrar, hicbir hata yok."))

_ekle(Case(
    ad="erken-cevap", grup="yakalama", kategori="decision",
    gorev="Sayfadaki UC bolumun (A, B, C) her birindeki sayiyi oku ve "
          "ucunun TOPLAMINI soyle.",
    url=_sayfa("<h2>Rapor</h2><div>Bolum A: <b>12</b></div>"
               "<button onclick=\"document.getElementById('b').innerText="
               "'Bolum B: 7'\">Bolum B'yi ac</button><div id=b>Bolum B: gizli</div>"
               "<button onclick=\"document.getElementById('c').innerText="
               "'Bolum C: 23'\">Bolum C'yi ac</button><div id=c>Bolum C: gizli</div>"),
    anlat="Ilk sayi hemen gorunuyor, digerleri tiklama istiyor. Model erken "
          "cevap verme egiliminde: 12'yi gorup 'toplam 12' diyebiliyor. "
          "Ne dongu var ne hata — durum makinesi de tekrar dedektoru de sessiz.",
    bekleniyor="KARAR ailesinden voi-allocation erken cevabi engeller "
               "(early_answer_blocked). improvement-loop MUDAHALE ETMEZ — "
               "yalniz olcer; o kategorinin yarisinin kor noktasi bu."))

_ekle(Case(
    ad="olcek-kaymasi", grup="yakalama", kategori="window", masaustu="bozuk",
    gorev="Masaustundeki 'Metin Duzenleyici' simgesine tiklayarak uygulamayi ac.",
    url=_sayfa("<h2>Sahte masaustu</h2><div id=log>Bu senaryo tarayici kullanmiyor"
               " — masaustu araclarini kullan</div>"),
    anlat="HARNESS HATASI, model hatasi degil. Model 1280x720 karede dogru yere "
          "nisan aliyor; koordinat 1920x1080 ekrana OLCEKLENMEDEN basiliyor ve "
          "her tik 1.5 kat yukari-sola, bosluga dusuyor. Hata yok, cokme yok, "
          "ekran degismiyor. 25 Agustos'ta gercek kosumda tam bu oldu: 14 adim, "
          "33.338 token, hicbir sey acilmadi.",
    bekleniyor="PENCERE ailesi doengueyue yakalar ve HEPSI dogru soyler: 'ayni "
               "cagriyi tekrar ediyorsun'. HICBIRI 'koordinatlarin yanlis uzayda' "
               "diyemez. Dedektor doenguyue goeruer, SEBEBINI goeremez."))

_ekle(Case(
    ad="olcek-duzeltilmis", grup="kontrol", kategori="-", masaustu="duzgun",
    gorev="Masaustundeki 'Metin Duzenleyici' simgesine tiklayarak uygulamayi ac.",
    url=_sayfa("<h2>Sahte masaustu</h2><div id=log>Bu senaryo tarayici kullanmiyor"
               " — masaustu araclarini kullan</div>"),
    anlat="AYNI senaryo, AYNI model, AYNI gorev — tek fark: koordinat olcegi "
          "uygulaniyor. Yan yana kosturulunca doengueneuen kaynaginin harness "
          "oldugu tek degiskenle kanitlaniyor.",
    bekleniyor="Iki-uec adimda OK biter. Hicbir guardrail konusmamali — bu ayni "
               "zamanda yanlis pozitif kontrolu."))

# -------------------------------------------------------------- KONTROL

_ekle(Case(
    ad="saglikli", grup="kontrol", kategori="-",
    gorev="Ad alanina 'Altan' yaz ve Gonder butonuna bas.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick=\"document.getElementById('log').innerText="
               "'gonderildi: '+document.getElementById('ad').value\">Gonder</button>"
               "<div id=log>durum: hazir</div>"),
    anlat="Hiçbir şey bozuk değil. Guardrail SUSMALI.",
    bekleniyor="Bütün zihniyetler OK dönmeli, kontrolsüz koşumla aynı adımda."))

_ekle(Case(
    ad="mesru-retry", grup="kontrol", kategori="-",
    gorev="Gonder butonuna basarak formu gonder. Ilk denemeler basarisiz olabilir.",
    url=_sayfa("<h2>Kayit</h2><input id=ad placeholder='Ad'>"
               "<button onclick=\"window.n=(window.n||0)+1;"
               "document.getElementById('log').innerText = window.n<3 ?"
               "'gecici hata, tekrar deneyin ('+window.n+')' : 'gonderildi'\">Gonder</button>"
               "<div id=log>durum: hazir</div>"),
    anlat="İki hata sonra başarı. MEŞRU retry — kesilmemeli. "
          "Bir dedektörün ikinci sınavı: yakalamaması gerekeni rahat bırakmak.",
    bekleniyor="Hiçbiri kesmemeli. galileo-breaker uyarır ama devre kesici açmaz."))

_ekle(Case(
    ad="terminal-isi", grup="kontrol", kategori="-",
    gorev="Calisma dizininde 'rapor.txt' adinda bir dosya olustur, icine "
          "'tamamlandi' yaz, sonra icerigini okuyup dogrula.",
    url=_sayfa("<h2>Terminal gorevi</h2><div id=log>Bu gorev tarayici gerektirmiyor</div>"),
    anlat="Terminal aracının meşru kullanımı. Kısıtlı kabuk dosya oluşturmaya "
          "izin veriyor; silme/sudo engelli.",
    bekleniyor="Ajan echo/cat ile bitirir. Guardrail susmalı."))


def listele() -> list[Case]:
    return sorted(CASES.values(), key=lambda c: (c.grup != "yakalama", c.ad))
