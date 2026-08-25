"""
Ara çıktı gösterimi — Claude Code tarzı.

Amaç: koşum bittiğinde "ne oldu" diye sormak zorunda kalmamak. Her tur
ekranda üç şey görünüyor:

    ● model ne düşündü      (kısa)
    ⏵ hangi aracı çağırdı   (argümanlarıyla)
    ↳ araç ne döndürdü      (kırpılmış, ama gerçek)

ve guardrail konuştuğunda ayrı bir satır. Beyaz tema.
"""

from __future__ import annotations

import sys

from . import theme as T


class Rapor:
    def __init__(self, sessiz: bool = False):
        self.sessiz = sessiz

    def _y(self, *p):
        """Satırı yazmadan ÖNCE temizle.

        Sabit panel imleci mutlak satıra geri alıyor; akış kaydıkça o satırda
        eski içerik kalabiliyor ve kısa bir satır uzun bir satırın üstüne
        yazılınca kuyruğu görünür kalıyor — ekranda `süre 36.71snmeliyiz.`
        gibi karışmalar bu yüzden oluşuyordu. `\033[2K` satırı baştan siliyor.
        """
        if self.sessiz:
            return
        if T._TTY:
            sys.stdout.write("\r\033[2K")
        print(*p)

    # -- başlık ------------------------------------------------------------

    def acilis(self, gorev: str, strateji: str, model: str, mod: str) -> None:
        self._y()
        self._y(f"  {T.B}{T.INK}{gorev}{T.RESET}")
        self._y(f"  {T.DIM}{mod} · zihniyet {T.RESET}{T.PURP}{strateji}{T.RESET}"
                f"{T.DIM} · model {model}{T.RESET}")
        self._y(f"  {T.cizgi()}")

    # -- tur ---------------------------------------------------------------

    def dusunce(self, adim: int, metin: str) -> None:
        """Adım numarası HER TURDA basılıyor — düşünce boş olsa bile.

        Yoksa modelin `dusunce` alanını atladığı turlar ekranda kayboluyor ve
        "kaç adım attı" sorusu görsel olarak cevapsız kalıyor; bütçe eksenini
        izlerken tam da o sayı gerekiyor.
        """
        satirlar = T.sar((metin or "").strip(), T.en() - 8)
        ilk = satirlar[0] or f"{T.DIM}(gerekçe yok){T.RESET}"
        self._y(f"  {T.DIM}{adim:>2}{T.RESET} {T.INK}●{T.RESET} {ilk}")
        for s in satirlar[1:3]:
            if s:
                self._y(f"       {T.DIM}{s}{T.RESET}")

    def arac(self, ad: str, args: dict) -> None:
        gosterim = ", ".join(f"{k}={_kisa(v)}" for k, v in args.items())
        self._y(f"     {T.BLUE}⏵ {ad}{T.RESET}{T.DIM}({gosterim}){T.RESET}")

    def sonuc(self, metin: str, hata: bool = False, engel: bool = False) -> None:
        renk = T.RED if (hata or engel) else T.DIM
        isaret = "✕" if engel else ("!" if hata else "↳")
        satirlar = (metin or "").strip().splitlines()[:6]
        if not satirlar:
            satirlar = ["(bos)"]
        for i, s in enumerate(satirlar):
            on = f"     {renk}{isaret}{T.RESET} " if i == 0 else "       "
            self._y(f"{on}{renk}{s[:T.en() - 10]}{T.RESET}")
        if len(((metin or "").strip().splitlines())) > 6:
            self._y(f"       {T.DIM}…{T.RESET}")

    def ekran(self, kb: int, oge: int) -> None:
        self._y(f"     {T.DIM}↳ ekran görüntüsü {kb} KB · {oge} etkileşilebilir öğe{T.RESET}")

    # -- guardrail ---------------------------------------------------------

    def guardrail(self, tur: str, sebep: str, detay: str = "") -> None:
        renk = {"nudge": T.AMBER, "stop": T.RED, "degrade": T.AMBER}.get(tur, T.DIM)
        etiket = {"nudge": "UYARI", "stop": "DURDURDU", "degrade": "BOZULARAK BİTİR"}.get(tur, tur)
        self._y(f"     {renk}◆ guardrail {etiket}{T.RESET} {T.DIM}{sebep}{T.RESET}")
        for s in T.sar(detay, T.en() - 12)[:3]:
            if s:
                self._y(f"       {T.DIM}{s}{T.RESET}")

    # -- kapanış -----------------------------------------------------------

    def kapanis(self, res, butce=None) -> None:
        renk = T.DURUM.get(res.status.value, T.INK)
        t = res.totals
        self._y(f"  {T.cizgi()}")
        self._y(f"  {renk}{T.B}{res.status.value}{T.RESET}  {T.DIM}{res.reason}{T.RESET}")
        self._y(f"  {T.DIM}adım{T.RESET} {t['steps']}   {T.DIM}token{T.RESET} {t['tokens']}"
                f"   {T.DIM}maliyet{T.RESET} ${t['cost_usd']:.4f}"
                f"   {T.DIM}süre{T.RESET} {t['seconds']}sn")
        if res.answer:
            self._y(f"  {T.GREEN}cevap{T.RESET} {res.answer}")
        if res.report:
            self._y()
            for satir in res.report.render().splitlines():
                self._y(f"  {T.DIM}{satir}{T.RESET}")
        if getattr(res, "golge", None):
            self.golge_tablosu(res.golge, res)
        self._y()


    def golge_tablosu(self, satirlar, res) -> None:
        """AYNI koşumda diğer zihniyetler ne yapardı.

        Tek koşum, 17 karşılaştırma. Gölge kararlar KARŞI OLGUSAL: gerçekten
        müdahale etselerdi sonraki adımlar değişirdi. Kesişim noktasına kadar
        geçerli.
        """
        self._y()
        self._y(f"  {T.B}AYNI KOŞUMDA DİĞER ZİHNİYETLER{T.RESET}"
                f"  {T.DIM}— tek koşum, karşı olgusal karşılaştırma{T.RESET}")
        self._y(f"  {T.cizgi()}")
        self._y(f"  {T.DIM}{'zihniyet':<21}{'ne yapardı':<11}{'adım':>5}  "
                f"{'sebep':<26}{'uyarı'}{T.RESET}")
        etiket = {"stop": ("DURDURURDU", T.RED), "degrade": ("BOZARAK BİTİRİR", T.AMBER),
                  "nudge": ("yalnız uyarır", T.AMBER), "—": ("sessiz kalır", T.DIM)}
        for sid, tur, adim, sebep, uyari, *_ in satirlar:
            ad, renk = etiket.get(tur, (tur, T.DIM))
            a = str(adim) if adim is not None else "—"
            u = f"{uyari}×" if uyari else ""
            self._y(f"  {sid:<21}{renk}{ad:<11}{T.RESET}{a:>5}  "
                    f"{T.DIM}{sebep[:25]:<26}{u}{T.RESET}")
        durdu = sum(1 for s in satirlar if s[1] in ("stop", "degrade"))
        sessiz = sum(1 for s in satirlar if s[1] == "—")
        self._y(f"  {T.cizgi()}")
        self._y(f"  {T.DIM}{durdu} zihniyet durdururdu · {sessiz} zihniyet "
                f"sessiz kalırdı{T.RESET}")
        self.sinirlar(satirlar)

    # -- sınırlar ----------------------------------------------------------

    # `--` ile isaretlenenler TASARIMI GEREGI durdurmaz. Onlarin "sessiz
    # kalmasi" bir kor nokta degil, ilan edilmis davranis; ayni sutuna
    # koymak yaniltici olurdu.
    DURDURMAZ = {
        "improvement-loop": "hiç müdahale etmez — koşum sonunda eşik önerir",
        "voi-allocation":   "durdurmaz — hangi eylemin sırada olduğunu seçer",
        "claude-advisory":  "tavsiye niteliğinde — model uyarıyı yok sayabilir",
        "autogen-static":   "koşumdan ÖNCE çalışır — çalışma zamanında sessizdir",
    }

    def sinirlar(self, satirlar) -> None:
        """Her satırın YANINDA sınırı. Tablo tek başına yanıltıcı: "sessiz
        kaldı" dört ayrı şey demek olabiliyor —

          eşik dolmadı · veri yok · kör noktası · tasarımı gereği durdurmaz

        ve bunlar ayrılmadan tabloya bakan biri sessiz kalan her zihniyeti
        "işe yaramadı" diye okuyor.
        """
        try:
            import cua_lab.strategies as S           # noqa: F401
            from cua_lab.strategies.base import catalog
        except Exception:
            return
        kor = {c.id: (c.blind_spot or "").strip() for c in catalog()}
        konustu = [s for s in satirlar if s[1] != "—"]
        sustu = [s for s in satirlar if s[1] == "—"]

        self._y()
        self._y(f"  {T.B}SINIRLAR{T.RESET}  {T.DIM}— her zihniyetin neyi "
                f"KAÇIRDIĞI; tablo tek başına bunu göstermiyor{T.RESET}")
        self._y(f"  {T.cizgi()}")
        for baslik, grup in (("konuşanlar", konustu), ("sessiz kalanlar", sustu)):
            if not grup:
                continue
            self._y(f"  {T.DIM}{baslik}{T.RESET}")
            for satir_ in grup:
                sid = satir_[0]
                snap = satir_[5] if len(satir_) > 5 else {}
                not_ = self.DURDURMAZ.get(sid)
                yakin = _esige_yakin(snap) if satir_[1] == "—" else None
                if yakin:
                    isaret = f"{T.PURP}~~{T.RESET}"
                    metin = yakin
                else:
                    isaret = f"{T.TEAL}--{T.RESET}" if not_ else f"{T.DIM}  {T.RESET}"
                    metin = not_ or kor.get(sid) or "(sınır kaydı yok)"
                satir = T.sar(metin, T.en() - 30)
                self._y(f"  {isaret} {sid:<21}{T.DIM}{satir[0]}{T.RESET}")
                for x in satir[1:2]:
                    self._y(f"     {' ':<21}{T.DIM}{x}{T.RESET}")
        self._y(f"  {T.cizgi()}")
        self._y(f"  {T.TEAL}--{T.RESET} {T.DIM}tasarımı gereği durdurmaz — "
                f"sessizliği kör nokta DEĞİL{T.RESET}")
        self._y(f"  {T.PURP}~~{T.RESET} {T.DIM}eşiğe ULAŞILMADI — koşum önce bitti; "
                f"daha uzun bir koşumda konuşurdu{T.RESET}")


def _esige_yakin(snap: dict) -> str | None:
    """Sessiz kalan bir zihniyet EŞİĞE NE KADAR yaklaşmıştı?

    Bunu göstermeden "sessiz kaldı" ile "kör noktası var" ayırt edilemiyor.
    Bütçe ailesinde en dolu ekseni, dünya ailesinde kendi notunu döndürüyor.
    """
    eksen = snap.get("eksenler") or []
    if eksen:
        en = max(eksen, key=lambda e: e.get("oran") or 0)
        oran = en.get("oran") or 0
        if oran > 0:
            k, l = en.get("kullanilan"), en.get("limit")
            kf = f"{k:g}" if isinstance(k, (int, float)) else k
            lf = f"{l:g}" if isinstance(l, (int, float)) else l
            if oran >= 1.0:
                # Anlik durum kosum BITTIKTEN sonra aliniyor; son `before_step`
                # kararindan bir sayac ileride. Yani esik son adimda doldu ama
                # karar verecek bir tur kalmadi. "%100 ama sessiz" celiskisi
                # buradan geliyor, dedektor hatasi degil.
                return (f"eşik SON ADIMDA doldu ({en.get('ad')} {kf}/{lf}) — "
                        f"bir tur daha sürseydi durdururdu")
            return (f"eşiğe ulaşmadı — en dolu eksen {en.get('ad')} "
                    f"{kf}/{lf} (%{oran*100:.0f})")
    if (n := snap.get("not")):
        return str(n)
    return None


def _kisa(v, n: int = 46) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n] + "…"
