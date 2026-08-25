"""
agentcli — terminal uygulaması.

Bir kez başlatılıyor, içinde yaşanıyor. `--` ile başlayan her şey KOMUT,
geri kalan her şey AJANA VERİLEN GÖREV.

Tarayıcı bir kez açılıyor ve oturum boyunca AÇIK KALIYOR — görevler arasında
sayfa korunuyor, her görevde yeniden Chrome başlatma maliyeti yok.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_KOK / "cua_lab"))

from cua_lab import strategies as S                      # noqa: E402
from cua_lab.detect.guardrails import BudgetLimits       # noqa: E402
from cua_lab.strategies.kinds import KATEGORI            # noqa: E402

from . import theme as T                                 # noqa: E402
from .agent import Ajan                                  # noqa: E402
from .cases import CASES, listele                        # noqa: E402
from .model import VLM, VLMModel                         # noqa: E402
from .panel import Panel                              # noqa: E402
from .render import Rapor                                # noqa: E402
from .tools.browser import Browser                       # noqa: E402
from .tools.terminal import Terminal                     # noqa: E402

try:
    import readline                                      # noqa: F401
except ImportError:
    readline = None

KOMUTLAR = ["--case", "--kategori", "--strategy", "--model", "--url", "--workdir", "--budget",
            "--screen", "--browser", "--tema", "--panel", "--desktop", "--info",
            "--status",
            "--help",
            "--quit"]


class Uygulama:
    def __init__(self, port: int = 9222, workdir: str | None = None):
        self.strateji = "openhands-stuck"
        self.model_ad = VLM
        self.gorsel = True
        self.gorunur = False
        self.port = port
        # Varsayılan çalışma dizini GÖRÜNÜR ve KALICI olmalı.
        # `mkdtemp` her açılışta rastgele bir `/tmp/agentcli_xxxx` üretiyordu;
        # ajan dosyayı doğru oluşturuyor ama kullanıcı bulamıyordu — ölçüldü,
        # "hiçbir şey oluşturmadı" diye rapor edildi, oysa dosya oradaydı.
        self.kok = workdir or str(Path.cwd() / "agentcli-calisma")
        os.makedirs(self.kok, exist_ok=True)
        self.limits = BudgetLimits(max_steps=14, max_replans=5, max_tokens=60000,
                                   max_seconds=240.0, max_cost_usd=0.5)
        self.panel_acik = True
        self.masaustu = None            # --desktop ile aciliyor
        self.browser: Browser | None = None
        self.terminal = Terminal(self.kok)
        self._model: VLMModel | None = None
        self.gecmis: list = []

    # ------------------------------------------------------------ kaynaklar

    def _tarayici(self) -> Browser:
        if self.browser is None:
            print(f"  {T.DIM}tarayıcı başlatılıyor…{T.RESET}", end="", flush=True)
            self.browser = Browser(port=self.port, headless=not self.gorunur,
                                   konum="sag" if self.gorunur else None)
            self.browser.start()
            nerede = " · ekranın sağ yarısında" if self.gorunur else ""
            print(f"\r  {T.GREEN}✓{T.RESET} tarayıcı hazır"
                  f"{T.DIM}{nerede}{T.RESET}          ")
        return self.browser

    def _vlm(self) -> VLMModel:
        if self._model is None or self._model.model != self.model_ad:
            self._model = VLMModel(self.model_ad, gorsel=self.gorsel)
        self._model.gorsel = self.gorsel
        return self._model

    def _tarayici_kapat(self) -> None:
        if self.browser:
            self.browser.stop()
            self.browser = None

    # ------------------------------------------------------------ ekran

    def banner(self) -> None:
        w = T.en()
        print()
        print(f"  {T.LINE}╭{'─' * (w - 6)}╮{T.RESET}")
        baslik = "agentcli · VLM computer-use ajanı"
        alt = "terminal · tarayıcı (DOM + ekran) · seçilebilir guardrail"
        # Kutu geniÅŸliği: "  │ " + metin + dolgu + "│"  =  ust kenarla ayni
        def _satir(metin: str, stil: str) -> str:
            dolgu = " " * max(0, w - 7 - len(metin))
            return f"  {T.LINE}│{T.RESET} {stil}{metin}{T.RESET}{dolgu}{T.LINE}│{T.RESET}"
        print(_satir(baslik, T.B + T.INK))
        print(_satir(alt, T.DIM))
        print(f"  {T.LINE}╰{'─' * (w - 6)}╯{T.RESET}")

    def durum(self) -> None:
        k = next((c for c in S.catalog() if c.id == self.strateji.split(",")[0].split(":")[0]), None)
        l = self.limits
        print()
        print(f"  {T.DIM}{'zihniyet':<11}{T.RESET}{T.PURP}{T.B}{self.strateji}{T.RESET}")
        if k:
            print(f"  {'':<11}{T.DIM}{k.mentality}{T.RESET}")
        print(f"  {T.DIM}{'model':<11}{T.RESET}{self.model_ad.split('/')[-1]}"
              f"{T.DIM}   ekran görüntüsü: {'AÇIK' if self.gorsel else 'kapalı'}{T.RESET}")
        sayfa = ""
        if self.browser:
            try:
                sayfa = f"   {self.browser.dom()['url'][:52]}"
            except Exception:
                pass
        print(f"  {T.DIM}{'tarayıcı':<11}{T.RESET}"
              f"{'açık' if self.browser else 'kapalı'}"
              f"{T.DIM}   {'görünür' if self.gorunur else 'headless'}"
              f"{sayfa}{T.RESET}")
        if self.masaustu is None:
            md = f"{T.DIM}kapalı{T.RESET}"
        else:
            md = (f"{T.RED}AÇIK · gerçek fare/klavye{T.RESET}"
                  if self.masaustu.allow_input
                  else f"{T.AMBER}salt okuma — dokunmaz{T.RESET}")
        print(f"  {T.DIM}{'masaüstü':<11}{T.RESET}{md}")
        print(f"  {T.DIM}{'dizin':<11}{T.RESET}{self.kok}")
        print(f"  {T.DIM}{'bütçe':<11}{T.RESET}{l.max_steps} adım · {l.max_tokens} tok"
              f" · {l.max_seconds:g}sn · ${l.max_cost_usd:g}")
        print()

    def yardim(self) -> None:
        satirlar = [
            ("--case", "hazır senaryo seç ve koştur (liste açılır)"),
            ("--kategori", "bir ZİHNİYET AİLESİNİN tamamını aynı göreve koştur"),
            ("--strategy", "guardrail zihniyeti seç · virgülle üst üste konur"),
            ("--model", "VLM seç"),
            ("--url", "tarayıcıyı bir adrese götür"),
            ("--workdir", "terminal çalışma dizinini değiştir"),
            ("--budget", "bütçe eksenlerini ayarla"),
            ("--screen", "ekran görüntüsü göndermeyi aç/kapa"),
            ("--browser", "tarayıcıyı görünür/gizli yap"),
            ("--tema", "beyaz arka planı aç/kapa"),
            ("--panel", "sağ üst canlı kısıt panelini aç/kapa"),
            ("--desktop", "GERÇEK masaüstü aracı: kapalı → salt okuma → tam"),
            ("--info", "seçili zihniyeti anlat"),
            ("--status", "mevcut ayarlar"),
            ("--help", "bu liste"),
            ("--quit", "çık  (Ctrl-D de olur)"),
        ]
        print()
        for a, b in satirlar:
            print(f"  {T.BLUE}{a:<13}{T.RESET}{T.DIM}{b}{T.RESET}")
        print(f"\n  {T.DIM}`--` ile başlamayan her şey AJANA VERİLEN GÖREV olarak koşar."
              f"{T.RESET}")
        print(f"  {T.DIM}örnek:  {T.RESET}wikipedia.org'a git ve Türkiye sayfasını aç")
        print(f"  {T.DIM}örnek:  {T.RESET}çalışma dizininde notlar.txt oluştur, "
              f"içine üç satır yaz\n")

    # ------------------------------------------------------------ seçiciler

    def _sec(self, baslik: str, secenekler: list[tuple[str, str]], simdiki: str,
             basliklar: dict | None = None, coklu: bool = False) -> str | None:
        ad_en = max((len(a) for a, _ in secenekler), default=12) + 1
        girinti = 2 + 1 + 1 + 2 + 2 + ad_en
        metin_en = max(28, T.en() - girinti - 2)
        print(f"\n  {T.B}{baslik}{T.RESET}")
        print(f"  {T.cizgi()}")
        for i, (ad, aciklama) in enumerate(secenekler, 1):
            if basliklar and (i - 1) in basliklar:
                print(f"\n  {T.DIM}{basliklar[i - 1]}{T.RESET}")
            isaret = f"{T.GREEN}●{T.RESET}" if ad == simdiki else " "
            sat = T.sar(aciklama, metin_en)
            print(f"  {isaret} {T.BLUE}{i:>2}{T.RESET}  {ad:<{ad_en}}{T.DIM}{sat[0]}{T.RESET}")
            for s in sat[1:]:
                print(f"{' ' * girinti}{T.DIM}{s}{T.RESET}")
        print(f"    {T.DIM} 0  vazgeç{T.RESET}")
        ipucu = "numara seç · virgülle birden fazla (üst üste konur)" if coklu else "numara seç"
        print(f"\n  {T.DIM}{ipucu}{T.RESET}")
        try:
            ham = input(f"  {T.BLUE}seçim ▸{T.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return None
        if not ham or ham == "0":
            return None
        secili = []
        for p in [x.strip() for x in ham.split(",") if x.strip()]:
            if p.isdigit() and 1 <= int(p) <= len(secenekler):
                secili.append(secenekler[int(p) - 1][0])
            elif p in {a for a, _ in secenekler}:
                secili.append(p)
            else:
                print(f"  {T.RED}geçersiz: {p}{T.RESET}"); return None
        return ",".join(secili)

    def sec_strateji(self) -> None:
        kat = S.catalog()
        secenekler, basliklar = [], {}
        for k, (ad, ozet) in KATEGORI.items():
            grup = [c for c in kat if getattr(c, "kind", "-") == k]
            if not grup:
                continue
            basliklar[len(secenekler)] = f"{ad} — {ozet}"
            for c in sorted(grup, key=lambda x: (x.priority or 99, x.id)):
                secenekler.append((c.id, c.action))
        yeni = self._sec("ZİHNİYET SEÇ  ·  sıra = uygulama önceliği",
                         secenekler, self.strateji, basliklar, coklu=True)
        if yeni:
            self.strateji = yeni
            print(f"  {T.GREEN}✓{T.RESET} zihniyet: {T.B}{yeni}{T.RESET}")

    def sec_case(self) -> None:
        secenekler, basliklar, grup = [], {}, None
        for c in listele():
            if c.grup != grup:
                grup = c.grup
                basliklar[len(secenekler)] = (
                    "YAKALAMA — guardrail devreye girmeli" if grup == "yakalama"
                    else "KONTROL — guardrail SUSMALI, meşru koşum")
            secenekler.append((c.ad, c.anlat))
        yeni = self._sec("SENARYO SEÇ", secenekler, "", basliklar)
        if not yeni or "," in yeni:
            return
        c = CASES[yeni]
        print(f"\n  {T.DIM}beklenen:{T.RESET} {c.bekleniyor}")
        br = self._tarayici()
        print(f"  {T.DIM}{br.goto(c.url)[:70]}{T.RESET}")
        self.kos(c.gorev, f"senaryo: {c.ad}")

    def kategori_kos(self, arg: str = "") -> None:
        """Bir zihniyet ailesinin TAMAMINI aynı göreve karşı gerçekten koştur.

        Gölge tablosundan farkı: orada tek koşum var ve diğerleri "müdahale
        etseydi ne olurdu" diye tahmin ediliyor. Burada her zihniyet kendi
        koşumunu yapıyor, kendi token'ını yakıyor. Tahmin değil ölçüm.
        """
        from . import kategori as K

        kind = arg.strip().lower()
        if kind not in K.KATEGORILER:
            secenekler = [(k, f"{ad} — {ne}  ({len(K.uyeler(k))} zihniyet)")
                          for k, (ad, ne) in K.KATEGORILER.items()]
            kind = self._sec("ZİHNİYET AİLESİ SEÇ", secenekler, "") or ""
            if not kind or "," in kind or kind not in K.KATEGORILER:
                return

        # Görev: bu kategoriyi sınayan case'i öner, yoksa serbest görev sor.
        onerilen = [c for c in listele() if c.kategori == kind]
        secenekler = [(c.ad, f"{c.grup.upper()} · {c.anlat[:66]}") for c in listele()]
        basliklar = {}
        if onerilen:
            print(f"\n  {T.DIM}bu aileyi sınamak için tasarlanan senaryo: "
                  f"{T.RESET}{T.B}{', '.join(c.ad for c in onerilen)}{T.RESET}")
        sec = self._sec("HANGİ SENARYODA", secenekler, onerilen[0].ad if onerilen else "",
                        basliklar)
        if not sec or "," in sec or sec not in CASES:
            return
        c = CASES[sec]
        print(f"  {T.DIM}beklenen:{T.RESET} {c.bekleniyor}")

        model = self._vlm()
        br = self._tarayici()

        def kur(sid, gorev, url, _):
            # Her zihniyet TEMİZ sayfayla başlasın: öncekinin bıraktığı
            # durum ikinci koşumu kirletir ve karşılaştırma anlamını yitirir.
            br.goto(url)
            ajan = Ajan(gorev, model, br, self.terminal, sid,
                        limits=self.limits, rapor=Rapor(sessiz=True),
                        gorsel=self.gorsel, desktop=self.masaustu, golge=False)
            return ajan.kos()

        try:
            satirlar = K.kos(kind, c.gorev, c.url, kur)
        except S.UnknownStrategy as e:
            print(f"  {T.RED}'{e.id}' zihniyeti yok{T.RESET}\n"); return
        K.tablo(satirlar, kind, c.gorev)

    def sec_model(self) -> None:
        secenekler = [
            ("Qwen/Qwen2.5-VL-72B-Instruct", "vision · GUI grounding · varsayılan"),
            ("google/gemma-3-27b-it", "vision · daha ucuz, daha az kesin"),
            ("meta-llama/Llama-3.1-8B-Instruct", "GÖRÜNTÜ YOK — yalnız DOM metniyle çalışır"),
        ]
        yeni = self._sec("MODEL SEÇ", secenekler, self.model_ad)
        if yeni and "," not in yeni:
            self.model_ad = yeni
            self._model = None
            print(f"  {T.GREEN}✓{T.RESET} model: {T.B}{yeni.split('/')[-1]}{T.RESET}")
            if "VL" not in yeni and "gemma" not in yeni:
                self.gorsel = False
                print(f"  {T.AMBER}!{T.RESET} {T.DIM}bu model görüntü okumuyor — "
                      f"ekran görüntüsü kapatıldı{T.RESET}")

    def ayarla_butce(self) -> None:
        l = self.limits
        alanlar = [("max_steps", l.max_steps), ("max_replans", l.max_replans),
                   ("max_tokens", l.max_tokens), ("max_seconds", l.max_seconds),
                   ("max_cost_usd", l.max_cost_usd)]
        print(f"\n  {T.B}BÜTÇE — beş eksen{T.RESET}  "
              f"{T.DIM}(boş = değiştirme · '-' = ekseni kapat){T.RESET}")
        print(f"  {T.cizgi()}")
        yeni = {}
        for ad, simdi in alanlar:
            try:
                v = input(f"  {ad:<14}{T.DIM}[{simdi}]{T.RESET} ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            if not v:
                continue
            if v == "-":
                yeni[ad] = None
            else:
                try:
                    yeni[ad] = (float(v) if ad in ("max_seconds", "max_cost_usd")
                                else int(v))
                except ValueError:
                    print(f"  {T.RED}sayı değil: {v}{T.RESET}"); return
        if yeni:
            self.limits = BudgetLimits(**{**dict(alanlar), **yeni})
            print(f"  {T.GREEN}✓{T.RESET} bütçe güncellendi")

    def _dizin_durumu(self) -> dict:
        d = {}
        for kok, _, dosyalar in os.walk(self.kok):
            for ad in dosyalar:
                y = os.path.join(kok, ad)
                try:
                    d[os.path.relpath(y, self.kok)] = os.path.getsize(y)
                except OSError:
                    pass
        return d

    def _dizin_ozeti(self, oncesi: dict) -> None:
        """Koşumda hangi dosyalar oluştu/değişti — ve NEREDE.

        Ajan dosyayı doğru yere yazsa bile kullanıcı yolu görmezse iş
        yapılmamış görünüyor.
        """
        sonrasi = self._dizin_durumu()
        yeni = [a for a in sonrasi if a not in oncesi]
        degisen = [a for a in sonrasi if a in oncesi and sonrasi[a] != oncesi[a]]
        if not yeni and not degisen:
            return
        print(f"  {T.GREEN}dosyalar{T.RESET} {T.DIM}{self.kok}{T.RESET}")
        for a in sorted(yeni):
            print(f"    {T.GREEN}+{T.RESET} {a}  {T.DIM}({sonrasi[a]} B){T.RESET}")
        for a in sorted(degisen):
            print(f"    {T.AMBER}~{T.RESET} {a}  {T.DIM}({oncesi[a]} → {sonrasi[a]} B){T.RESET}")

    def _masaustu_kademe(self) -> None:
        """Üç kademe: kapalı → salt okuma → tam kontrol.

        Kademeler tek tek geçiliyor; "kapalı"dan doğrudan "tam kontrol"e
        atlanamıyor. Gerçek fare/klavye yetkisi kazara açılmasın diye.
        """
        from .tools.desktop import Desktop
        if self.masaustu is None:
            self.masaustu = Desktop(allow_input=False)
            print(f"  {T.AMBER}✓{T.RESET} masaüstü: {T.B}SALT OKUMA{T.RESET}"
                  f"{T.DIM} — ekranı görür, ne yapacağını söyler, DOKUNMAZ{T.RESET}")
            print(f"  {T.DIM}  ekran görüntün artık VLM'e gidiyor: o anda ekranda "
                  f"ne varsa.{T.RESET}")
        elif not self.masaustu.allow_input:
            print(f"  {T.RED}!{T.RESET} Gerçek fare ve klavye açılacak. "
                  f"Ölçtüğümüz gerçek: bu model 40 çağrının 40'ında aynı yere tıkladı.")
            print(f"  {T.DIM}  Kaçış: fareyi SOL ÜST KÖŞEYE taşı — koşum anında iptal."
                  f"{T.RESET}")
            try:
                onay = input(f"  yazarak onayla [{T.B}ACIK{T.RESET}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            if onay != "ACIK":
                print(f"  {T.DIM}vazgeçildi — salt okuma devam{T.RESET}"); return
            self.masaustu.stop()
            self.masaustu = Desktop(allow_input=True)
            print(f"  {T.RED}✓ masaüstü: TAM KONTROL{T.RESET}"
                  f"{T.DIM} — gerçek fare/klavye. Silme tuşları ve terminal "
                  f"pencereleri hâlâ engelli.{T.RESET}")
        else:
            self.masaustu.stop()
            self.masaustu = None
            print(f"  {T.GREEN}✓{T.RESET} masaüstü: kapalı")

    def bilgi(self) -> None:
        for sid in self.strateji.split(","):
            ad = sid.split(":")[0].strip()
            k = next((c for c in S.catalog() if c.id == ad), None)
            if not k:
                continue
            kat = KATEGORI.get(getattr(k, "kind", "-"), ("-", ""))[0]
            print(f"\n  {T.B}{k.id}{T.RESET} — {k.title}")
            print(f"  {T.DIM}{kat} · öncelik {k.priority or '-'} · {k.source}{T.RESET}\n")
            for etiket, metin, renk in (("NEDEN GEREKLİ", k.why, T.GREEN),
                                        ("NE YAPAR", k.action, T.BLUE),
                                        ("NEYİ KAÇIRIR", k.blind_spot, T.RED)):
                sat = T.sar(metin, T.en() - 20)
                print(f"  {renk}{etiket:<16}{T.RESET}{sat[0]}")
                for s in sat[1:]:
                    print(f"  {'':<16}{s}")
        print(f"\n  {T.DIM}tam anlatım: cua_lab/docs/zihniyetler.md{T.RESET}\n")

    # ------------------------------------------------------------ koşum

    def kos(self, gorev: str, mod: str = "görev") -> None:
        br = self._tarayici()
        model = self._vlm()
        oncesi = self._dizin_durumu()
        rapor = Rapor()
        rapor.acilis(gorev, self.strateji, model.ad, mod)
        try:
            panel = Panel(aktif=self.panel_acik)
            ajan = Ajan(gorev, model, br, self.terminal, self.strateji,
                        limits=self.limits, rapor=rapor, gorsel=self.gorsel,
                        panel=panel, desktop=self.masaustu)
            try:
                res = ajan.kos()
            finally:
                panel.kaldir()
        except S.UnknownStrategy as e:
            print(f"  {T.RED}'{e.id}' zihniyeti yok.{T.RESET} "
                  f"{T.DIM}mevcut: {', '.join(e.mevcut)}{T.RESET}\n")
            return
        except KeyboardInterrupt:
            print(f"\n  {T.AMBER}kesildi{T.RESET}\n"); return
        rapor.kapanis(res)
        self._dizin_ozeti(oncesi)
        if self.masaustu is not None:
            print(f"  {T.DIM}masaüstü: {self.masaustu.rapor()}{T.RESET}")
            for e in self.masaustu.engellenen[-4:]:
                print(f"    {T.RED}✕ {e}{T.RESET}")
        if self.terminal.engellenen:
            print(f"  {T.RED}engellenen komutlar{T.RESET}")
            for e in self.terminal.engellenen[-5:]:
                print(f"    {T.DIM}{e}{T.RESET}")
            print()
        self.gecmis.append((gorev, self.strateji, res))

    # ------------------------------------------------------------ döngü

    def _tamamla(self, metin, durum):
        adaylar = [k for k in KOMUTLAR if k.startswith(metin)]
        return adaylar[durum] if durum < len(adaylar) else None

    def calistir(self) -> int:
        if readline:
            readline.set_completer(self._tamamla)
            readline.parse_and_bind("tab: complete")
        T.beyaz_ac()
        self.banner()
        self.durum()
        print(f"  {T.DIM}komutlar için {T.RESET}{T.BLUE}--help{T.RESET}"
              f"{T.DIM} · görev yazıp Enter'a bas{T.RESET}\n")
        try:
            while True:
                try:
                    satir = input(f"{T.BLUE}▸{T.RESET} ").strip()
                except (EOFError, KeyboardInterrupt):
                    print(f"\n  {T.DIM}görüşürüz{T.RESET}\n"); return 0
                if not satir:
                    continue
                if satir.startswith("--"):
                    self._komut(satir)
                    continue
                self.kos(satir)
        finally:
            self._tarayici_kapat()
            T.beyaz_kapat()

    def _komut(self, satir: str) -> None:
        parca = satir.split(maxsplit=1)
        k, arg = parca[0], (parca[1].strip() if len(parca) > 1 else "")
        if k in ("--quit", "--exit"):
            self._tarayici_kapat()
            print(f"  {T.DIM}görüşürüz{T.RESET}")
            T.beyaz_kapat()
            raise SystemExit(0)
        if k == "--help":
            self.yardim()
        elif k == "--status":
            self.durum()
        elif k == "--strategy":
            if arg:
                self.strateji = arg
                print(f"  {T.GREEN}✓{T.RESET} zihniyet: {T.B}{arg}{T.RESET}")
            else:
                self.sec_strateji()
            self.durum()
        elif k == "--case":
            self.sec_case()
        elif k == "--kategori":
            self.kategori_kos(arg)
        elif k == "--model":
            self.sec_model()
        elif k == "--info":
            self.bilgi()
        elif k == "--budget":
            self.ayarla_butce(); self.durum()
        elif k == "--url":
            if not arg:
                print(f"  {T.DIM}kullanım: --url wikipedia.org{T.RESET}"); return
            print(f"  {T.DIM}{self._tarayici().goto(arg)[:80]}{T.RESET}")
        elif k == "--workdir":
            if not arg:
                print(f"  {T.DIM}şu an: {T.RESET}{self.kok}")
                d = self._dizin_durumu()
                for ad in sorted(d)[:15]:
                    print(f"    {ad}  {T.DIM}({d[ad]} B){T.RESET}")
                if not d:
                    print(f"    {T.DIM}(boş){T.RESET}")
                return
            self.kok = str(Path(arg).expanduser().resolve())
            self.terminal = Terminal(self.kok)
            print(f"  {T.GREEN}✓{T.RESET} dizin: {self.kok}")
        elif k == "--screen":
            self.gorsel = not self.gorsel
            print(f"  {T.GREEN}✓{T.RESET} ekran görüntüsü: "
                  f"{'AÇIK' if self.gorsel else 'kapalı'}"
                  f"{T.DIM}  (kapalıyken yalnız DOM metni gider — çok daha ucuz){T.RESET}")
        elif k == "--desktop":
            self._masaustu_kademe()
        elif k == "--panel":
            self.panel_acik = not self.panel_acik
            print(f"  {T.GREEN}✓{T.RESET} canlı panel: "
                  f"{'AÇIK' if self.panel_acik else 'kapalı'}")
        elif k == "--tema":
            if T._acik:
                T.beyaz_kapat()
                print(f"  {T.GREEN}✓{T.RESET} terminalin kendi teması")
            else:
                T.beyaz_ac()
                self.banner(); self.durum()
                print(f"  {T.GREEN}✓{T.RESET} beyaz arka plan")
        elif k == "--browser":
            self.gorunur = not self.gorunur
            self._tarayici_kapat()
            print(f"  {T.GREEN}✓{T.RESET} tarayıcı: "
                  f"{'GÖRÜNÜR — ekranın sağ yarısında' if self.gorunur else 'headless'}"
                  f"{T.DIM}  (sonraki görevde açılacak){T.RESET}")
        else:
            print(f"  {T.RED}bilinmeyen komut: {k}{T.RESET}  {T.DIM}--help{T.RESET}")
