"""
İnteraktif kabuk — `--strategy` yazınca zihniyet seçicisi açılıyor, serbest
metin yazınca o zihniyetle koşuyor.

TASARIM: Zihniyet bir BAYRAK değil, bir MOD. Kabuk açıkken seçili zihniyet
ekranın üstünde duruyor ve her görev onun altında koşuyor. Amaç şu soruyu
kolay sorulabilir hale getirmek: "aynı görevi başka bir zihniyetle koşarsam
ne değişir?"
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import strategies as S
from . import ui
from .events import BudgetLimits
from .loop import Runner
from .sandbox.fake import SCENARIOS, FakeSandbox

try:
    import readline  # noqa: F401  — ok tuşu geçmişi, stdlib
except ImportError:
    readline = None


AILE = {"src": "ben_ekledim", "harness": "harness", "-": "taban"}


def _sar(metin: str, en: int) -> list[str]:
    """Basit sözcük sarma — textwrap yerine, tek amaçlı ve kısa."""
    out, satir = [], ""
    for kelime in metin.split():
        if len(satir) + len(kelime) + 1 > en:
            out.append(satir); satir = kelime
        else:
            satir = f"{satir} {kelime}".strip()
    if satir:
        out.append(satir)
    return out


class Shell:
    def __init__(self, models: dict, planned: list, task: str = "Formu doldur ve gonder"):
        self.models = models
        self.planned = planned
        self.strategy = "openhands-stuck"
        self.scenario = "dead_button"
        self.model = "stubborn"
        self.task = task
        self.trace_dir: Path | None = None
        self.limits = BudgetLimits()
        self.gecmis: list = []
        self._gorev_uyarisi = False

    # ------------------------------------------------------------------ ekran

    def banner(self) -> None:
        # `none` bir zihniyet degil taban cizgisi — sayima girmiyor.
        yazili, toplam = len(set(S.all_ids()) - {"none"}), len(self.planned)
        print()
        print(ui.title("cua-lab  ·  computer-use ajani",
                       f"secilebilir guvenilirlik zihniyetleri  —  {yazili}/{toplam} kodda"))

    def durum(self) -> None:
        c = ui.CYAN
        print()
        print(ui.kv("zihniyet", self.strategy, 11, ui.B + c))
        kat = next((x for x in S.catalog() if x.id == self.strategy.split(",")[0]), None)
        if kat:
            print(f"  {ui.GREY}{'':<11}{ui.DIM}{kat.mentality}{ui.RESET}")
        print(ui.kv("ortam", self.scenario, 11))
        print(ui.kv("model", self.model, 11,
                    ui.YELLOW if self.model == "hf" else ""))
        l = self.limits
        print(ui.kv("butce", f"{l.max_steps} adim · {l.max_tokens} tok · "
                             f"{l.max_seconds:g}sn · ${l.max_cost_usd:g}", 11))
        print()

    def _blok(self, etiket: str, metin: str, renk: str = "") -> None:
        """Etiketli, sarmalı metin bloğu. Boş alan basılmıyor."""
        if not metin:
            return
        satirlar = _sar(metin, ui.width() - 20)
        print(f"  {renk}{etiket:<15}{ui.RESET}{satirlar[0]}")
        for s in satirlar[1:]:
            print(f"  {'':<15}{s}")

    def ekran_ozeti(self) -> str:
        """Ajanin gordugu ekran — gorev yazarken neye gore yazacagini gostermek icin."""
        s = FakeSandbox(self.scenario)
        s.start()
        return s.describe()

    def yardim(self) -> None:
        satirlar = [
            ("--strategy", "zihniyet sec (liste acilir)"),
            ("--scenario", "ortam sec"),
            ("--model", "model sec"),
            ("--budget", "butce eksenlerini ayarla"),
            ("--info", "secili zihniyeti anlat"),
            ("--compare", "bu gorevi BUTUN zihniyetlerle kosur"),
            ("--trace", "iz dosyasi yaz/kapat"),
            ("--status", "mevcut ayarlar"),
            ("--help", "bu liste"),
            ("--quit", "cik  (Ctrl-D de olur)"),
        ]
        print()
        for k, v in satirlar:
            print(f"  {ui.CYAN}{k:<12}{ui.RESET}{ui.DIM}{v}{ui.RESET}")
        print(f"\n  {ui.DIM}Baska ne yazarsan GOREV sayilir ve secili zihniyetle kosar."
              f"  Sohbet arayuzu DEGIL.{ui.RESET}")
        print(f"  {ui.GREY}ajanin gordugu ekran:{ui.RESET} {ui.DIM}{self.ekran_ozeti()}{ui.RESET}\n")

    # ------------------------------------------------------------------ seçici

    def _sec(self, baslik: str, secenekler: list[tuple[str, str]],
             simdiki: str, basliklar: dict | None = None,
             coklu: bool = False) -> str | None:
        """Numaralı seçici. 0 = vazgeç. Virgüllü giriş = üst üste koyma."""
        print(f"\n  {ui.B}{baslik}{ui.RESET}")
        print(ui.rule())
        # Sutun genisligi TERMINALDEN hesaplaniyor; sabit kesme aciklamayi
        # yarida birakiyordu.
        ad_en = max((len(a) for a, _ in secenekler), default=12) + 1
        # "  " + isaret(1) + " " + numara(2) + "  " + ad
        girinti = 2 + 1 + 1 + 2 + 2 + ad_en
        metin_en = max(28, ui.width() - girinti - 2)
        for i, (ad, aciklama) in enumerate(secenekler, 1):
            if basliklar and (i - 1) in basliklar:
                print(f"\n  {ui.GREY}{basliklar[i - 1]}{ui.RESET}")
            isaret = f"{ui.GREEN}●{ui.RESET}" if ad == simdiki else " "
            satirlar = _sar(aciklama, metin_en) or [""]
            print(f"  {isaret} {ui.CYAN}{i:>2}{ui.RESET}  {ad:<{ad_en}}"
                  f"{ui.DIM}{satirlar[0]}{ui.RESET}")
            for s in satirlar[1:]:
                print(f"{' ' * girinti}{ui.DIM}{s}{ui.RESET}")
        print(f"    {ui.GREY} 0{ui.RESET}  {ui.DIM}vazgec{ui.RESET}")
        ipucu = ("numara sec · virgulle birden fazla (ust uste konur) · "
                 "ayrinti icin --info") if coklu else "numara sec"
        print(f"\n  {ui.GREY}{ipucu}{ui.RESET}")
        try:
            ham = input(f"  {ui.CYAN}secim ▸{ui.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not ham or ham == "0":
            return None
        # Virgüllü: birden fazla zihniyeti üst üste koy.
        parcalar = [p.strip() for p in ham.split(",") if p.strip()]
        secili = []
        for p in parcalar:
            if p.isdigit() and 1 <= int(p) <= len(secenekler):
                secili.append(secenekler[int(p) - 1][0])
            elif p in {a for a, _ in secenekler}:
                secili.append(p)
            else:
                print(f"  {ui.RED}gecersiz: {p}{ui.RESET}")
                return None
        return ",".join(secili)

    def sec_strateji(self) -> None:
        # Kategoriye gore grupla: ayni kategoridekiler ayni ORTAK MANTIGI
        # paylasiyor, yalnizca bir kararda ayrisiyorlar.
        from .strategies.kinds import KATEGORI
        kat = S.catalog()
        kayitli, basliklar = [], {}
        for k, (ad, ozet) in KATEGORI.items():
            grup = [c for c in kat if getattr(c, "kind", "-") == k]
            if not grup:
                continue
            basliklar[len(kayitli)] = f"{ad} — {ozet}"
            # Kategori icinde ONCELIGE gore sirala: "once hangisini koymaliyim"
            # sorusunun cevabi listenin kendisinde dursun.
            # Oncelige gore sirali; secim numarasi zaten oncelik sirasini
            # veriyor, o yuzden numarayi aciklamaya TEKRAR yazmiyoruz.
            for c in sorted(grup, key=lambda x: (x.priority or 99, x.id)):
                kayitli.append((c.id, c.action))
        yeni = self._sec("ZIHNIYET SEC  ·  sira = uygulama onceligi",
                         kayitli, self.strategy, basliklar, coklu=True)
        if yeni:
            self.strategy = yeni
            print(f"  {ui.GREEN}✓{ui.RESET} zihniyet: {ui.B}{yeni}{ui.RESET}")
        # Yazılmamışları da göster ki "neden yok" sorusu kodda cevaplansın.
        eksik = [(i, lv, t) for i, lv, t in self.planned if i not in set(S.all_ids())]
        if eksik:
            print(f"\n  {ui.DIM}henuz yazilmadi ({len(eksik)}): "
                  f"{', '.join(i for i, _, _ in eksik)}{ui.RESET}")
            print(f"  {ui.DIM}anlatimlari docs/zihniyetler.md icinde{ui.RESET}")

    def sec_senaryo(self) -> None:
        aciklama = {
            "healthy": "temiz ortam",
            "dead_button": "buton tiklaniyor, hata yok, hicbir sey olmuyor",
            "flaky": "iki hata sonra basari — MESRU retry",
            "silent_success": "OK biter ama iki cagri bosa gitmistir",
            "broken_tool": "arac KALICI bozuk, her cagri hata",
        }
        yeni = self._sec("ORTAM SEC", [(s, aciklama.get(s, "")) for s in SCENARIOS],
                         self.scenario)
        if yeni and "," in yeni:
            print(f"  {ui.RED}ortam tek secilir{ui.RESET} "
                  f"{ui.DIM}— birden fazla ortami karsilastirmak icin "
                  f"her birini ayri kosur{ui.RESET}")
        elif yeni:
            self.scenario = yeni
            print(f"  {ui.GREEN}✓{ui.RESET} ortam: {ui.B}{yeni}{ui.RESET}")

    def sec_model(self) -> None:
        aciklama = {
            "stubborn": "ayni noktaya tiklar — klasik dongu",
            "alternating": "iki nokta arasi gidip gelir — A-B-A-B",
            "solver": "gorevi cozer — KONTROL modeli",
            "patient": "hata alinca tekrar dener, sonra bitirir",
            "liar": "yapmadan 'bitirdim' der — verify-gate kontrolu",
            "hf": "GERCEK LLM (HF Inference, internet + token gerekir)",
        }
        yeni = self._sec("MODEL SEC", [(m, aciklama.get(m, "")) for m in sorted(self.models)],
                         self.model)
        if yeni and "," in yeni:
            print(f"  {ui.RED}model tek secilir{ui.RESET}")
        elif yeni:
            self.model = yeni
            print(f"  {ui.GREEN}✓{ui.RESET} model: {ui.B}{yeni}{ui.RESET}")
            if yeni == "hf":
                print(f"  {ui.YELLOW}!{ui.RESET} {ui.DIM}gercek API cagrisi yapilacak. "
                      f"'none' zihniyetiyle birlikte kullanma — butce kancasi yok.{ui.RESET}")

    def ayarla_butce(self) -> None:
        l = self.limits
        alanlar = [("max_steps", l.max_steps), ("max_replans", l.max_replans),
                   ("max_tokens", l.max_tokens), ("max_seconds", l.max_seconds),
                   ("max_cost_usd", l.max_cost_usd)]
        print(f"\n  {ui.B}BUTCE — bes eksen{ui.RESET}  {ui.DIM}(bos birak = degistirme, "
              f"'-' = o ekseni kapat){ui.RESET}")
        print(ui.rule())
        yeni = {}
        for ad, simdi in alanlar:
            try:
                v = input(f"  {ad:<14}{ui.GREY}[{simdi}]{ui.RESET} ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            if not v:
                continue
            if v == "-":
                yeni[ad] = None
            else:
                try:
                    yeni[ad] = float(v) if "." in v or ad in ("max_seconds", "max_cost_usd") else int(v)
                except ValueError:
                    print(f"  {ui.RED}sayi degil: {v}{ui.RESET}"); return
        if yeni:
            self.limits = BudgetLimits(**{**{a: b for a, b in alanlar}, **yeni})
            print(f"  {ui.GREEN}✓{ui.RESET} butce guncellendi")

    def bilgi(self) -> None:
        from .strategies.kinds import KATEGORI
        for sid in self.strategy.split(","):
            ad = sid.split(":")[0].strip()
            varyant = sid.split(":")[1].strip() if ":" in sid else ""
            k = next((c for c in S.catalog() if c.id == ad), None)
            if not k:
                continue
            kat = KATEGORI.get(getattr(k, "kind", "-"), ("-", ""))[0]
            oncelik = f"oncelik {k.priority}" if k.priority else "olcum araci"
            print(f"\n  {ui.B}{k.id}{ui.RESET}{ui.DIM}"
                  f"{':' + varyant if varyant else ''}{ui.RESET} — {k.title}")
            print(f"  {ui.GREY}{kat} · {AILE.get(k.family, k.family)} · "
                  f"{oncelik}{ui.RESET}")
            print(ui.kv("kaynak", k.source, 11))
            print()
            self._blok("TEK CUMLEDE", k.mentality, ui.CYAN)
            self._blok("NEDEN ONEMLI", k.why, ui.GREEN)
            self._blok("TIPIK AKSIYON", k.action, ui.YELLOW)
            self._blok("NEYI KACIRIR", k.blind_spot, ui.RED)
            if k.variants:
                print(f"  {ui.MAGENTA}{'VARYANTLAR':<15}{ui.RESET}", end="")
                print(f"{ui.DIM}{', '.join(k.variants)}"
                      f"   (--strategy {k.id}:<varyant>){ui.RESET}")
            # Docstring'deki ZİHNİYET satırını göster — her strateji dosyası
            # kaynağın ne düşündüğünü orada tek cümleyle yazıyor.
            # ZİHNİYET cümlesi MODUL docstring'inde duruyor (sinif degil):
            # her strateji dosyasi kaynagin ne dusundugunu basta yaziyor.
            import sys as _s
            modul = _s.modules.get(k.__module__)
            print()
            doc = ((getattr(modul, "__doc__", None) or k.__doc__) or "").splitlines()
            for i, s in enumerate(doc):
                if s.strip().upper().startswith(("ZIHNIYET", "ZİHNİYET")):
                    parca = [s.strip()]
                    for devam in doc[i + 1:]:
                        if not devam.strip():
                            break
                        parca.append(devam.strip())
                    metin = " ".join(parca)
                    for satir in _sar(metin, ui.width() - 6):
                        print(f"  {ui.DIM}{satir}{ui.RESET}")
                    break
        print(f"\n  {ui.DIM}tam anlatim: docs/zihniyetler.md{ui.RESET}\n")

    # ------------------------------------------------------------------ koşum

    def _ilerleme(self, faz: str, ctx) -> None:
        b = ctx.budget.state
        ui.status_line(
            f"{ui.CYAN}▸{ui.RESET} adim {ui.B}{b.steps}{ui.RESET} "
            f"{ui.GREY}·{ui.RESET} {b.tokens} tok "
            f"{ui.GREY}·{ui.RESET} ${b.cost_usd:.4f} "
            f"{ui.GREY}·{ui.RESET} {ui.DIM}{faz}{ui.RESET} "
            f"{ui.bar(b.steps, self.limits.max_steps)}")

    def kos(self, gorev: str, strateji: str | None = None, sessiz: bool = False):
        spec = strateji or self.strategy
        iz = self.trace_dir / f"{spec.replace(',', '+')}.jsonl" if self.trace_dir else None
        r = Runner(gorev, FakeSandbox(self.scenario), self.models[self.model](),
                   S.get(spec), limits=self.limits, trace_path=iz,
                   progress=None if sessiz else self._ilerleme).run()
        ui.clear_line()
        return r, iz

    def sonuc(self, r, iz) -> None:
        renk = ui.STATUS_COLOR.get(r.status.value, "")
        t = r.totals
        print()
        print(f"  {renk}{ui.B}{r.status.value}{ui.RESET}  {ui.GREY}·{ui.RESET}  "
              f"{ui.DIM}{r.reason}{ui.RESET}")
        print(ui.rule())
        print(ui.kv("adim", str(t["steps"]), 11) +
              f"   {ui.GREY}token{ui.RESET} {t['tokens']}"
              f"   {ui.GREY}maliyet{ui.RESET} ${t['cost_usd']:.4f}"
              f"   {ui.GREY}sure{ui.RESET} {t['seconds']}sn")
        if r.nudges:
            for n in r.nudges:
                print(f"  {ui.YELLOW}!{ui.RESET} {ui.DIM}{n}{ui.RESET}")
        if r.answer:
            print(ui.kv("cevap", r.answer, 11, ui.GREEN))
        if r.report:
            print(f"\n  {ui.GREY}durma raporu{ui.RESET}")
            print(r.report.render())
        if iz:
            print(f"\n  {ui.GREY}iz{ui.RESET}  {iz}")
        print()

    def karsilastir(self, gorev: str) -> None:
        print(f"\n  {ui.B}KARSILASTIRMA{ui.RESET}  {ui.DIM}{self.scenario} / "
              f"{self.model} / \"{gorev}\"{ui.RESET}")
        print(ui.rule())
        print(f"  {ui.GREY}{'zihniyet':<20}{'durum':<19}{'sebep':<24}"
              f"{'adim':>5}{'token':>8}{'$':>9}{ui.RESET}")
        rows = []
        for sid in S.all_ids():
            ui.status_line(f"{ui.DIM}kosuyor: {sid}{ui.RESET}")
            r, _ = self.kos(gorev, sid, sessiz=True)
            rows.append((sid, r))
            t = r.totals
            renk = ui.STATUS_COLOR.get(r.status.value, "")
            print(f"  {sid:<20}{renk}{r.status.value:<19}{ui.RESET}"
                  f"{ui.DIM}{r.reason[:23]:<24}{ui.RESET}"
                  f"{t['steps']:>5}{t['tokens']:>8}{t['cost_usd']:>9.3f}")
        ui.clear_line()
        # Cakisan zihniyetleri sessizce gecme — iki ayri satir gormek,
        # iki ayri zihniyet gormek demek degil.
        from .cli import _cakismalari_bildir
        _cakismalari_bildir(rows)
        print()

    # ------------------------------------------------------------------ döngü

    KOMUTLAR = {"--strategy", "--scenario", "--model", "--budget", "--info",
                "--compare", "--trace", "--status", "--help", "--quit"}

    def _tamamla(self, metin, durum):
        adaylar = [k for k in sorted(self.KOMUTLAR) if k.startswith(metin)]
        return adaylar[durum] if durum < len(adaylar) else None

    def calistir(self) -> int:
        if readline:
            readline.set_completer(self._tamamla)
            readline.parse_and_bind("tab: complete")
        self.banner()
        self.durum()
        print(f"  {ui.DIM}komutlar icin {ui.CYAN}--help{ui.RESET}{ui.DIM} · "
              f"gorev yazip Enter'a bas{ui.RESET}\n")

        while True:
            try:
                satir = input(f"{ui.CYAN}cua ▸{ui.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {ui.DIM}gorusuruz{ui.RESET}\n")
                return 0
            if not satir:
                continue

            if satir.startswith("--"):
                k = satir.split()[0]
                if k in ("--quit", "--exit"):
                    print(f"  {ui.DIM}gorusuruz{ui.RESET}\n"); return 0
                if k == "--help":
                    self.yardim()
                elif k == "--status":
                    self.durum()
                elif k == "--strategy":
                    self.sec_strateji(); self.durum()
                elif k == "--scenario":
                    self.sec_senaryo()
                elif k == "--model":
                    self.sec_model()
                elif k == "--budget":
                    self.ayarla_butce(); self.durum()
                elif k == "--info":
                    self.bilgi()
                elif k == "--trace":
                    if self.trace_dir:
                        self.trace_dir = None
                        print(f"  {ui.GREEN}✓{ui.RESET} iz kapatildi")
                    else:
                        self.trace_dir = Path("izler"); self.trace_dir.mkdir(exist_ok=True)
                        print(f"  {ui.GREEN}✓{ui.RESET} izler {self.trace_dir}/ altina yazilacak")
                elif k == "--compare":
                    self.karsilastir(self.task)
                else:
                    print(f"  {ui.RED}bilinmeyen komut: {k}{ui.RESET}  "
                          f"{ui.DIM}--help{ui.RESET}")
                continue

            # Serbest metin = GOREV. Sohbet degil.
            if satir.endswith("?") or satir.lower().startswith(
                    ("sen ", "sana ", "kim ", "nedir", "ne ", "nasil", "neden")):
                print(f"  {ui.YELLOW}!{ui.RESET} Bu bir sohbet arayuzu degil. "
                      f"Yazdigin metin AJANA VERILEN GOREV olarak kosuyor.")
                print(f"  {ui.DIM}  ekran: {self.ekran_ozeti()}{ui.RESET}")
                print(f"  {ui.DIM}  ornek gorev: 'Ad alanina Altan yaz ve Gonder "
                      f"butonuna bas'{ui.RESET}")
                onay = input(f"  yine de gorev olarak kosayim mi? [e/H] ").strip().lower()
                if onay not in ("e", "evet", "y", "yes"):
                    continue
            if self.model != "hf" and not self._gorev_uyarisi:
                self._gorev_uyarisi = True
                print(f"  {ui.YELLOW}!{ui.RESET} {ui.DIM}model '{self.model}' BETIKLI — "
                      f"gorev metnini okumuyor, ekrana tepki veriyor. Metnin sonucu "
                      f"degistirmesi icin --model ile 'hf' sec.{ui.RESET}")
            self.task = satir
            try:
                r, iz = self.kos(satir)
            except S.UnknownStrategy as e:
                ui.clear_line()
                print(f"  {ui.RED}'{e.id}' kayitli degil.{ui.RESET} "
                      f"{ui.DIM}secilebilenler: {', '.join(e.mevcut)}{ui.RESET}")
                continue
            except KeyboardInterrupt:
                ui.clear_line()
                print(f"  {ui.YELLOW}kesildi{ui.RESET}")
                continue
            self.sonuc(r, iz)
            self.gecmis.append((satir, self.strategy, r))
