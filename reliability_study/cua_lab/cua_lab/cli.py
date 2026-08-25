"""
cua-lab CLI — stdlib argparse, ek bağımlılık yok.

    cua-lab list-strategies
    cua-lab run --task "..." --strategy openhands-stuck --scenario dead_button
    cua-lab compare --scenario dead_button --strategies all
    cua-lab replay <trace.jsonl>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import strategies as S
from .events import Act, BudgetLimits, ComputerCall, Finish
from .loop import Runner, Status
from .model import AlternatingModel, ScriptedModel, StubbornModel
from .sandbox.fake import SCENARIOS, FakeSandbox
from .trace import read_trace

# Belgelenmis 17 zihniyet, zorluk sirasina gore (docs/zihniyetler.md ile ayni numara).
# Kayit defterinde OLMAYANLAR henuz yazilmadi; CLI bunu traceback yerine
# duzgun bir mesajla soylemek icin bu tabloyu kullaniyor.
PLANNED: list[tuple[str, str, str]] = [
    ("arize-control",        "1 Sayac",   "Modele sorma, say"),
    ("budget-grace",         "1 Sayac",   "Lutuf butcesi (varyant: agentscope | hermes)"),
    ("claude-advisory",      "1 Sayac",   "Modele saatini goster"),
    ("strands-entropy",      "2 Pencere", "Tekrari sayma, cesitliligi olc"),
    ("openhands-stuck",      "2 Pencere", "Bes desen, STUCK ayri terminal durum"),
    ("openclaw-pingpong",    "2 Pencere", "Adlandirilmis dedektorler + sikistirma tuzagi"),
    ("loopguard-dignity",    "2 Pencere", "Hareket ilerleme degildir"),
    ("verify-gate",          "3 Dunya",   "'Bitirdim' bir istektir, kanit degil"),
    ("galileo-breaker",      "3 Dunya",   "Suc ajanda degil, aracta"),
    ("agentbudget-dollar",   "3 Dunya",   "Tokeni degil, dolari say"),
    ("modexa-statemachine",  "4 Sekil",   "Serbestligi kisitla"),
    ("autogen-static",       "4 Sekil",   "Kosum baslamadan yakala"),
    ("telemetry-repair",     "5 Kademe",  "Kural koy, ihlali soyle, geri sar"),
    ("pi-signature",         "5 Kademe",  "Alti ucuz sinyal, kademeli mudahale"),
    ("voi-allocation",       "6 Karar",   "Butce bir tavan degil, bir butce"),
    ("improvement-loop",     "6 Karar",   "Esigi tahmin etme, olc"),
]


def _solver(req):
    """Alan -> yaz -> gonder -> bitir. Ekranda 'gonderildi' gorunce durur."""
    scr = req.screen
    if "gonderildi" in scr:
        return Finish("Form gonderildi.", tokens=150, cost_usd=0.0015)
    if 'Ad=""' in scr:
        if "Ad odaklandi" in scr:
            return ComputerCall(Act.TYPE, {"text": "Altan"}, tokens=350, cost_usd=0.0035)
        return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 120}, tokens=350, cost_usd=0.0035)
    return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}, tokens=350, cost_usd=0.0035)


def _patient(req):
    """Solver ile ayni ama gecici hatalara karsi sabirli — mesru retry."""
    tries = sum(1 for h in req.history if "gecici hata" in h)
    if tries >= 4:
        return Finish("Arac surekli hata verdi, kismi sonuc.", tokens=150, cost_usd=0.0015)
    return _solver(req)


def _hf_model():
    """Gercek LLM. Import BURADA yapiliyor — `hf` secilmedikce ag koduna
    hic dokunulmuyor, PoC'nin cevrimdisi calismasi bozulmasin diye."""
    from .hf_model import DEFAULT_MODEL, HFInferenceModel
    import os
    return HFInferenceModel(os.environ.get("CUA_HF_MODEL", DEFAULT_MODEL))


def _liar(req):
    """Hicbir sey yapmadan "bitirdim" diyen model.

    verify-gate'in varlik sebebi: modelin "bitti" demesi bir BILGI degil bir
    TALEPTIR. `none` bu kosumu OK sayar ve kullaniciya yapilmamis bir isi
    teslim eder — sessiz basarisizligin en saf hali.
    """
    if "gonderildi" in req.screen:
        return Finish("Form gonderildi.", tokens=150, cost_usd=0.0015)
    return Finish("Formu doldurup gonderdim.", tokens=200, cost_usd=0.002)


MODELS = {
    "stubborn": lambda: StubbornModel(Act.LEFT_CLICK, {"x": 200, "y": 200}),
    "alternating": lambda: AlternatingModel(
        ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}),
        ComputerCall(Act.LEFT_CLICK, {"x": 320, "y": 200}),
    ),
    # Gorevi cozup BITIREN model — kontrol senaryosu.
    # Hicbir strateji bunda tetiklenmemeli. Bir dedektorun ikinci sinavi
    # yakalamamasi gerekeni rahat birakmaktir.
    "solver": lambda: ScriptedModel(_solver),
    # Flaky araca karsi makul davranan model: hata alinca tekrar dener ama
    # bir noktada vazgecip bitirir. Mesru retry — dongu degil.
    "patient": lambda: ScriptedModel(_patient),
    # Gorevi yapmadan "bitirdim" diyen model — verify-gate'in kontrolu.
    "liar": lambda: ScriptedModel(_liar),
    # GERCEK LLM — HF Inference API. Tek internet gerektiren model.
    # Ayri modulde tutuluyor ki digerleri agsiz calismaya devam etsin.
    "hf": _hf_model,
}


def _limits(a) -> BudgetLimits:
    return BudgetLimits(
        max_steps=a.max_steps, max_replans=a.max_replans,
        max_tokens=a.max_tokens, max_seconds=a.max_seconds,
        max_cost_usd=a.max_cost,
    )


def _sar(metin: str, en: int) -> list[str]:
    """Basit sözcük sarma — tek amaçlı, textwrap'e bağımlılık kurmadan."""
    out, satir = [], ""
    for k in (metin or "").split():
        if len(satir) + len(k) + 1 > en:
            out.append(satir); satir = k
        else:
            satir = f"{satir} {k}".strip()
    if satir:
        out.append(satir)
    return out or [""]


def cmd_list(a) -> int:
    """Kategoriye gore gruplu liste.

    Gruplama kozmetik degil: ayni kategorideki zihniyetler ayni ORTAK MANTIGI
    (kinds/*.py) paylasiyor ve yalnizca bir kararda ayrisiyorlar. Listenin bunu
    gostermesi, "17 ayri sey" yerine "5 mekanizma x farkli kararlar" resmini
    veriyor.
    """
    from .strategies.kinds import KATEGORI
    fam = {"src": "ben_ekledim", "harness": "harness", "-": "taban"}
    kat = S.catalog()
    for k, (ad, ozet) in KATEGORI.items():
        grup = [c for c in kat if getattr(c, "kind", "-") == k]
        if not grup:
            continue
        print(f"\n  {ad}  —  {ozet}   [{len(grup)}]")
        print("  " + "-" * 96)
        for c in sorted(grup, key=lambda x: (x.priority or 99, x.id)):
            on = f"{c.priority:>2}" if c.priority else " -"
            v = f"  varyant: {'|'.join(c.variants)}" if c.variants else ""
            print(f"  {on}  {c.id:<20}{fam.get(c.family, c.family):<13}{c.mentality}{v}")
            for etiket, metin in (("neden", c.why), ("aksiyon", c.action),
                                  ("kacirdigi", c.blind_spot)):
                for i, satir in enumerate(_sar(metin, 60)):
                    on = etiket if i == 0 else ""
                    print(f"      {'':<20}{on:<13}{satir}")
    yazili = set(S.all_ids())
    eksik = [(i, lv, ttl) for i, lv, ttl in PLANNED if i not in yazili]
    if eksik:
        print(f"\n  HENUZ YAZILMADI ({len(eksik)})")
        for i, lv, ttl in eksik:
            print(f"  {i:<20}{'sv ' + lv:<14}{ttl}")
    zihniyet = yazili - {"none"}
    print(f"\n  ilk kolon = UYGULAMA ONCELIGI (1 = once bunu koy)")
    print(f"  kayitli {len(zihniyet)} / belgelenmis {len(PLANNED)} zihniyet"
          f" (+ `none` taban cizgisi)"
          f"   ·   ortak mantik: cua_lab/strategies/kinds/")
    print("  anlatim: docs/zihniyetler.md\n")
    return 0


def _sandbox(a):
    """Sentetik mi gercek masaustu mu.

    `--scenario desktop` GERCEK ekrani kullaniyor. Girdi VARSAYILAN KAPALI —
    `--allow-input` acikca verilmeden tek bir fare/klavye olayi gonderilmiyor.
    """
    if a.scenario != "desktop":
        return FakeSandbox(a.scenario), None
    from .safety import SafetyPolicy
    from .sandbox.x11 import X11Sandbox
    from .watch import Watcher
    pol = SafetyPolicy(
        allow_input=getattr(a, "allow_input", False),
        allow_window_close=False,          # silme/kapatma yetkisi YOK
        max_real_actions=getattr(a, "max_real_actions", 60),
        dwell_seconds=getattr(a, "dwell", 0.35),
    )
    hud = None
    hud_w = 0
    if getattr(a, "hud", True):
        from .hud import HUD_WIDTH, Hud
        gecici = X11Sandbox(display=getattr(a, "display", None), policy=pol)
        gecici.start()
        hud_w = HUD_WIDTH
        hud = Hud(gecici.width, gecici.height, width=hud_w)
        print(f"  panel: {hud.url}  (ekranin saginda)")
    w = Watcher(hud=hud)
    sb = X11Sandbox(display=getattr(a, "display", None), policy=pol,
                    gozlemci=w, hud_width=hud_w)
    return sb, w


def _run_one(a, strategy_spec: str):
    sandbox, _w = _sandbox(a)
    # HUD varsa bütçe/sonuç da panele aksın. Döngünün `progress` kancası
    # zaten her adımda çağrılıyor — ayrı bir zamanlayıcıya gerek yok.
    ilerleme = None
    if _w is not None and getattr(_w, "hud", None) is not None:
        def ilerleme(faz, ctx, _w=_w, _s=strategy_spec, _g=a.task):
            _w.butce(ctx, _s, _g)
    model = MODELS[a.model]()
    stack = S.get(strategy_spec)
    trace = Path(a.trace) if a.trace else None
    res = Runner(a.task, sandbox, model, stack, limits=_limits(a),
                 trace_path=trace, progress=ilerleme).run()
    if _w is not None:
        _w.bitir(res)
    return res


def cmd_run(a) -> int:
    res = _run_one(a, a.strategy)
    t = res.totals
    print(f"\n  gorev     : {a.task}")
    print(f"  senaryo   : {a.scenario}   model: {a.model}   strateji: {a.strategy}")
    print(f"  {'-'*70}")
    print(f"  DURUM     : {res.status.value}")
    print(f"  sebep     : {res.reason}")
    print(f"  adim      : {t['steps']}   token: {t['tokens']}   ${t['cost_usd']}   {t['seconds']}sn")
    if res.nudges:
        print(f"  uyarilar  : {', '.join(res.nudges)}")
    if res.answer:
        print(f"  cevap     : {res.answer}")
    if res.report:
        print("  rapor:")
        print(res.report.render())
    if a.trace:
        print(f"  iz        : {a.trace}")
    print()
    return 0 if res.status.clean else 1


def cmd_compare(a) -> int:
    specs = S.all_ids() if a.strategies == "all" else [s.strip() for s in a.strategies.split(",")]
    print(f"\n  senaryo: {a.scenario}   model: {a.model}   gorev: {a.task}")
    print(f"\n  {'strateji':<20}{'durum':<20}{'sebep':<26}{'adim':>5}{'token':>8}{'$':>8}")
    print("  " + "-" * 87)
    rows = []
    for spec in specs:
        res = _run_one(a, spec)
        t = res.totals
        rows.append((spec, res))
        print(f"  {spec:<20}{res.status.value:<20}{res.reason[:25]:<26}"
              f"{t['steps']:>5}{t['tokens']:>8}{t['cost_usd']:>8.3f}")
    print("  " + "-" * 87)
    base = next((r for s, r in rows if s == "none"), None)
    if base and base.totals["cost_usd"] > 0:
        best = min((r.totals["cost_usd"] for s, r in rows if s != "none"), default=0)
        if best > 0:
            print(f"  kontrolsuz kosum en iyi stratejiye gore {base.totals['cost_usd']/best:.1f}x pahali")
    _cakismalari_bildir(rows)
    print()
    return 0


# Ayirt edilemeyen stratejiler icin ipuclari — "neden ayni ciktilar" sorusunun
# cevabi kaynagin degil KOSUMUN ozelligi oldugunda bunu soylemek gerekiyor.
AYIRICI = {
    "improvement-loop": "bu zihniyet MUDAHALE ETMEZ; farki `rapor` alanindaki "
                        "olculmus esik onerisinde — `run` ile bak",
    "autogen-static": "yapisal kontrol; ancak HICBIR butce ekseni acik degilken "
                      "ayrisir — `--max-steps -` gibi hepsini kapat",
    "voi-allocation": "durdurmaz, eylem secimine karisir; farki `uyarilar` "
                      "satirinda gorunur",
    "telemetry-repair": "bitirme iddiasi bekler; `--model liar` ile ayrisir",
    "verify-gate": "bitirme iddiasi bekler; `--model liar` ile ayrisir",
    "galileo-breaker": "hata ORANINA bakar; `--scenario broken_tool` ile ayrisir",
    "none": "taban cizgisi — hicbir kontrol yok",
}


def _cakismalari_bildir(rows) -> None:
    """Bu kosumda AYIRT EDILEMEYEN stratejileri raporla.

    Iki strateji ayni ciktiyi veriyorsa bunu sessizce gecmek yaniltici olur:
    kullanici iki ayri "zihniyet" gordugunu sanir. Ama cakisma her zaman
    "ayni sey" demek de degil — bazen kosumun onlari ayiracak kosulu
    icermemesi demektir. Ikisini ayirmak icin ipucu basiliyor.
    """
    import collections
    kume = collections.defaultdict(list)
    for spec, r in rows:
        kume[(r.status.value, r.reason, r.totals["steps"],
              r.totals["tokens"])].append(spec)
    cakisan = [g for g in kume.values() if len(g) > 1]
    if not cakisan:
        return
    print(f"\n  BU KOSUMDA AYIRT EDILEMEYENLER ({len(cakisan)} kume) — "
          f"ayni durum/sebep/adim/token")
    for grup in cakisan:
        print(f"    {' = '.join(grup)}")
        for s in grup:
            if s in AYIRICI:
                print(f"      {s}: {AYIRICI[s]}")


def cmd_replay(a) -> int:
    spans = read_trace(a.path)
    print(f"\n  {len(spans)} adim — {a.path}\n")
    print(f"  {'#':>3} {'eylem':<16}{'ekran':<14}{'karar':<10}{'sebep'}")
    print("  " + "-" * 76)
    for s in spans:
        mark = "  " if s.verdict == "continue" else "!!"
        err = f"  HATA: {s.error[:24]}" if s.error else ""
        print(f"{mark}{s.i:>3} {s.action:<16}{s.screen_hash:<14}{s.verdict:<10}{s.verdict_reason}{err}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cua-lab", description="Computer-use agent + secilebilir guvenilirlik stratejileri")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--task", default="Formu doldur ve gonder")
        sp.add_argument("--scenario", default="dead_button",
                        choices=list(SCENARIOS) + ["desktop"])
        sp.add_argument("--allow-input", action="store_true",
                        help="GERCEK fare/klavye olayi gonder (yalniz --scenario desktop)")
        sp.add_argument("--display", default=None, help="X ekrani, orn :1")
        sp.add_argument("--max-real-actions", type=int, default=60)
        sp.add_argument("--dwell", type=float, default=0.35,
                        help="tiklamadan once hedefte bekleme (sn) — izleyebilesin")
        sp.add_argument("--no-hud", dest="hud", action="store_false",
                        help="masaustunun sagindaki canli izleme panelini acma")
        sp.add_argument("--model", default="stubborn", choices=sorted(MODELS))
        sp.add_argument("--max-steps", type=int, default=12)
        sp.add_argument("--max-replans", type=int, default=5)
        sp.add_argument("--max-tokens", type=int, default=20000)
        sp.add_argument("--max-seconds", type=float, default=30.0)
        sp.add_argument("--max-cost", type=float, default=0.5)
        sp.add_argument("--trace", default=None)

    sp = sub.add_parser("list-strategies", help="stratejileri listele"); sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("run", help="tek kosum"); common(sp)
    sp.add_argument("--strategy", default="none"); sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("compare", help="stratejileri ayni gorevde karsilastir"); common(sp)
    sp.add_argument("--strategies", default="all"); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("replay", help="izi adim adim oynat")
    sp.add_argument("path"); sp.set_defaults(fn=cmd_replay)
    sp = sub.add_parser("doctor", help="gercek masaustu icin hazir miyiz")
    sp.add_argument("--display", default=None); sp.set_defaults(fn=cmd_doctor)
    sp = sub.add_parser("shell", help="interaktif kabuk (varsayilan)")
    sp.set_defaults(fn=cmd_shell)
    return p


def cmd_doctor(a) -> int:
    """Gercek masaustu backend'i icin hazirlik raporu.

    Ekran YAKALAMAZ — yalniz araclarin varligina ve X ortamina bakar.
    """
    import os
    from .sandbox.x11 import Arac
    d = a.display or os.environ.get("DISPLAY", "(yok)")
    ar = Arac.tara()
    print(f"\n  GERCEK MASAUSTU HAZIRLIK RAPORU")
    print("  " + "-" * 66)
    print(f"  {'DISPLAY':<22}{d}")
    print(f"  {'oturum tipi':<22}{os.environ.get('XDG_SESSION_TYPE', '?')}")
    if os.environ.get("WAYLAND_DISPLAY"):
        print(f"  {'':<22}{'UYARI: Wayland — xdotool calismaz'}")
    print()
    for ad, var, ne in (("xdotool", ar.xdotool, "fare/klavye + pencere sorgusu"),
                        ("ffmpeg", ar.ffmpeg, "ekran goruntusu (x11grab)"),
                        ("scrot", ar.scrot, "ekran goruntusu (alternatif)"),
                        ("PIL", ar.pil, "goruntu kucultme (hash icin)")):
        print(f"  {'[+]' if var else '[-]'} {ad:<18}{ne}")
    print()
    okuma = ar.ekran_alinabilir
    girdi = ar.xdotool and okuma
    print(f"  salt okuma (girdisiz)   : {'HAZIR' if okuma else 'HAZIR DEGIL'}")
    print(f"  tam kontrol (girdili)   : {'HAZIR' if girdi else 'HAZIR DEGIL'}")
    eksik = ar.eksikler(True)
    if eksik:
        print(f"\n  eksikler:")
        for e in eksik:
            print(f"    {e}")
    # VLM tarafı
    from .hf_model import DEFAULT_MODEL, read_token
    tok = read_token()
    print(f"\n  {'[+]' if tok else '[-]'} HF token{'':<11}"
          f"{'bulundu' if tok else 'yok — .env icine HF_Token=... koy'}")
    print(f"      {'':<18}VLM: {DEFAULT_MODEL}")
    if tok:
        print(f"\n  {'!':>3} GIZLILIK: `--model hf` ile ekran goruntun HuggingFace'e gider.")
        print(f"      O anda ekranda ne varsa (terminal, parola yoneticisi, ozel")
        print(f"      mesaj) dis bir servise yuklenmis olur. Once ekrani temizle.")
        print(f"      Gorsel token pahali: ~1700 token/adim olculdu.")

    print(f"\n  ONCE SALT OKUMA ile dene (hicbir sey gondermez):")
    print(f"    python3 -m cua_lab.cli run --scenario desktop --strategy arize-control \\")
    print(f"        --model solver --max-steps 6")
    print(f"\n  Girdi acmak icin --allow-input ekle. Kacis: fareyi SOL UST KOSEYE tasi.\n")
    return 0 if okuma else 1


def cmd_shell(a) -> int:
    from .shell import Shell
    return Shell(MODELS, PLANNED).calistir()


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not getattr(a, "fn", None):
        # Argumansiz cagri: interaktif kabuk. Zihniyet bir bayrak degil bir MOD.
        return cmd_shell(a)
    try:
        return a.fn(a)
    except S.UnknownStrategy as e:
        # Traceback basmak yerine, id'nin "hic yok" mu yoksa
        # "belgelendi ama henuz yazilmadi" mi oldugunu soyle.
        istenen = e.id
        kunye = {i: (lv, ttl) for i, lv, ttl in PLANNED}
        print(f"\n  hata: '{istenen}' stratejisi kayitli degil.", file=sys.stderr)
        if istenen in kunye:
            lv, ttl = kunye[istenen]
            print(f"  bu zihniyet BELGELENDI ama henuz yazilmadi — seviye {lv}: {ttl}", file=sys.stderr)
            print(f"  anlatimi: docs/zihniyetler.md", file=sys.stderr)
        print(f"  su an secilebilenler: {', '.join(e.mevcut)}", file=sys.stderr)
        print(f"  tam liste: python3 -m cua_lab.cli list-strategies\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
