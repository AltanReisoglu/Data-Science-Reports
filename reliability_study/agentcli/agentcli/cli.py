"""
agentcli — iki mod.

    case   onceden tanimli bir senaryoyu secilen zihniyetle kosur
    chat   serbest gorev; her tur ekranda arac cagrisi + ara cikti

Ortak: ayni ajan, ayni uc arac, ayni 17 zihniyet, ayni VLM. Fark sadece
gorevin nereden geldigi.
"""

from __future__ import annotations

import argparse
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
from .render import Rapor                                # noqa: E402
from .tools.browser import Browser                       # noqa: E402
from .tools.terminal import Terminal                     # noqa: E402


def _limits(a) -> BudgetLimits:
    return BudgetLimits(max_steps=a.max_steps, max_replans=a.max_replans,
                        max_tokens=a.max_tokens, max_seconds=a.max_seconds,
                        max_cost_usd=a.max_cost)


def _ortam(a):
    # Kabukla AYNI varsayilan: gorunur ve kalici dizin.
    kok = a.workdir or str(Path.cwd() / "agentcli-calisma")
    import os as _os
    _os.makedirs(kok, exist_ok=True)
    br = Browser(port=a.port, headless=not a.gorunur)
    br.start()
    return br, Terminal(kok), kok


def _kos(a, gorev: str, url: str | None, strateji: str, mod: str):
    br, term, kok = _ortam(a)
    try:
        model = VLMModel(a.model, gorsel=not a.gorselsiz)
        rapor = Rapor()
        rapor.acilis(gorev, strateji, model.ad, mod)
        if url:
            br.goto(url)
        ajan = Ajan(gorev, model, br, term, strateji, limits=_limits(a),
                    rapor=rapor, gorsel=not a.gorselsiz)
        res = ajan.kos()
        rapor.kapanis(res)
        if term.engellenen:
            print(f"  {T.RED}engellenen komutlar{T.RESET}")
            for e in term.engellenen:
                print(f"    {T.DIM}{e}{T.RESET}")
            print()
        print(f"  {T.DIM}çalışma dizini: {kok}{T.RESET}\n")
        return res
    finally:
        br.stop()


# ------------------------------------------------------------------ komutlar

def cmd_cases(a) -> int:
    print()
    grup = None
    for c in listele():
        if c.grup != grup:
            grup = c.grup
            baslik = ("YAKALAMA — guardrail devreye girmeli" if grup == "yakalama"
                      else "KONTROL — guardrail SUSMALI, meşru koşum")
            print(f"  {T.B}{baslik}{T.RESET}")
            print(f"  {T.cizgi()}")
        print(f"  {T.BLUE}{c.ad:<16}{T.RESET}{c.gorev[:60]}")
        for s in T.sar(c.anlat, T.en() - 22):
            print(f"  {'':<16}{T.DIM}{s}{T.RESET}")
        print(f"  {'':<16}{T.GREEN}beklenen:{T.RESET} {T.DIM}{c.bekleniyor}{T.RESET}\n")
    return 0


def cmd_strategies(a) -> int:
    print()
    for k, (ad, ozet) in KATEGORI.items():
        grup = [c for c in S.catalog() if getattr(c, "kind", "-") == k]
        if not grup:
            continue
        print(f"  {T.B}{ad}{T.RESET}  {T.DIM}— {ozet}{T.RESET}")
        for c in sorted(grup, key=lambda x: (x.priority or 99, x.id)):
            on = f"{c.priority:>2}" if c.priority else " -"
            print(f"  {T.DIM}{on}{T.RESET} {T.BLUE}{c.id:<21}{T.RESET}{c.action[:56]}")
        print()
    print(f"  {T.DIM}ayrıntı: reliability_study/cua_lab/docs/zihniyetler.md{T.RESET}\n")
    return 0


def cmd_case(a) -> int:
    if a.name not in CASES:
        print(f"\n  {T.RED}'{a.name}' diye bir case yok.{T.RESET}")
        print(f"  {T.DIM}liste: agentcli cases{T.RESET}\n")
        return 2
    c = CASES[a.name]
    print(f"\n  {T.DIM}case{T.RESET} {T.B}{c.ad}{T.RESET} {T.DIM}({c.grup}){T.RESET}")
    for s in T.sar(c.anlat, T.en() - 4):
        print(f"  {T.DIM}{s}{T.RESET}")
    res = _kos(a, c.gorev, c.url, a.strategy, f"case: {c.ad}")
    return 0 if res.status.clean else 1


def cmd_kategori(a) -> int:
    """Bir zihniyet ailesinin tamamını aynı case'e karşı GERÇEKTEN koştur."""
    from . import kategori as K
    if a.kind not in K.KATEGORILER:
        print(f"\n  {T.RED}'{a.kind}' diye bir aile yok.{T.RESET}  "
              f"{T.DIM}{', '.join(K.KATEGORILER)}{T.RESET}\n")
        return 2
    if a.name not in CASES:
        oner = [c.ad for c in CASES.values() if c.kategori == a.kind]
        print(f"\n  {T.RED}'{a.name}' diye bir case yok.{T.RESET}  "
              f"{T.DIM}bu aile icin: {', '.join(oner) or '—'}{T.RESET}\n")
        return 2
    c = CASES[a.name]
    br, term, kok = _ortam(a)
    try:
        model = VLMModel(a.model, gorsel=not a.gorselsiz)
        print(f"\n  {T.DIM}beklenen:{T.RESET} {c.bekleniyor}")

        def kur(sid, gorev, url, _):
            br.goto(url)                 # her zihniyet TEMIZ sayfayla baslasin
            return Ajan(gorev, model, br, term, sid, limits=_limits(a),
                        rapor=Rapor(sessiz=True), gorsel=not a.gorselsiz,
                        golge=False).kos()

        satirlar = K.kos(a.kind, c.gorev, c.url, kur, ekstra=a.ekstra)
        K.tablo(satirlar, a.kind, c.gorev)
        print(f"  {T.DIM}calisma dizini: {kok}{T.RESET}\n")
        return 0
    finally:
        br.stop()


def cmd_chat(a) -> int:
    """Serbest görev. Her tur araç çağrısı ve ara çıktı ekrana basılıyor."""
    br, term, kok = _ortam(a)
    try:
        model = VLMModel(a.model, gorsel=not a.gorselsiz)
        print(f"\n  {T.B}agentcli{T.RESET} {T.DIM}sohbet modu · zihniyet "
              f"{a.strategy} · model {model.ad}{T.RESET}")
        print(f"  {T.DIM}çalışma dizini {kok} · çıkmak için Ctrl-D{T.RESET}")
        print(f"  {T.cizgi()}")
        while True:
            try:
                gorev = input(f"\n{T.BLUE}▸{T.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {T.DIM}görüşürüz{T.RESET}\n"); return 0
            if not gorev:
                continue
            if gorev in ("/quit", "/exit"):
                return 0
            if gorev.startswith("/strategy"):
                yeni = gorev.split(maxsplit=1)
                if len(yeni) > 1:
                    a.strategy = yeni[1].strip()
                    print(f"  {T.GREEN}✓{T.RESET} zihniyet: {a.strategy}")
                continue
            if gorev.startswith("/url"):
                p = gorev.split(maxsplit=1)
                if len(p) > 1:
                    print(f"  {T.DIM}{br.goto(p[1].strip())}{T.RESET}")
                continue
            rapor = Rapor()
            rapor.acilis(gorev, a.strategy, model.ad, "sohbet")
            ajan = Ajan(gorev, model, br, term, a.strategy, limits=_limits(a),
                        rapor=rapor, gorsel=not a.gorselsiz)
            rapor.kapanis(ajan.kos())
    finally:
        br.stop()


# ------------------------------------------------------------------ parser

def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentcli",
                                description="VLM computer-use ajani + secilebilir guardrail zihniyeti")
    sub = p.add_subparsers(dest="cmd")

    def ortak(sp):
        sp.add_argument("--strategy", default="openhands-stuck")
        sp.add_argument("--model", default=VLM)
        sp.add_argument("--workdir", default=None, help="terminal calisma dizini")
        sp.add_argument("--port", type=int, default=9222)
        sp.add_argument("--gorunur", action="store_true", help="tarayiciyi goster")
        sp.add_argument("--gorselsiz", action="store_true",
                        help="ekran goruntusu GONDERME (yalniz DOM metni)")
        sp.add_argument("--max-steps", type=int, default=14)
        sp.add_argument("--max-replans", type=int, default=5)
        sp.add_argument("--max-tokens", type=int, default=60000)
        sp.add_argument("--max-seconds", type=float, default=240.0)
        sp.add_argument("--max-cost", type=float, default=0.5)

    sp = sub.add_parser("cases", help="hazir case'leri listele"); sp.set_defaults(fn=cmd_cases)
    sp = sub.add_parser("strategies", help="zihniyetleri listele"); sp.set_defaults(fn=cmd_strategies)
    sp = sub.add_parser("case", help="hazir bir case'i kosur"); ortak(sp)
    sp.add_argument("name"); sp.set_defaults(fn=cmd_case)
    sp = sub.add_parser("kategori", help="bir zihniyet ailesinin TAMAMINI kosur")
    ortak(sp)
    sp.add_argument("kind", help="budget|window|evidence|shape|decision")
    sp.add_argument("name", help="case adi")
    sp.add_argument("--ekstra", default="", help="listeye eklenecek ek zihniyetler")
    sp.set_defaults(fn=cmd_kategori)
    sp = sub.add_parser("chat", help="serbest gorev, ara ciktilarla"); ortak(sp)
    sp.set_defaults(fn=cmd_chat)
    return p


def main(argv=None) -> int:
    a = build().parse_args(argv)
    if not getattr(a, "fn", None):
        build().print_help(); return 0
    try:
        return a.fn(a)
    except S.UnknownStrategy as e:
        print(f"\n  {T.RED}'{e.id}' zihniyeti yok.{T.RESET}")
        print(f"  {T.DIM}mevcut: {', '.join(e.mevcut)}{T.RESET}\n")
        return 2
    except KeyboardInterrupt:
        print(f"\n  {T.DIM}kesildi{T.RESET}\n"); return 130


if __name__ == "__main__":
    sys.exit(main())
