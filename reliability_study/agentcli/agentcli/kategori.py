"""
Kategori koşumu — aynı zihniyet ailesinin tamamını AYNI göreve karşı koştur.

NEDEN GÖLGE TABLOSU YETMİYOR: gölge kurulu tek koşumda 17 zihniyeti izletiyor
ama bu KARŞI OLGUSAL — "müdahale etseydi ne olurdu" tahmini, kesişim noktasına
kadar geçerli. Bir zihniyet 3. adımda durdursaydı 4. adım hiç olmayacaktı ve
sonraki her satır anlamını kaybediyor.

Burada her zihniyet GERÇEKTEN koşuyor: kendi kararlarını kendi veriyor, kendi
adımını, kendi token'ını harcıyor. Karşılaştırma tablosu ölçüm oluyor, tahmin
değil. Bedeli: N kat model çağrısı.

`none` her zaman ilk satır. Taban çizgisi olmadan "7 adımda durdu" bir şey
söylemiyor — kontrolsüz koşum da 7 adım sürmüş olabilir.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from . import theme as T

# kind → (merdiven adı, tek cümlede ne yaptığı)
KATEGORILER: dict[str, tuple[str, str]] = {
    "budget":   ("SAYAÇ",   "Sayar ve tavana vurunca durur — ne olduğunu sormaz"),
    "window":   ("PENCERE", "Son N adıma bakar, tekrar/durgunluk arar"),
    "evidence": ("DÜNYA",   "Dünyaya bakar: iddia gerçekten oldu mu"),
    "shape":    ("ŞEKİL",   "Döngünün biçimini kısıtlar — izinsiz geçiş yok"),
    "decision": ("KARAR",   "Bütçeyi tavan değil tahsis sayar"),
}


def uyeler(kind: str) -> list[str]:
    """Bir kategorideki zihniyet id'leri."""
    from cua_lab.strategies.base import catalog
    return sorted(c.id for c in catalog() if getattr(c, "kind", "-") == kind)


def hepsi() -> dict[str, list[str]]:
    return {k: uyeler(k) for k in KATEGORILER}


@dataclass
class Satir:
    sid: str
    durum: str
    sebep: str
    adim: int
    token: int
    saniye: float
    cevap: str
    uyari: int
    hata: str = ""


def tablo(satirlar: list[Satir], kind: str, gorev: str) -> None:
    ad, ne = KATEGORILER.get(kind, (kind, ""))
    print()
    print(f"  {T.B}{ad} AİLESİ — GERÇEK KOŞUM{T.RESET}"
          f"  {T.DIM}{len(satirlar) - 1} zihniyet + taban çizgisi{T.RESET}")
    print(f"  {T.DIM}{ne}{T.RESET}")
    print(f"  {T.DIM}görev: {gorev[:76]}{T.RESET}")
    print(f"  {T.cizgi()}")
    print(f"  {T.DIM}{'zihniyet':<21}{'durum':<19}{'adım':>5}{'token':>8}"
          f"{'süre':>7}  {'sebep'}{T.RESET}")

    taban = next((s for s in satirlar if s.sid == "none"), None)
    for s in satirlar:
        renk = T.DURUM.get(s.durum, T.INK)
        if s.hata:
            print(f"  {s.sid:<21}{T.RED}{'HATA':<19}{T.RESET}"
                  f"{T.DIM}{s.hata[:44]}{T.RESET}")
            continue
        # Taban çizgisinden SAPMA işareti: aynı adımda bitmişse müdahale
        # etmemiş demektir, o satır "sessiz kaldı" diye okunmalı.
        isaret = " "
        if taban and s.sid != "none":
            if s.adim < taban.adim or s.durum != taban.durum:
                isaret = f"{T.AMBER}◆{T.RESET}"
        u = f" {s.uyari}×uyarı" if s.uyari else ""
        print(f"  {s.sid:<21}{renk}{s.durum:<19}{T.RESET}{s.adim:>5}{s.token:>8}"
              f"{s.saniye:>6.1f}s  {T.DIM}{s.sebep[:30]}{u}{T.RESET}{isaret}")

    print(f"  {T.cizgi()}")
    if taban:
        mudahale = [s for s in satirlar
                    if s.sid != "none" and not s.hata
                    and (s.adim < taban.adim or s.durum != taban.durum)]
        sessiz = [s for s in satirlar
                  if s.sid != "none" and not s.hata and s not in mudahale]
        print(f"  {T.AMBER}◆{T.RESET} {T.DIM}taban çizgisinden saptı: "
              f"{', '.join(s.sid for s in mudahale) or '(hiçbiri)'}{T.RESET}")
        print(f"  {T.DIM}  taban çizgisiyle aynı bitti: "
              f"{', '.join(s.sid for s in sessiz) or '(hiçbiri)'}{T.RESET}")
        yakilan = sum(s.token for s in satirlar if not s.hata)
        print(f"  {T.DIM}  toplam {yakilan} token · taban çizgisi "
              f"{taban.adim} adım / {taban.token} token{T.RESET}")
    print()


def kos(kind: str, gorev: str, url: str | None, kur, rapor_ac: bool = False,
        ekstra: str = "") -> list[Satir]:
    """Kategorideki her zihniyeti sırayla koştur.

    `kur(strateji)` → (ajan, kapat) döndüren bir fabrika. Her zihniyet için
    TEMİZ ortam gerekiyor: aynı tarayıcı sayfası devam ederse ikinci zihniyet
    birincinin bıraktığı durumu görür ve karşılaştırma bozulur.
    """
    idler = ["none"] + uyeler(kind)
    if ekstra:
        idler += [e.strip() for e in ekstra.split(",") if e.strip()]
    ad, _ = KATEGORILER.get(kind, (kind, ""))
    satirlar: list[Satir] = []

    for i, sid in enumerate(idler, 1):
        etiket = f"{T.DIM}[{i}/{len(idler)}]{T.RESET} {T.PURP}{sid}{T.RESET}"
        sys.stdout.write(f"\r\033[2K  {ad} · {etiket} koşuyor…")
        sys.stdout.flush()
        t0 = time.monotonic()
        try:
            res = kur(sid, gorev, url, rapor_ac)
        except KeyboardInterrupt:
            sys.stdout.write("\r\033[2K")
            print(f"  {T.AMBER}kesildi — {len(satirlar)} zihniyet koştu{T.RESET}")
            break
        except Exception as e:                      # bir zihniyet patlarsa
            satirlar.append(Satir(sid, "HATA", "", 0, 0, 0.0, "", 0,
                                  hata=f"{type(e).__name__}: {e}"))
            continue
        t = res.totals
        uyari = sum(1 for e in getattr(res, "uyarilar", []) or [])
        satirlar.append(Satir(
            sid=sid, durum=res.status.value, sebep=res.reason or "—",
            adim=int(t.get("steps", 0)), token=int(t.get("tokens", 0)),
            saniye=float(t.get("seconds", time.monotonic() - t0)),
            cevap=(res.answer or "")[:70], uyari=uyari))

    sys.stdout.write("\r\033[2K")
    return satirlar
