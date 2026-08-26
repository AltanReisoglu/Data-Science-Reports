"""
Gölge gözlemciler — tek koşum, bütün zihniyetler izliyor.

PROBLEM: "hangi zihniyet daha iyi" sorusunu cevaplamak için her zihniyetle
ayrı koşum yapmak gerekiyordu. Gerçek VLM ile bu 17× para ve 17× süre demek —
ve koşumlar farklı olduğu için sonuçlar da tam karşılaştırılabilir değil.

ÇÖZÜM: ajanı BİR KEZ koştur. Aynı olay akışını 17 zihniyete birden ver.
Yalnız SEÇİLİ olan gerçekten müdahale etsin; diğerleri sadece "ben burada ne
yapardım" diye kayıt tutsun.

Sonuç: aynı koşum üzerinde 17 zihniyetin kararları YAN YANA, tek fatura.
"Openhands 3. adımda dururdu, arize 13'te, strands 6'da, verify-gate hiç
konuşmazdı" — hepsi aynı ize göre.

SINIR — dürüst kayıt: gölge kararlar KARŞI OLGUSAL. Gerçekten müdahale
etselerdi koşum o noktada değişirdi ve sonraki adımlar farklı olurdu. Yani
"3. adımda dururdu" doğru; "3. adımda durup sonra şunu yapardı" bilinmiyor.
Bu, kesişim noktasına kadar geçerli bir karşılaştırma.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_KOK / "cua_lab"))

from cua_lab import strategies as S                       # noqa: E402
from cua_lab.detect.guardrails import Action              # noqa: E402


@dataclass
class GolgeKayit:
    sid: str
    ilk_karar: str | None = None      # ilk tetiklenen sebep
    ilk_adim: int | None = None
    ilk_tur: str | None = None        # nudge | stop | degrade
    uyari: int = 0
    snapshot: dict = field(default_factory=dict)


class GolgeKurulu:
    """Bütün zihniyetleri aynı olay akışıyla besler, kararlarını kaydeder."""

    def __init__(self, aktif: str, haric: tuple[str, ...] = ("none",)):
        self.aktif = aktif
        self.aktif_idler = {s.strip().split(":")[0] for s in aktif.split(",")}
        self.golgeler: dict[str, object] = {}
        self.kayit: dict[str, GolgeKayit] = {}
        for sid in S.all_ids():
            if sid in haric or sid in self.aktif_idler:
                continue
            try:
                self.golgeler[sid] = S.get(sid).items[0]
                self.kayit[sid] = GolgeKayit(sid)
            except Exception:
                pass

    # -- besleme -----------------------------------------------------------

    def baslat(self, ctx) -> None:
        for s in self.golgeler.values():
            try:
                s.on_run_start(ctx)
            except Exception:
                pass

    def _kaydet(self, sid: str, v, adim: int) -> None:
        if not v.triggered:
            return
        k = self.kayit[sid]
        if v.action is Action.NUDGE:
            k.uyari += 1
            if k.ilk_karar is None:
                k.ilk_karar, k.ilk_adim, k.ilk_tur = v.reason, adim, "nudge"
            return
        if k.ilk_tur in (None, "nudge"):
            k.ilk_karar, k.ilk_adim = v.reason, adim
            k.ilk_tur = "degrade" if v.action is Action.DEGRADE else "stop"

    def kanca(self, ad: str, *args, adim: int = 0) -> None:
        """`before_step` / `on_action` / `on_observation` / `on_finish_claim`
        aynı argümanlarla gölgelere de gitsin."""
        for sid, s in self.golgeler.items():
            try:
                v = getattr(s, ad)(*args)
                self._kaydet(sid, v, adim)
            except Exception:
                pass

    def durumlar(self) -> dict[str, dict]:
        out = {}
        for sid, s in self.golgeler.items():
            try:
                out[sid] = s.snapshot() or {}
            except Exception:
                out[sid] = {}
        return out

    # -- rapor -------------------------------------------------------------

    def tablo(self) -> list[tuple]:
        """(sid, tur, adim, sebep, uyari, snapshot) — ilk duranlar önce.

        `snapshot` ŞART: "sessiz kaldı" tek başına dört ayrı şey demek —
        eşik dolmadı / veri yok / kör noktası / tasarımı gereği durdurmaz.
        Ölçüldü: bir koşum 7 adımda bitti, bütçe limiti 8'di ve SAYAÇ ailesi
        "sessiz kalır" göründü. Doğruydu ama okuyan "işe yaramadı" sandı.
        Anlık durum olmadan bu ayrım yapılamıyor.
        """
        durum = self.durumlar()
        satir = []
        for sid, k in self.kayit.items():
            satir.append((sid, k.ilk_tur or "—", k.ilk_adim, k.ilk_karar or "—",
                          k.uyari, durum.get(sid) or {}))
        durdu = [s for s in satir if s[1] in ("stop", "degrade")]
        uyardi = [s for s in satir if s[1] == "nudge"]
        sessiz = [s for s in satir if s[1] == "—"]
        durdu.sort(key=lambda s: (s[2] if s[2] is not None else 999))
        uyardi.sort(key=lambda s: (s[2] if s[2] is not None else 999))
        return durdu + uyardi + sorted(sessiz)
