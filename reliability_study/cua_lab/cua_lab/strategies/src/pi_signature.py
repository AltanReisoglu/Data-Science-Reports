"""
pi — altı ucuz sinyal, kademeli müdahale.  KATEGORİ: PENCERE · SEVİYE 5

Kaynak: `pi-anti-doom-loop` — yayımlanmış npm eklentisi

ZİHNİYET: Tek sinyal yetmez, tek tepki de yetmez. Birden çok UCUZ sinyali
topla, müdahaleyi basamak basamak sertleştir.

Varlık sebebi Seviye 2'nin kapatamadığı boşluk: AJAN AYNI ŞEYİ FARKLI
KELİMELERLE SÖYLÜYOR. `ls -la` → `ls -al` → `ls -a -l`. Üçü de aynı şey, ama
parmak izleri farklı; hiçbir imza dedektörü tetiklenmiyor.

Çözümü zekice ve UCUZ: ardışık iki mesajın KELİME ÖRTÜŞME ORANI. %55'ten fazla
ortak kelime taşıyorlarsa model aynı adımı yeniden ifade ediyor demektir.
Anlamsal benzerlik denince akla gömme modelleri gelir — her adımda bir model,
gecikme, para. Burada yapılan iş sadece kelime saymak: deterministik, anında,
bedava.

ALTI SİNYAL (kaynaktan, ayrı eşiklerle):
  1 birebir ayni (arac, argüman)      3x / son 10 cagri
  2 ayni aracin ARDISIK hatasi        3x
  3 birebir ayni metin                3x / pencere
  4 TEK mesaj icinde ayni cumle       3x+
  5 yakin-benzer ARDISIK metin        3x ust uste  (>=%55 ortusme)
  6 yakin-benzer CEVRIM               3x / pencere, ne birebir ne ardisik

Ve OpenHands'ten farkı: `escalate=True`. Metin döngülerinde önce yönlendir,
sonra kes. Kaynak, engellenen çağrıyı pencereye SOKMUYOR ve ayrı bir blok
sayacı tutuyor — bu ayrıntı `_bloklu` içinde.
"""

from __future__ import annotations

from collections import Counter, deque

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import register
from ..kinds import WindowStrategy


def _ortusme(a: str, b: str) -> float:
    """Kelime örtüşme oranı — Jaccard'ın ucuz hâli. Gömme modeli yok."""
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


@register
class PiSignature(WindowStrategy):
    id = "pi-signature"
    title = "pi: alti ucuz sinyal, kademeli mudahale"
    source = "pi-anti-doom-loop (npm)"
    mentality = "Cok sinyalli ucuz tespit + kademeli mudahale"
    priority = 7
    why = (
        "Ajan ayni cumleyi kurmasa bile AYNI FIKRI yeniden ifade ediyorsa dongudedir. Hicbir imza dedektoru bunu goremez — bu bosluk icin var.")
    action = "Alti ucuz sinyal (>=%55 kelime ortusmesi dahil); once yonlendir, tekrarda kes"
    blind_spot = (
        "Alti sinyal x ayri esik x iki merdiven — yanlis uygulanirsa sessizce olur.")
    family = "src"

    escalate = True          # once yonlendir, tekrarda kes
    benzerlik = 0.55         # >=%55 ortusme = ayni adimin yeniden ifadesi
    tekrar_cagri = 3
    tekrar_hata = 3
    tekrar_metin = 3
    cumle_tekrari = 3
    yakin_ardisik = 3
    yakin_cevrim = 3
    cagri_penceresi = 10

    def _setup(self, ctx) -> None:
        self._cagri: deque[str] = deque(maxlen=self.cagri_penceresi)
        self._hata: deque[str] = deque(maxlen=self.tekrar_hata)
        self._metin: deque[str] = deque(maxlen=self.cagri_penceresi)
        self._bloklu: dict[str, int] = {}

    # -- sinyal 1 ve 2: araç çağrıları -------------------------------------

    def probe_action(self, ev, ctx) -> Verdict:
        imza = ev.signature()
        self._cagri.append(imza)
        if self._cagri.count(imza) >= self.tekrar_cagri:
            # Kaynak: engellenen cagri pencereye GIRMIYOR; ayri bir blok
            # sayaci ikinci blokta turu bitiriyor.
            self._bloklu[imza] = self._bloklu.get(imza, 0) + 1
            return Verdict(Action.STOP, "signal1_identical_call",
                           {"sinyal": 1, "imza": imza[:8],
                            "adet": self._cagri.count(imza),
                            "blok": self._bloklu[imza]})
        return CONTINUE

    def probe_observation(self, ev, ctx) -> Verdict:
        if ev.kind is EventKind.ERROR:
            self._hata.append(ev.signature())
            if (len(self._hata) >= self.tekrar_hata
                    and len(set(self._hata)) == 1):
                return Verdict(Action.STOP, "signal2_repeat_error",
                               {"sinyal": 2, "adet": len(self._hata)})
        elif ev.kind is EventKind.MESSAGE:
            return self._metin_sinyalleri(ev)
        return CONTINUE

    # -- sinyal 3-6: metin --------------------------------------------------

    def _metin_sinyalleri(self, ev) -> Verdict:
        metin = str(ev.payload.get("text", ""))
        if not metin.strip():
            return CONTINUE

        # 4 — TEK mesaj icinde ayni cumle tekrari
        cumleler = [c.strip() for c in metin.replace("!", ".").replace("?", ".").split(".")
                    if len(c.strip()) > 8]
        if cumleler:
            en_cok = Counter(cumleler).most_common(1)[0][1]
            if en_cok >= self.cumle_tekrari:
                return Verdict(Action.STOP, "signal4_sentence_repeat",
                               {"sinyal": 4, "adet": en_cok})

        # 3 — birebir ayni metin
        if self._metin.count(metin) + 1 >= self.tekrar_metin:
            self._metin.append(metin)
            return Verdict(Action.STOP, "signal3_identical_text",
                           {"sinyal": 3, "adet": self._metin.count(metin)})

        # 5 — yakin-benzer ARDISIK (ust uste 3x)
        if len(self._metin) >= self.yakin_ardisik - 1:
            son = list(self._metin)[-(self.yakin_ardisik - 1):] + [metin]
            if all(_ortusme(a, b) >= self.benzerlik for a, b in zip(son, son[1:])):
                self._metin.append(metin)
                return Verdict(Action.STOP, "signal5_near_consecutive",
                               {"sinyal": 5, "esik": self.benzerlik})

        # 6 — yakin-benzer CEVRIM: pencerede birikiyor, ne birebir ne ardisik
        benzer = sum(1 for m in self._metin if _ortusme(m, metin) >= self.benzerlik)
        self._metin.append(metin)
        if benzer + 1 >= self.yakin_cevrim:
            return Verdict(Action.STOP, "signal6_near_cycle",
                           {"sinyal": 6, "benzer": benzer + 1})
        return CONTINUE

    def nudge_text(self, reason: str, detail: dict) -> str:
        return (f"DEDEKTOR '{reason}' (sinyal {detail.get('sinyal')}) tetiklendi — "
                f"ayni adimi tekrar etme, YAKLASIMI degistir.")

    def snapshot(self) -> dict:
        """Altı sinyalin sayaçları. Hangi sinyal ne kadar dolu."""
        cagri = list(getattr(self, "_cagri", []))
        en_cok_cagri = max((cagri.count(x) for x in set(cagri)), default=0)
        hata = list(getattr(self, "_hata", []))
        ardisik_hata = len(hata) if hata and len(set(hata)) == 1 else 0
        metin = list(getattr(self, "_metin", []))
        en_cok_metin = max((metin.count(x) for x in set(metin)), default=0)
        return {"tur": "pencere", "sinyal": [
            ("1 birebir cagri", en_cok_cagri, self.tekrar_cagri),
            ("2 ardisik hata", ardisik_hata, self.tekrar_hata),
            ("3 birebir metin", en_cok_metin, self.tekrar_metin),
            ("5/6 yakin-benzer", len(metin), self.yakin_cevrim)],
            "not": f"benzerlik esigi %{int(self.benzerlik * 100)}"}

    def reasons(self):
        return ("signal1_identical_call", "signal2_repeat_error",
                "signal3_identical_text", "signal4_sentence_repeat",
                "signal5_near_consecutive", "signal6_near_cycle")

    def why_text(self, reason: str, ctx) -> str:
        return f"alti sinyalden biri tetiklendi: {reason}"
