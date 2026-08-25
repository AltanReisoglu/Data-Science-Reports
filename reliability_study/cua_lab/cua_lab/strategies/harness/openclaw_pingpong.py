"""
OpenClaw — adlandırılmış dedektörler ve sıkıştırma tuzağı.  KATEGORİ: PENCERE · SEVİYE 2

Kaynak: openclaw 2026.7.1-2, bu makinedeki kurulum (`tools.loopDetection`)

ZİHNİYET: Üretimde yıllanmış bir dedektör kümesi — ve bir olgunluk göstergesi:
ÇÖZÜMÜN KENDİSİ SORUN KAYNAĞI OLABİLİR.

Uzun süren ajanların bağlamı dolar; çözüm sıkıştırma. Ama sıkıştırmanın kendisi
döngü üretebiliyor: sıkıştır → bağlam yine dolsun → tekrar sıkıştır. OpenClaw
buna özel bir savunma koymuş: sıkıştırmadan sonra üç deneme boyunca AYRI bir
koruma kurulu kalıyor.

OpenHands ile AYNI pencereyi kullanıyor. Farkı üç yerde:

  1. Dedektorlerin ADI var: genericRepeat / knownPollNoProgress / pingPong.
     Isim vermek kozmetik degil — hangi dedektorun konustugunu bilmek,
     ne oldugunu bilmek demek.
  2. UC KADEME: 10 uyari -> 20 kritik -> 30 kuresel devre kesici.
  3. Parmak izi = arac + argüman + SONUC hash'i.

Not: kaynakta `enabled` varsayılanı **false**. Burada açık — "kapalı bir
varsayılan, olmayan bir limitle aynı şeydir".
"""

from __future__ import annotations

from collections import Counter, deque

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import register
from ..kinds import WindowStrategy


@register
class OpenClawPingPong(WindowStrategy):
    id = "openclaw-pingpong"
    title = "OpenClaw: adlandirilmis dedektorler + kademeli kesici"
    source = "openclaw 2026.7.1-2 tools.loopDetection"
    mentality = "Adlandirilmis dedektorler + sikistirma sonrasi koruma"
    priority = 6
    why = (
        "Ayni hatanin ya da dongunun sistemi surekli tuketmesini engeller. Ayrica CozUMUN KENDISININ sorun olabilecegini gorur: baglam sikistirmasi dongu uretebilir.")
    action = "Adlandirilmis dedektorler + uc kademe (10 uyari -> 20 kritik -> 30 kesici)"
    blind_spot = (
        "Kaynakta varsayilan KAPALI. Ucuncu kademeye kadar cok para yanabilir.")
    family = "harness"

    escalate = True          # kademe var: 10 uyari -> 20 kritik
    warning_threshold = 10
    critical_threshold = 20
    ping_pong_min = 4        # A-B-A-B icin en az kac olay
    post_compaction_guard = 3

    def _setup(self, ctx) -> None:
        self._fp: deque[str] = deque(maxlen=30)   # historySize=30

    def probe_observation(self, ev, ctx) -> Verdict:
        # Parmak izi ARAC + ARGUMAN + SONUC — kaynagin tercihi.
        if not self._fp and not ctx.events:
            return CONTINUE
        son = ctx.events[-2:]
        if len(son) == 2:
            self._fp.append(son[0].signature() + ":" + son[1].signature())

        if len(self._fp) < self.ping_pong_min:
            return CONTINUE

        sayac = Counter(self._fp)
        en_cok, adet = sayac.most_common(1)[0]

        # 1) pingPong — iki imza arasinda gidip gelme
        kuyruk = list(self._fp)[-self.ping_pong_min:]
        if len(set(kuyruk)) == 2 and kuyruk[0] == kuyruk[2] and kuyruk[1] == kuyruk[3]:
            return Verdict(Action.STOP, "pingPong",
                           {"dedektor": "pingPong", "pencere": self.ping_pong_min})

        # 2) genericRepeat — ayni cift tekrar tekrar
        if adet >= self.critical_threshold:
            return Verdict(Action.STOP, "genericRepeat_critical",
                           {"dedektor": "genericRepeat", "adet": adet,
                            "esik": self.critical_threshold})
        if adet >= self.warning_threshold:
            return Verdict(Action.STOP, "genericRepeat",
                           {"dedektor": "genericRepeat", "adet": adet,
                            "esik": self.warning_threshold})

        # 3) knownPollNoProgress — ekran hic degismiyor ve ayni cagri
        son_ekran = ctx.screen_hashes[-self.warning_threshold:]
        if (len(son_ekran) >= self.warning_threshold
                and len(set(son_ekran)) == 1 and adet >= 3):
            return Verdict(Action.STOP, "knownPollNoProgress",
                           {"dedektor": "knownPollNoProgress",
                            "ekran": son_ekran[-1]})
        return CONTINUE

    def snapshot(self) -> dict:
        from collections import Counter
        fp = list(getattr(self, "_fp", []))
        adet = Counter(fp).most_common(1)[0][1] if fp else 0
        return {"tur": "pencere", "sinyal": [
            ("ayni cift tekrari", adet, self.warning_threshold),
            ("kritik esik", adet, self.critical_threshold)],
            "not": f"parmak izi = arac+arg+SONUC · {len(fp)} cift"}

    def reasons(self):
        return ("pingPong", "genericRepeat", "genericRepeat_critical",
                "knownPollNoProgress")

    def why_text(self, reason: str, ctx) -> str:
        return f"dedektor '{reason}' tetiklendi (adlandirilmis dedektor kumesi)"
