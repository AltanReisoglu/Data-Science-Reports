"""
LoopGuard — onurunla dur.  KATEGORİ: PENCERE · SEVİYE 2 (üçüncü seviyeye köprü)

Kaynak: *Ne Zaman Durması Gerektiğini Bilen Ajanlar*
(kavramsal yazı + çalışan referans kod: `check_budget` / `record_progress` /
`record_tool_call` / `should_retry` / `detect_repeat_pattern`)

ZİHNİYET: Durmak başarısızlık değildir; KÖTÜ durmak başarısızlıktır.
    "Ajanlar sadece yanıldıkları için başarısız olmazlar. Israrcı oldukları
     için başarısız olurlar. Ve ısrar pahalıdır."

Aynı pencereyi kullanıyor ama BAŞKA BİR ŞEYE bakıyor: eyleme değil DÜNYANIN
DURUMUNA. Ajan her turda farklı bir şey deniyor olabilir — hiçbir tekrar
dedektörü tetiklenmez ama sistem yerinde sayıyordur.
    "Hareket etmek, ilerlemek demek değildir."

İki farkı daha:
  * CEKIMSER KALMA birinci sinif sonuc — `NEEDS_INPUT`, hata degil.
  * DORT ALANLI durma raporu: ne denedi, ne buldu, neden durdu, sirada ne var.
    "Bu, sonsuz donguyu faydali kismi sonuca donusturur."

KAYNAĞIN SAYILARI DÜZELTİLDİ — ölçerek bulundu (bkz. tests/test_faz2.py):
  * `state_hash` esigi 3 -> 8. `flaky` izinde ekran 1-2-3. adimlarda ayni
    kaliyor ve basari 4. adimda geliyor; kaynagin sayisi mesru kosumu
    BASARIDAN BIR ADIM ONCE keserdi.
  * retry anahtari `dict[arac]` -> `(eylem, argüman)`. Kaynak bir METIN ajani
    varsayiyor; computer use'da `left_click` isin %90'ini yapiyor.
"""

from __future__ import annotations

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import StopReport, register
from ..kinds import WindowStrategy


@register
class LoopGuardDignity(WindowStrategy):
    id = "loopguard-dignity"
    title = "LoopGuard: onurunla dur"
    source = "Ne Zaman Durmasi Gerektigini Bilen Ajanlar"
    mentality = "Hareket ilerleme degildir — cekimser kalmak mesru sonuctur"
    priority = 9
    why = (
        "Ajan farkli eylemler yapsa bile dis dunyada hicbir sey degismiyorsa ilerleme yok demektir. Tekrar sayan hicbir dedektor bunu goremez.")
    action = "Ekran/durum hash'i degismiyorsa CEKIMSER KAL (NEEDS_INPUT) + dort alanli rapor"
    blind_spot = (
        "Hizli ve ucuz donguler sinirlar dolmadan cok tur donebilir.")
    family = "src"

    escalate = False
    terminal = "NEEDS_INPUT"
    max_retries_per_action = 3    # kaynak 2 diyor; flaky kontrolu icin 3
    no_progress = 8               # kaynak 3 diyor; olculmus degerimiz 8

    def _setup(self, ctx) -> None:
        self._retries: dict[str, int] = {}
        self._tried: list[str] = []

    def probe_action(self, ev, ctx) -> Verdict:
        etiket = f"{ev.name}({ev.payload})"
        if etiket not in self._tried:
            self._tried.append(etiket)
        return CONTINUE

    def probe_observation(self, ev, ctx) -> Verdict:
        # 1) eylem bazlı deneme hakkı
        if ev.kind is EventKind.ERROR:
            key = ev.signature()
            self._retries[key] = self._retries.get(key, 0) + 1
            if self._retries[key] >= self.max_retries_per_action:
                return Verdict(Action.STOP, "abstain_need_input",
                               {"neden": "action_retry_exhausted", "eylem": ev.name,
                                "deneme": self._retries[key], "abstain": True})
        elif ctx.last_screen_changed():
            # İlerleyen bir retry, tükenen bir retry değildir.
            self._retries.clear()

        # 2) ilerleme yokluğu — dünyanın durumu değişmiyor
        son = ctx.screen_hashes[-self.no_progress:]
        if len(son) >= self.no_progress and len(set(son)) == 1:
            return Verdict(Action.STOP, "abstain_need_input",
                           {"neden": "no_progress", "adim": self.no_progress,
                            "ekran": son[-1], "abstain": True})
        return CONTINUE

    def snapshot(self) -> dict:
        ctx = getattr(self, "_ctx", None)
        durgun = 0
        if ctx is not None:
            son = ctx.screen_hashes[-self.no_progress:]
            durgun = len(son) if son and len(set(son)) == 1 else 0
        en_cok = max(self._retries.values(), default=0)
        return {"tur": "dunya", "sinyal": [
            ("ekran degismedi", durgun, self.no_progress),
            ("ayni eylem hatasi", en_cok, self.max_retries_per_action)],
            "not": f"{len(self._tried)} farkli yaklasim denendi"}

    def reasons(self):
        return ("abstain_need_input",)

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason != "abstain_need_input":
            return None
        return StopReport(
            reason=reason,
            why="ilerleme durdu — devam etmek yerine cekimser kaliyorum "
                "(yenilgi degil, ayri bir sonuc)",
            tried=" | ".join(self._tried[-4:]) or "-",
            found=f"{ctx.step} adimda ekran {len(set(ctx.screen_hashes))} farkli "
                  f"durum gordu; son {self.no_progress} adimda hic degismedi",
            next_step="SORU: hedef buton tiklanabilir durumda mi, yoksa baska bir "
                      "alan mi doldurulmali? Cevap gelmezse elimdekiyle raporlarim.",
        )
