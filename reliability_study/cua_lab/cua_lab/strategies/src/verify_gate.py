"""
Loop Engineering — "bitirdim" bir istektir, kanıt değil.  KATEGORİ: DÜNYA · SEVİYE 3

Kaynak: *sde_offer_loop* (Loop Engineering)

ZİHNİYET: Ajan işini bitirdiğini söylediğinde bu bir BİLGİ değil, bir TALEPTİR:
"durmak istiyorum." Bunu doğrudan kabul etmek, öğrencinin kendi sınav kâğıdını
okuması gibi.
    "'Bitirdim dedi', ajan dünyasının 'benim makinemde derleniyor'udur."

Ortak reddetme merdiveni `kinds/evidence.py`'de. Bu dosyanın FARKI kanıtı
NEREDEN aldığı: ORTAMDAN. Modelin görüşü değil, dünyanın hâli.

KÖR NOKTASI aynı yerden geliyor ve BELGELENMİŞ bir sınır: ajan hiç "bitirdim"
demezse bu kapı hiç açılmaz. Gerçek bir LLM ile ölçüldü — 50 adım boyunca
`continue`, en pahalı sütun oldu. Mutlaka bir bütçe stratejisiyle birlikte:

    --strategy verify-gate,arize-control
"""

from __future__ import annotations

from ...events import CONTINUE, Verdict
from ..base import register
from ..kinds import EvidenceStrategy


@register
class VerifyGate(EvidenceStrategy):
    id = "verify-gate"
    title = "Loop Engineering: dogrulama kapili durma"
    source = "sde_offer_loop (Loop Engineering)"
    mentality = "'Bitirdim' bir istektir, kanit degil — ortama sor"
    priority = 10
    why = (
        "Ajanin 'tamamlandi' iddiasi bir BILGI degil bir TALEPTIR. Dogrudan kabul etmek, ogrencinin kendi sinav kagidini okumasi gibidir.")
    action = "Testi/dosyayi/ekrani dogrula; gecmezse KOSUMU BITIRME — gozleme geri ver"
    blind_spot = (
        "Ajan hic 'bitirdim' demezse bu kapi HIC acilmaz. Butceyle birlikte sart.")

    def _dogrula(self, ctx) -> tuple[bool, str]:
        """Doğrulayıcı GÖREVE ÖZEL — koşum onu sağlıyorsa o kullanılıyor.

        Sabit bir kanıt dizesine bağlanmak yapısal yanlış pozitif üretiyor:
        `"gonderildi"` bu paketin sentetik form senaryosunun başarı işareti.
        Aynı strateji bir terminal görevinde koşunca o dize hiç görünmez ve
        HER bitirme iddiası reddedilir — ölçüldü, `agentcli`'de tam olarak
        bu oldu.

        `ctx.extra["dogrulayici"]` bir çağrılabilir: `(ctx, iddia) -> (bool, str)`.
        Yoksa sentetik senaryonun kontrolüne düşülüyor.
        """
        dv = (ctx.extra or {}).get("dogrulayici")
        if callable(dv):
            try:
                return dv(ctx, getattr(self, "_son_iddia", ""))
            except Exception as e:
                return False, f"dogrulayici hata verdi: {e}"
        ekran = ctx.sandbox.describe()
        if "gonderildi" in ekran:
            return True, "ekranda 'gonderildi' gorundu"
        return False, f"ekranda tamamlanma kaniti yok: {ekran[:60]}"

    def on_finish_claim(self, fin, ctx) -> Verdict:
        self._son_iddia = getattr(fin, "answer", "")
        ok, kanit = self._dogrula(ctx)
        if ok:
            return CONTINUE
        return self._reddet("verify_failed", kanit)

    def snapshot(self) -> dict:
        return {"tur": "dunya", "sinyal": [
            ("reddedilen iddia", getattr(self, "_red", 0), self.max_rejections + 1)],
            "not": "iddia gelmezse HIC calismaz — kor nokta"}

    def reasons(self):
        return ("verify_failed",)

    def why_text(self, reason: str, ctx) -> str:
        return f"{self._red} kez 'bitirdim' dendi, {self._red} kez dogrulama dustu"
