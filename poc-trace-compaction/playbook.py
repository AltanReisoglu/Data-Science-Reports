"""
playbook.py — ACE öğrenen bağlam artefaktı (ACE, arXiv 2510.04618; §11 K4, §12.7).

Sorun (§13'te açık kalan): compactor bir hata-zincirini veya keşif dizisini
katladığında, DERS trace özetinin içinde kalır. O özet sonra bir kez daha
sıkıştırılırsa (Faz 6 / sonraki geçiş) ders kaybolur — ACE'nin "context
collapse" dediği tam bu (§11 K4, brevity bias + context collapse).

ACE'nin çözümü: dersleri trace'ten AYRI, kalıcı bir **playbook**'a yaz. Trace
evict edilse de playbook durur; bağlamın en üstüne kısa hâliyle enjekte edilir.

Üç rolün POC karşılığı (§12.7):
  - Generator = ajan (trace'i üretir)                        → agent.py
  - Reflector = compactor (ne işe yaradı/yaramadı çıkarır)   → compactor Faz 3/4
  - Curator   = Playbook.curate (ARTIMLI DELTA olarak yazar) → burası

Kritik ACE ilkesi: **yeniden yazma değil, artımlı delta.** Playbook baştan
yeniden üretilmez; yalnızca yeni madde eklenir. Bir madde zaten varsa
yeniden yazılmaz — helpful sayacı artırılır (context collapse yapısal olarak yok).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re

from config import estimate_tokens


def _norm(text: str) -> str:
    """Dedup için kaba normalizasyon: küçük harf, fazla boşluk atılır."""
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class Bullet:
    """Playbook'taki tek bir öğrenilmiş madde (ACE 'delta' birimi)."""
    id: int
    text: str
    tag: str                       # "hata-dersi" | "bulgu" | "kısıt"
    helpful: int = 1               # ACE: kaç kez teyit edildi (dedup'ta artar)
    harmful: int = 0               # ACE: kaç kez yanlış çıktı
    source_seq: Optional[int] = None

    def render(self) -> str:
        conf = f" (×{self.helpful})" if self.helpful > 1 else ""
        return f"[{self.tag}] {self.text}{conf}"


class Playbook:
    """ACE playbook: append-only, delta-güncellemeli öğrenen bağlam.

    Trace'ten bağımsız yaşar. Compactor bir ders çıkardığında curate() ile
    buraya yazar; agent.py her turda render()'ı bağlamın üstüne enjekte eder.
    """

    def __init__(self) -> None:
        self.bullets: list[Bullet] = []
        self._id = 0
        self.delta_log: list[str] = []   # her artımlı işlemi izler (ACE denetlenebilirliği)

    def curate(self, text: str, tag: str,
               source_seq: Optional[int] = None) -> Optional[Bullet]:
        """ACE Curation adımı — ARTIMLI DELTA.

        Ders zaten varsa YENİDEN YAZILMAZ; helpful sayacı artar ve None döner
        (context collapse'a karşı yapısal koruma). Yeniyse eklenir ve döner.
        """
        text = text.strip()
        if not text:
            return None
        norm = _norm(text)
        for b in self.bullets:
            if _norm(b.text) == norm:
                b.helpful += 1                      # teyit — yeniden yazma YOK
                self.delta_log.append(f"= b{b.id} teyit (×{b.helpful})")
                return None
        b = Bullet(id=self._id, text=text, tag=tag, source_seq=source_seq)
        self.bullets.append(b)
        self.delta_log.append(f"+ b{self._id} [{tag}] {text[:50]}")
        self._id += 1
        return b

    def demote(self, bullet_id: int) -> None:
        """Bir madde yanlış çıktıysa harmful++ (ACE: zararlı maddeler budanır)."""
        for b in self.bullets:
            if b.id == bullet_id:
                b.harmful += 1
                self.delta_log.append(f"- b{b.id} zararlı (×{b.harmful})")

    def active_bullets(self) -> list[Bullet]:
        """Net faydalı maddeler (helpful > harmful) — bağlama bunlar girer."""
        return [b for b in self.bullets if b.helpful > b.harmful]

    def render(self) -> str:
        """Bağlamın üstüne enjekte edilen kısa hâl (ajanın gördüğü)."""
        act = self.active_bullets()
        if not act:
            return ""
        lines = ["## Playbook (öğrenilmiş — evict'ten korunur)"]
        lines += [f"- {b.render()}" for b in act]
        return "\n".join(lines)

    def token_cost(self) -> int:
        return estimate_tokens(self.render())

    def stats(self) -> dict:
        return {"bullets": len(self.bullets),
                "active": len(self.active_bullets()),
                "deltas": len(self.delta_log)}
