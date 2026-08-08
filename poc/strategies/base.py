"""
strategies/base.py — Strateji sözleşmesi + LLM-özet köprüsü.

Her strateji bir `Strategy` alt sınıfıdır ve `compact(results, conv, budget)`
uygular. Sözleşme kasıtlı olarak tek yönlü ve YERİNDE'dir:

  * strateji, `results` içindeki her `ToolResult`'ın .fate/.view/.note alanını doldurur
    (tool_call_id / mesaj yapısı ASLA bozulmaz — sadece gövde küçülür),
  * ve isteğe bağlı bir `preamble` (ör. QM'in contextSummaryPayload'ı) döndürür.

Modelin gördüğü bağlam = preamble + her result'ın .shown()'u.

Sadıklık kuralı: her strateji kendi ORİJİNAL repo'sunun fonksiyon adlarını (metod
adı olarak), sabitlerini ve placeholder string'lerini BİREBİR kullanır. Kaynak ve
§ referansı her modülün başında.
"""
from __future__ import annotations

import os
from harness import ToolResult, Conversation, est


# --------------------------------------------------------------------------
# Kader etiketleri — göstermede kullanılır (her sistem kendi terimini seçer).
# --------------------------------------------------------------------------
class Fate:
    TAM = "TAM"             # dokunulmadı
    KES = "KES"             # ortadan/uçtan kesildi (Codex, OpenClaw)
    OZET = "ÖZET"           # tek satıra indirildi (Hermes) / LLM özeti
    MASKE = "MASKE"         # gövdesi placeholder'la maskelendi (OpenHands)
    GIZLE = "GİZLE"         # depoda kalır, prompt'tan çıkar (OpenCode)
    SIL = "SİL"             # yerinde temizlendi / kaldırıldı (Claude Code, Cline, bizim)
    KATLA = "KATLA"         # yapısal outline'a katlandı (Roo)
    SUPERSEDE = "SUPERSEDE" # bayat snapshot yenisiyle geçersizleşti (gemini-cli)
    DEDUP = "DEDUP"         # aynı kaynağın eski okuması kaldırıldı (Cline, bizim)
    CRUSH = "CRUSH"         # tip-özel algoritmik sıkıştırma (Headroom)


# --------------------------------------------------------------------------
# LLM özetleyici — LLM-tabanlı stratejiler (QM, OpenHands LLMSummarizing,
# OpenCode Adım-2) için. Varsayılan OFFLINE deterministik (extractive) özet;
# LLM_LIVE=1 ve config varsa gerçek endpoint'e gider.
# --------------------------------------------------------------------------
def summarize(text: str, instruction: str = "") -> str:
    """Bir metin bloğunu özetle. Offline: baş+son satır + '(N satır özetlendi)'.
    Canlı: internal endpoint (LLM_LIVE=1). POC deterministik kalsın diye varsayılan offline."""
    if os.getenv("LLM_LIVE") == "1":
        live = _live_summary(text, instruction)
        if live:
            return live
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 3:
        return text.strip()
    head, tail = lines[0][:120], lines[-1][:120]
    return f"[LLM-özet] {head} … {tail}  ({len(lines)} satır → özet)"


def _live_summary(text: str, instruction: str) -> str:
    """İsteğe bağlı gerçek LLM özeti (internal OpenAI-uyumlu endpoint). Hata olursa boş."""
    try:
        import json
        import urllib.request
        base = os.getenv("LLM_BASE_URL", "")
        key = os.getenv("LLM_API_KEY", "")
        model = os.getenv("LLM_MODEL_NAME", "")
        if not (base and key and model):
            return ""
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": instruction or "Aşağıdaki tool çıktısını 2 satırda özetle."},
                {"role": "user", "content": text[:6000]},
            ],
            "max_tokens": 160, "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        return "[LLM-özet] " + data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Strateji tabanı.
# --------------------------------------------------------------------------
class Strategy:
    #: kısa isim (chat'te /strategy <name>)
    name: str = "base"
    #: orijinal repo
    repo: str = ""
    #: landscape § referansı
    ref: str = ""
    #: tek cümle: tool-trace mantığı
    blurb: str = ""
    #: LLM kullanıyor mu (mor) yoksa saf deterministik mi (yeşil)
    uses_llm: bool = False

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        """results'ı YERİNDE işaretle/yeniden yaz; opsiyonel preamble döndür."""
        raise NotImplementedError

    # --- ortak yardımcılar ---
    #: son kaç tool sonucu "yakın/korunur" sayılsın (POZİSYON-tabanlı; turn değil).
    #: Not: gerçek tool-use ajanında tek user turn'de onlarca tool olur — turn-tabanlı
    #: koruma tüm turu korur ve hiçbir şey sıkışmaz. Konum-tabanlı koruma çoğu repoya
    #: (attention_window, keep-active, LastN) zaten daha sadık.
    RECENT = 2

    @classmethod
    def _recent_ids(cls, results: list[ToolResult], k: int | None = None) -> set:
        return {id(r) for r in results[-(k or cls.RECENT):]}

    @staticmethod
    def _over_budget(results: list[ToolResult], budget: int, preamble: str = "") -> bool:
        base = est(preamble) if preamble else 0
        return base + sum(r.shown_tokens() for r in results) > budget

    @staticmethod
    def _shown_tokens(results: list[ToolResult], preamble: str = "") -> int:
        base = est(preamble) if preamble else 0
        return base + sum(r.shown_tokens() for r in results)
