#!/usr/bin/env python3
"""
brain_core — brain_chat_V2'nin MINIMAL, framework-BAĞIMSIZ çekirdeği.

Gerçek brain_chat_V2 kapalı kaynaktır. Bu modül onun ÇEKİRDEK ajan-döngüsünü
temsil eden, 4 adımlı bir "iş"i (A) modeller. Bütün wrap POC'ları (Temporal /
Celery / Hermes) ve build-ourselves POC'u AYNI bu çekirdeği kullanır — böylece
"aynı beyin, farklı task-mgmt altyapısı" karşılaştırması adil olur.

  brain işi (A) = "kullanıcı isteğini yanıtla":
    1) retrieve  — bağlam/RAG topla        [PAHALI dış okuma → retry'da tekrar-etmemesi kritik]
    2) reason    — LLM muhakeme/plan üret   [gerçekte activity'e sarılı LLM çağrısı]
    3) act       — tool çalıştır            [YAZMA yan etkisi → idempotency önemli]
    4) respond   — nihai yanıtı üret

Her adım (B) idempotenttir ve partial-state döndürür; bu yüzden bir checkpoint'ten
DEVAM edilebilir. Senaryo: 'reason' İLK denemede geçici hata verir (LLM gateway
timeout) → her framework'ün retry/recovery yolunu tetikler. Ölçtüğümüz asıl soru:
**retry/çökme sonrası `retrieve` kaç kez koşar?** (pahalı adımı koruyabiliyor muyuz?)
"""
from __future__ import annotations

STEPS = ("retrieve", "reason", "act", "respond")


def retrieve(order_id: str) -> str:
    """1) Bağlam topla (RAG). PAHALI dış okuma."""
    return f"ctx({order_id})=[kullanıcı geçmişi + 3 döküman parçası]"


def reason(ctx: str, attempt: int) -> str:
    """2) LLM muhakeme/plan. İLK denemede (attempt==0) geçici hata verir.

    Gerçek brain_chat_V2'de bu adım bir LLM çağrısıdır; Temporal'a sarınca
    determinizm için activity'e konur. Burada deterministik tutuldu ki
    replay/idempotency net görünsün."""
    if attempt == 0:
        raise RuntimeError("geçici LLM hatası (model gateway timeout)")
    return f"plan[{ctx}] → tool=create_order(#4711)"


def act(plan: str) -> str:
    """3) Tool çalıştır. YAZMA yan etkisi (idempotency önemli)."""
    return f"aksiyon<{plan}> = sipariş_oluşturuldu"


def respond(action: str) -> str:
    """4) Nihai yanıt."""
    return f"yanıt: {action} → kullanıcıya döndü"


def run_from(order_id, done: dict, attempt: int, on_step=None, on_attempt=None) -> str:
    """Tüm adımları, verilen `done` checkpoint'inden DEVAM ederek koşturur.

    - done: {step: output}  → zaten biten adımlar atlanır (kaldığı yerden devam).
    - Bir adım hata verirse exception yükselir ama `done` KORUNUR (checkpoint).
    - on_attempt(step): adım GERÇEKTEN koşmadan HEMEN önce çağrılır (hata verse de
      sayılır → doğru "kaç kez denendi" metriği). Atlanan adımda çağrılmaz.
    - on_step(step, output, skipped): adım BAŞARIYLA bittikten sonra çağrılır
      (checkpoint/heartbeat/log). Hata veren adımda çağrılmaz.

    Bu fonksiyon "build-ourselves" ve checkpoint'li senaryoların kalbi:
    retrieve bir kez bitip done'a yazıldıysa, retry'da BİR DAHA koşmaz.
    """
    def _do(step, fn):
        if step in done:
            if on_step:
                on_step(step, done[step], True)
            return done[step]
        if on_attempt:
            on_attempt(step)          # hata verse bile sayılır (gerçek deneme)
        out = fn()
        done[step] = out
        if on_step:
            on_step(step, out, False)
        return out

    ctx    = _do("retrieve", lambda: retrieve(order_id))
    plan   = _do("reason",   lambda: reason(ctx, attempt))
    action = _do("act",      lambda: act(plan))
    answer = _do("respond",  lambda: respond(action))
    return answer
