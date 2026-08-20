"""Turun tepegözü: OpenTelemetry span'leri, koşarken toplanıyor.

`observability.py` AutoGen'in **olay** akışını dinliyor — hangi tool çağrıldı,
kaç token gitti. Bu modül onun üstündeki katman: **span**'ler, yani her işin
başlangıç–bitiş aralığı ve iç içe geçme ilişkisi. İkisi farklı sorular
cevaplıyor: olaylar "ne oldu", span'ler "ne kadar sürdü ve neyin içinde oldu".

AutoGen bunu kendisi yayıyor. `SingleThreadedAgentRuntime(tracer_provider=...)`
verildiğinde `gen_ai.*` sözleşmesine uygun span'ler üretiyor
(`autogen_core/_telemetry/_genai.py`): `gen_ai.system="autogen"`, ajan adı, tool
adı, hata tipi. Yani buradaki iş bir izleyici *yazmak* değil, yayılanı toplamak.

### Neden dışarı bir toplayıcıya göndermiyoruz

Jaeger ya da Zipkin kurmak bu iş için doğru cevap; ama sunumda ikinci bir servis
demek, ve ekranın anlatacağı şey zaten tek bir turun içi. Span'ler bellekte
tutuluyor, tur başına, ve tur kaydıyla birlikte düşüyor. Dışarı çıkarmak
istendiğinde `provider()`'a ikinci bir işlemci eklemek yetiyor — mimari o yöne
kapalı değil.
"""

from __future__ import annotations

import threading
from typing import Any

# En fazla kaç span tutulacağı. Bir tur bunu aşıyorsa gösterilecek şey zaten
# okunabilir bir şelale değil.
CAP = 400


def _base():
    """SDK'nın `SpanProcessor`'ı — arayüzü ördek tiplemeyle karşılamak yetmiyor.

    Ölçüldü: yalnız `on_start`/`on_end` tanımlayan bir nesne verildiğinde SDK
    `_on_ending` diye özel bir kancayı çağırıyor ve
    `AttributeError: 'Collector' object has no attribute '_on_ending'` ile
    turu düşürüyor. Taban sınıfı türetmek, sürüm sürüm değişen bu kancaları
    bedava getiriyor.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    return SpanProcessor


class Collector(_base()):  # type: ignore[misc]
    """Biten span'leri bir listeye yazan işlemci.

    `on_end` runtime'ın iş parçacığında çağrılıyor, o yüzden kilit var ve
    içerisi kısa: bir izleyici, izlediği işi yavaşlatmamalı.
    """

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # --- OTel SpanProcessor arayüzü ---------------------------------------
    def on_end(self, span: Any) -> None:
        try:
            ctx = span.get_span_context()
            parent = getattr(span, "parent", None)
            row = {
                "name": span.name,
                "id": format(ctx.span_id, "016x"),
                "parent": format(parent.span_id, "016x") if parent else None,
                "start": span.start_time,          # ns
                "end": span.end_time,
                "ms": round((span.end_time - span.start_time) / 1e6, 2),
                "attrs": {k: str(v)[:120] for k, v in dict(span.attributes or {}).items()},
                "status": getattr(getattr(span, "status", None), "status_code", None)
                          and str(span.status.status_code),
            }
        except Exception:  # noqa: BLE001 — izleyici turu düşüremez
            return
        with self._lock:
            self.spans.append(row)
            if len(self.spans) > CAP:
                del self.spans[: CAP // 4]

    # --- okuma --------------------------------------------------------------
    def report(self) -> list[dict[str, Any]]:
        """Span'ler, en erken başlayan başta ve başlangıca göre göreli.

        Mutlak nanosaniyeler ekranda hiçbir şey anlatmıyor; şelalede okunan şey
        "ne zaman başladı, ne kadar sürdü" — ikisi de göreli.
        """
        with self._lock:
            rows = sorted(self.spans, key=lambda s: s["start"])
        if not rows:
            return []
        t0 = rows[0]["start"]
        total = max(r["end"] for r in rows) - t0 or 1
        out = []
        for r in rows:
            out.append({
                "name": r["name"],
                "id": r["id"],
                "parent": r["parent"],
                "at": round((r["start"] - t0) / 1e9, 3),
                "ms": r["ms"],
                "offset": round((r["start"] - t0) / total, 4),
                "width": round((r["end"] - r["start"]) / total, 4),
                "attrs": r["attrs"],
                "status": r["status"],
            })
        return out


def provider() -> tuple[Any, Collector]:
    """Bir tur için izleyici sağlayıcısı ve onun toplayıcısı.

    Tur başına ayrı sağlayıcı: küresel bir sağlayıcıya yazmak, iki turun
    span'lerini birbirine karıştırırdı ve hangi span'in hangi soruya ait olduğu
    kaybolurdu — bu ekranda cevaplanması gereken ilk soru tam olarak o.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    collector = Collector()
    tracer = TracerProvider(resource=Resource.create({"service.name": "vc-agent"}))
    tracer.add_span_processor(collector)
    return tracer, collector


def flush(tracer_provider: Any) -> None:
    """Kapanmadan önce bekleyen span'leri boşalt. Asla fırlatmaz."""
    try:
        tracer_provider.force_flush()
    except Exception:  # noqa: BLE001
        pass
