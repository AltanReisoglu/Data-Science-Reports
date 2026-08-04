"""
ptc.py — Programatik Tool Çağrısı (PTC; §11 K3 / B.9, §12.10).

Compaction/context-editing ZAMANSAL (biriktir, sonra sıkıştır). PTC UZAYSAL:
ara sonuçlar bağlama HİÇ girmez — modelin yazdığı kod sandbox'ta çalışır,
bağlama yalnızca son `print` girer (§12.10).

  Klasik:  200 tool çağrısı → 200 tool_result bağlamda
  PTC:     tek kod bloğu → 200 çağrı sandbox'ta → 1 print bağlamda

Bu POC'de sandbox, kısıtlı bir `exec` namespace'idir: yalnızca beyaz-listeli
builtin'ler + POC tool'ları enjekte edilir. Ara çağrılar `call_log`'a düşer
(yalnızca muhasebe/demo için); trace'e YALNIZCA nihai print bir olay olarak girer.

DÜRÜSTLÜK NOTU: gerçek PTC sandbox'ı süreç/konteyner izolasyonu ister
(code_execution_20250825 gibi). Kısıtlı exec eğitim amaçlı bir yaklaşımdır;
üretimde güvenilmez kod izole bir çekirdekte çalıştırılmalıdır (ek-a: kod da
model çıktısıdır → güvenilmez).
"""
from __future__ import annotations
import io
import contextlib
import traceback
from typing import Callable

# exec namespace'ine verilen güvenli builtin beyaz listesi (open/import/eval YOK)
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ("print", "range", "len", "sorted", "sum", "min", "max",
              "enumerate", "list", "dict", "str", "int", "float", "bool",
              "tuple", "set", "zip", "map", "filter", "abs", "any", "all",
              "reversed", "round", "isinstance", "Exception", "sorted")
}


class PTCSandbox:
    """Kısıtlı sandbox: tool'ları kod olarak çalıştırır, sadece print'i döndürür."""

    def __init__(self, tools: dict[str, Callable]) -> None:
        self.tools = tools
        self.call_log: list[tuple[str, dict]] = []   # ara çağrılar (yalnızca muhasebe)

    def _wrap(self, name: str, fn: Callable) -> Callable:
        """Bir tool'u, çağrıyı loglayıp asıl işi yapan bir sarmalayıcıya çevir.

        Kod tool'ları hem pozisyonel hem isimli çağırabilir: grep('PORT', f).
        """
        def caller(*args, **kwargs):
            self.call_log.append((name, {"args": args, "kwargs": kwargs}))
            return fn(*args, **kwargs)
        return caller

    def _namespace(self) -> dict:
        ns = {"__builtins__": dict(_SAFE_BUILTINS)}
        for name, fn in self.tools.items():
            ns[name] = self._wrap(name, fn)
        return ns

    def run(self, code: str) -> dict:
        """Kodu çalıştır. Döndürür: {output, status, inner_calls}.

        Hata olursa STACK TRACE sandbox sonucudur (§12.10 hata kurtarma):
        model bunu kod düzeyinde düzeltir, TUR harcamadan yeniden çalıştırır.
        """
        self.call_log = []
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<ptc>", "exec"), self._namespace())
            status = "ok"
            output = buf.getvalue().strip() or "(çıktı yok)"
        except Exception:
            status = "error"
            output = traceback.format_exc(limit=3).strip()   # kod hatası = ders
        return {"output": output, "status": status,
                "inner_calls": len(self.call_log)}
