"""Modelin yazdığı kodu bir konteynerde çalıştırmak — kaçış kapağı olarak.

Bu, AutoGen'in resmî sekiz deseninden sonuncusu: **Code Execution**. Diğer
yedisi orkestrasyon deseni; bu bir *yetenek*, ve bu depoda uzun süre bilerek yok
sayıldı. Şimdi giriyor, ama dar bir rolle.

### Rol: tool'ların yerine değil, olmadıklarında

Ajanın yirmi bir tool'u var. Bu tool o listeye yirmi ikinci bir seçenek olarak
değil, **listede karşılığı olmayan işler için** ekleniyor: bir hesap, bir
dönüştürme, bir ayrıştırma. Tarif bunu söylüyor ve sıralamayı zorunlu kılıyor,
çünkü `docs/05:2473`'ün dediği gibi modelin bir tool'a ne zaman uzanacağına karar
verdiği metin şemanın kendisi değil, **açıklaması**.

Ayrım pratik: her şeyi kodla yapan bir ajan, yirmi bir tool'u boşa çalıştırır ve
her hesabı yeniden icat eder.

### Ömür: konteyner sürece ait, çağrıya değil

`gateway/runtime.py`'nin uzun ömürlü runtime'ıyla aynı disiplin. Her çağrıda
konteyner ayağa kaldırmak iki-üç saniye sürüyor ve bunun tamamı kullanıcının
beklediği süreye ekleniyor. `start()` ve `stop()` sunucunun kendi yaşam
döngüsüne bağlı.

### Ölçülmüş sınır: KONTEYNERİN AĞ ERİŞİMİ VAR

`DockerCommandLineCodeExecutor`'da `network_mode`, `read_only`, `mem_limit`,
`user` gibi **hiçbir** parametre yok — kaynağında ağ ile ilgili tek kelime
geçmiyor (ölçüldü, autogen-ext 0.7.5). Konteyner Docker'ın varsayılan bridge
ağına bağlanıyor, yani **interneti var**.

Bu modül o sorunu çözmüyor, üç şey yapıyor:

1. Özellik varsayılan **kapalı** (`config.ALLOW_CODE_EXEC`).
2. Onay kartı ağ erişimini **açıkça yazıyor** — operatör neyi onayladığını
   bilerek onaylıyor.
3. Sertleştirme yolu burada kayıtlı: `DockerCommandLineCodeExecutor.start()`
   override edilip `containers.create(..., network_mode="none")` geçilebilir.
   Yapılmadı, çünkü yukarı akışın iç koduna bağımlılık yaratıyor ve sürüm
   değişince sessizce kırılır. Ayrı bir karar, ayrı bir iş.

"Sandbox güvenli" cümlesi bu dosyada kurulmuyor. Kurulan cümle şu: *kod izole
bir konteynerde koşuyor ve çalışmadan önce bir insan onaylıyor.*
"""

from __future__ import annotations

from typing import Any

import config

# Ajanın gördüğü ad. `PythonCodeExecutionTool` bunu kendisi koyuyor ve
# değiştirilemiyor; kapı kancası da bu adı arıyor, o yüzden tek yerde duruyor.
TOOL_NAME = "CodeExecutor"

# Tarif = arayüz. Modelin bu tool'a *ne zaman* uzanacağına karar verdiği metin
# bu; "kod çalıştırır" deseydi her şeyi kodla yapmaya başlardı.
DESCRIPTION = (
    "Kullanıcı açıkça kod isterse (\"kod yaz\", \"kodla hesapla\") BU TOOL'U "
    "ÇAĞIR — hesabı kafandan yapıp adımları yazmak, isteği karşılamak değil "
    "reddetmektir. Aksi hâlde son çare: ÖNCE mevcut tool'lara bak, sorulanı "
    "karşılayan bir tool varsa onu çağır. Hiçbiri uymuyorsa — bir hesap, bir "
    "dönüştürme, bir ayrıştırma, bir istatistik gibi — Python yaz ve burada "
    "çalıştır. Kod izole bir konteynerde koşar ve çalışmadan önce kullanıcı "
    "onaylar, o yüzden tek seferde çalışacak ve okunması kolay kod yaz. "
    "Sonucu print() ile yazdır."
)

# Onay kartında görünen gerekçe. Ağ erişimi burada yazılı: operatörün neyi
# onaylamadığını da bilmesi gerekiyor.
GATE_REASON = (
    "Modelin yazdığı Python kodu çalıştırılacak. Kod izole bir Docker "
    "konteynerinde koşuyor, ama konteynerin ağ erişimi var — AutoGen'in "
    "yürütücüsü ağ izolasyonu için bir parametre sunmuyor."
)


class CodeExecUnavailable(RuntimeError):
    """Yürütücü kurulamıyor — Docker yok, kapalı, ya da SDK eksik."""


_TOOL: Any = None
_EXECUTOR: Any = None


def available() -> bool:
    """Özellik açık mı ve gerekli parçalar yerinde mi?

    `ImportError` yakalamak yetmiyor: `autogen_ext.code_executors.docker`, docker
    ekstrası kurulu değilse import anında **`RuntimeError`** fırlatıyor
    (*"Missing dependecies for DockerCommandLineCodeExecutor"* — yazım hatası
    onların). Dar bir `except ImportError` bunu geçirir ve sunucu açılışta
    ölür — ki bu, özelliğin kapalı olmasından çok daha kötü bir arıza.
    Ölçüldü: ilk denemede tam olarak bu oldu.
    """
    if not config.ALLOW_CODE_EXEC:
        return False
    try:
        import docker  # noqa: F401
        from autogen_ext.code_executors.docker import (  # noqa: F401
            DockerCommandLineCodeExecutor,
        )
    except Exception:  # noqa: BLE001 — eksik bağımlılık sunucuyu düşürmemeli
        return False
    return True


def build_tool() -> Any:
    """Tool'u kur (ya da kurulmuş olanı ver).

    Yumuşak düşüyor: Docker olmayan bir makinede sunucunun açılmaması, bu
    özelliğin kapalı olmasından çok daha kötü. Çağıran `None` görürse tool'u
    listeye hiç koymuyor.
    """
    global _TOOL, _EXECUTOR

    if _TOOL is not None:
        return _TOOL
    if not available():
        return None

    from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
    from autogen_ext.tools.code_execution import PythonCodeExecutionTool

    _EXECUTOR = DockerCommandLineCodeExecutor(
        image=config.CODE_EXEC_IMAGE,
        timeout=config.CODE_EXEC_TIMEOUT,
        # Konteyner tur bitince değil, süreç bitince gidiyor.
        auto_remove=True,
        stop_container=True,
        work_dir=str(config.STATE / "codeexec"),
    )
    tool = PythonCodeExecutionTool(_EXECUTOR)
    # Tarif tool'un kendi varsayılanı ("Execute Python code blocks.") yerine
    # bizimki: rolü anlatan metin, şemanın parçası.
    tool._description = DESCRIPTION  # noqa: SLF001 — açık bir setter'ı yok
    _TOOL = tool
    return _TOOL


async def start() -> bool:
    """Konteyneri kaldır. Sunucunun `startup`'ında çağrılıyor."""
    if build_tool() is None:
        return False
    (config.STATE / "codeexec").mkdir(parents=True, exist_ok=True)
    await _EXECUTOR.start()
    return True


async def stop() -> None:
    """Konteyneri indir. Sunucunun `shutdown`'ında çağrılıyor."""
    if _EXECUTOR is None:
        return
    try:
        await _EXECUTOR.stop()
    except Exception:  # noqa: BLE001 — kapanış bir hatayı büyütmemeli
        pass


def make_gate_hook(gate: Any = None):
    """`before_tool_call` kancası — kodu adına göre değil, ne olduğuna göre.

    `"CodeExecutor"` hiçbir outbound markerına uymuyor (`send`, `post`, `write`,
    `delete`, `spawn`, `respond`, `approve`), yani `GATE.check()` onu sessizce
    geçirir. `openclaw_call`'daki sorunun birebir aynısı, ve çözümü de aynı:
    adı sormayan giriş noktası, `require()`.

    İmza `(tool, arguments)` üstünde olduğu için onay **kodun kendisine**
    bağlanıyor: kod değişirse eski onay tutmuyor. `docs/16 §2.2`'deki donmuş
    plan ilkesinin bedavaya gelen küçük hâli.

    Fabrika, çünkü kapı enjekte edilebilir olmalı — testler kendi kapılarını
    kullanıyor, ve tekile uzanan bir kanca isteği çağıranın göremeyeceği bir
    yere kaydeder.
    """

    def hook(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if str(payload.get("tool") or "") != TOOL_NAME:
                return {}

            from gateway import approval as approval_module

            target = gate or approval_module.GATE
            return target.require(
                TOOL_NAME,
                payload.get("arguments") or {},
                session=str(payload.get("session") or ""),
                reason=GATE_REASON,
            )
        except Exception as exc:  # noqa: BLE001
            # Ana kapıyla aynı asimetri: bozulan bir bekçi kapanır, açılmaz.
            return {"block": True,
                    "reason": f"code exec gate failed: {type(exc).__name__}: {exc}"}

    return hook


gate_hook = make_gate_hook()


def install_gate(registry=None, gate=None) -> None:
    """Kancayı `before_tool_call`'a tak, süsleyici kancalardan önce."""
    from gateway import hooks as hooks_module

    target = registry or hooks_module.REGISTRY
    target.unregister(hooks_module.BEFORE_TOOL_CALL, "code_exec_gate")
    target.register(
        hooks_module.BEFORE_TOOL_CALL,
        (make_gate_hook(gate) if gate is not None else gate_hook),
        name="code_exec_gate",
        order=-90,
    )


__all__ = [
    "DESCRIPTION", "GATE_REASON", "TOOL_NAME", "CodeExecUnavailable",
    "available", "build_tool", "gate_hook", "install_gate", "make_gate_hook",
    "start", "stop",
]


async def run_approved(code: str) -> dict[str, Any]:
    """Onaylanan kodu **olduğu gibi** çalıştır.

    Kapının reddi turu bitiriyor: ajan "reddedildim" diyor ve devam ediyor. Onay
    o turu geri getirmiyor, çünkü tur bitti. Geriye tek doğru şey kalıyor —
    operatörün onayladığı metni koşturmak.

    Ve **yeniden üretmek yerine saklananı** koşturmak bir zorunluluk, tercih
    değil: aynı soruyu iki kez sorduğumuzda model iki farklı program yazdı
    (ölçüldü: imzalar `029f4d1f…` ve `107fdfd1…`). Onaylanan kodla çalışan kodun
    aynı olmasının tek yolu, çalıştırılacak olanın onaylanan metin olması.
    `docs/16 §2.2`'nin donmuş planı tam olarak budur.
    """
    tool = build_tool()
    if tool is None:
        return {"ok": False, "output": "kod yürütme kapalı ya da Docker yok"}

    from autogen_core import CancellationToken
    from autogen_ext.tools.code_execution import CodeExecutionInput

    import time as _time

    started = _time.monotonic()
    try:
        result = await tool.run(CodeExecutionInput(code=code), CancellationToken())
    except Exception as exc:  # noqa: BLE001 — konteyner hatası sunucuyu düşürmemeli
        return {"ok": False, "output": f"{type(exc).__name__}: {exc}",
                "seconds": round(_time.monotonic() - started, 2)}
    return {"ok": bool(result.success), "output": result.output,
            "seconds": round(_time.monotonic() - started, 2)}
