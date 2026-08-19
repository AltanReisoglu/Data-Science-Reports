"""Rakip çerçeveleri **kurup** ölç — okuyup değil.

`docs/09` altı çerçeveyi karşılaştırıyor ama mimari iddiaların çoğu
`[teyitsiz]`: okundu, koşturulmadı. Bu betik o etiketi kaldırıyor. Her sayı
kurulu paketten okunuyor, ve hangi sembolden geldiği raporda yazıyor.

### Neden bu kadar önemli

Sunumda "LangGraph'ta şu yok" demek, LangGraph'ı kurmadan söylenirse bir
tahmindir. Ve ilk denememde **yanlış çıktı**: CrewAI'da kod yürütmeyi
`crewai_tools` altında aradım, bulamadım, "yok" yazacaktım — oysa
`Agent.allow_code_execution` diye duruyor. Bu yüzden bulunamayan her şey
"yok" değil **"bu adlarla yok"** diye raporlanıyor; ad tahmininin yanlış olma
ihtimali görünür kalmalı.

### Kullanım

    python docs/tools/olc_cerceveler.py --kur      # sanal ortamları kur
    python docs/tools/olc_cerceveler.py            # ölç ve JSON bas

Ortamlar `--work` altında (varsayılan `/tmp/vc-cerceveler`) ve bir kez
kuruluyor. Bu depodaki `.venv`'e hiçbiri karışmıyor: `agent-framework` ile
`autogen-*`'ın aynı ortama girmemesi bunun ilk dersiydi.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

# Paket adı → (sanal ortam adı, içe aktarılacak modül)
CERCEVELER = {
    "langgraph": ("lg", "langgraph"),
    "crewai": ("crewai", "crewai"),
    "openai-agents": ("openai", "agents"),
    "google-adk": ("google", "google.adk.agents"),
}

# Sürüm hızı için PyPI'dan sorulacaklar — kuruluma gerek yok.
HIZ = ["autogen-agentchat", "agent-framework", "langgraph", "crewai",
       "openai-agents", "google-adk", "semantic-kernel", "ag2"]

# Döngü tavanı: her çerçevede başka bir adla ve başka bir yerde duruyor.
# Kaynağı da yazılıyor, çünkü "25" tek başına doğrulanamaz bir sayı.
DONGU = {
    "langgraph": ("from langgraph._internal._config import DEFAULT_RECURSION_LIMIT as D; "
                  "print(D)", "recursion_limit"),
    "crewai": ("from crewai import Agent; print(Agent.model_fields['max_iter'].default)",
               "Agent.max_iter"),
    "openai-agents": ("import agents.run as r; print(r.DEFAULT_MAX_TURNS)",
                      "Runner.run(max_turns=)"),
    "google-adk": ("from google.adk.agents import LoopAgent; "
                   "print(LoopAgent.model_fields['max_iterations'].default)",
                   "LoopAgent.max_iterations"),
}

# Bir mekanizma var mı: modül + aranacak adlar. Ad listesi bilerek geniş —
# tek ada bakmak, farklı isimlendirmeyi "yok" diye raporlar.
YETENEK = {
    "langgraph": {
        "checkpoint": ("langgraph.checkpoint.memory", ["InMemorySaver", "MemorySaver"]),
        "insan döngüde": ("langgraph.types", ["interrupt", "Command"]),
        "bellek": ("langgraph.store.memory", ["InMemoryStore"]),
        "graf kurucu": ("langgraph.graph", ["StateGraph"]),
        "dağıtık runtime": ("langgraph.graph", ["GrpcRuntime", "DistributedRuntime"]),
    },
    "crewai": {
        "ajan/görev/ekip": ("crewai", ["Crew"]),
        "akış (flow)": ("crewai.flow.flow", ["Flow"]),
        "kod yürütücü": ("crewai", ["Agent"]),   # alan kontrolü aşağıda
        "dağıtık runtime": ("crewai", ["GrpcRuntime", "DistributedRuntime"]),
    },
    "openai-agents": {
        "devir (handoff)": ("agents", ["handoff"]),
        "korkuluk": ("agents", ["input_guardrail"]),
        "oturum": ("agents", ["SQLiteSession"]),
        "izleme": ("agents", ["trace"]),
        "dağıtık runtime": ("agents", ["GrpcRuntime", "DistributedRuntime"]),
    },
    "google-adk": {
        "akış tipleri": ("google.adk.agents", ["SequentialAgent", "ParallelAgent"]),
        "oturum": ("google.adk.sessions", ["InMemorySessionService"]),
        "bellek": ("google.adk.memory", ["InMemoryMemoryService"]),
        "kod yürütücü": ("google.adk.code_executors", ["BuiltInCodeExecutor"]),
        "dağıtık runtime": ("google.adk.agents", ["GrpcRuntime", "DistributedRuntime"]),
    },
}


def pypi(pkg: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=25) as r:
        d = json.load(r)
    v = d["info"]["version"]
    rel = d["releases"].get(v) or []
    up = rel[0]["upload_time"][:10] if rel else None
    age = (date.today() - date.fromisoformat(up)).days if up else None
    return {"sürüm": v, "tarih": up, "yaş_gün": age, "sürüm_sayısı": len(d["releases"])}


def run(py: Path, code: str) -> str | None:
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=180)
    return r.stdout.strip() if r.returncode == 0 else None


def kur(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for pkg, (venv, _) in CERCEVELER.items():
        d = work / f".venv-{venv}"
        if (d / "bin" / "python").exists():
            print(f"  {pkg}: zaten kurulu")
            continue
        print(f"  {pkg}: kuruluyor…", flush=True)
        subprocess.run(["uv", "venv", "-q", str(d)], check=True, timeout=200)
        subprocess.run(["uv", "pip", "install", "-q", "--python",
                        str(d / "bin" / "python"), pkg], check=True, timeout=900)


def olc(work: Path) -> dict:
    out: dict = {"çekildi": date.today().isoformat(), "hız": {}, "çerçeveler": {}}
    for p in HIZ:
        try:
            out["hız"][p] = pypi(p)
        except Exception as e:  # noqa: BLE001 — ağ yoksa ölçüm eksik kalsın, çökmesin
            out["hız"][p] = {"hata": type(e).__name__}

    for pkg, (venv, mod) in CERCEVELER.items():
        py = work / f".venv-{venv}" / "bin" / "python"
        if not py.exists():
            out["çerçeveler"][pkg] = {"hata": "kurulu değil — önce --kur"}
            continue
        row: dict = {"sürüm": run(py, f"import importlib.metadata as m;print(m.version({pkg!r}))")}

        code, kaynak = DONGU[pkg]
        row["döngü_tavanı"] = {"değer": run(py, code), "kaynak": kaynak}

        yet = {}
        for soru, (m, names) in YETENEK[pkg].items():
            expr = (f"import importlib;m=importlib.import_module({m!r});"
                    f"print(next((n for n in {names!r} if hasattr(m,n)),''))")
            hit = run(py, expr) or ""
            yet[soru] = hit or "bu adlarla yok"
        # CrewAI'da kod yürütme bir sınıf değil bir ALAN; ilk denememde
        # `crewai_tools`'ta arayıp "yok" bulmuştum — yanlıştı.
        if pkg == "crewai":
            v = run(py, "from crewai import Agent;"
                        "print(Agent.model_fields['allow_code_execution'].default,"
                        "Agent.model_fields['code_execution_mode'].default)")
            yet["kod yürütücü"] = f"Agent.allow_code_execution={v}" if v else "bu adlarla yok"
        row["yetenekler"] = yet
        out["çerçeveler"][pkg] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kur", action="store_true")
    ap.add_argument("--work", default="/tmp/vc-cerceveler")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    w = Path(a.work)
    if a.kur:
        kur(w)
        sys.exit(0)
    data = olc(w)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"{a.out} yazıldı")
    else:
        print(text)
