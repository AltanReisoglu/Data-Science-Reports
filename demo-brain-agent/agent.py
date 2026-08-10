#!/usr/bin/env python3
"""
agent.py — İŞÇİ AJAN: LangGraph döngüsü + GERÇEK LLM + seçilebilir tool-trace compaction.

Bu dosya SADECE ajan döngüsüdür. Task yönetimi burada DEĞİL:

    ┌─ TASK MANAGEMENT ──────────────────────────────────────────────────┐
    │  taskboard.py   : ajanın ÜRETTİĞİ task'ların durable board'u       │
    │  orchestrator.py: PLANLAMA (ajan task üretir) → DISPATCH (motor)   │
    └────────────────────────────────────────────────────────────────────┘
                      ↓ her task için bir işçi ajan koşar
    ┌─ AJAN DÖNGÜSÜ (bu dosya, LangGraph StateGraph) ────────────────────┐
    │  reason (LLM) ──tool_calls?──► act (tool'ları koştur) ──► compact ─┐│
    │     ▲                                                             ││
    │     └─────────────────────────────────────────────────────────────┘│
    │  tool_calls yoksa ──► END (nihai yanıt)                            │
    └────────────────────────────────────────────────────────────────────┘
                      ↓ her tool sonucundan sonra
    ┌─ TOOL-TRACE COMPACTION (compaction.py) ────────────────────────────┐
    │  none | hermes | opencode | openclaw | codex | claude_code         │
    └────────────────────────────────────────────────────────────────────┘

NOT: Önceki sürümde burada "ajan koşusunun tamamı = 1 task" varsayan bir
task-management sarmalayıcısı vardı (taskmgmt.py). O model YANLIŞTI — task'ları
ajan üretmeli — ve sistemden kaldırıldı. Task yönetimi için orchestrator.py'ye bak.

Çalıştır (tek ajan koşusu, task üretimi yok — yalnız compaction ölçümü):
    python3 agent.py --strategy hermes --budget 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "poc"))

import compaction  # noqa: E402
from compaction import STRATEGIES, STRATEGY_INFO  # noqa: E402

try:
    import llm  # poc/llm.py — gerçek LLM (anahtar .env'de, asla yazdırılmaz)
except Exception:
    llm = None


# ═════════════════════════ TOOL'LAR (bilerek BÜYÜK çıktı üretirler) ═════════════════════════

def _fake_file(path: str, n: int = 420) -> str:
    """Gerçekçi büyüklükte sahte kaynak dosya (compaction'ın işe yaraması için)."""
    head = f"# {path}\n# -*- coding: utf-8 -*-\nimport os, sys, json, logging\n\n"
    body = "\n".join(
        f"def handler_{i}(request, ctx=None):\n"
        f"    \"\"\"İş kuralı #{i} — doğrulama, dönüşüm ve kayıt.\"\"\"\n"
        f"    payload = request.get('data', {{}})\n"
        f"    if not payload: raise ValueError('boş istek #{i}')\n"
        f"    return {{'ok': True, 'step': {i}, 'path': {path!r}}}\n"
        for i in range(n))
    return head + body


_FILES = {
    "auth/login.py": _fake_file("auth/login.py", 380) + (
        "\n\ndef login(user, password, mfa_token=None):\n"
        "    \"\"\"BUG BURADA: mfa_token None ise kontrol atlanıyor.\"\"\"\n"
        "    if mfa_token is None:\n"
        "        return {'ok': True, 'user': user}   # ← güvenlik açığı: MFA baypas\n"
        "    return verify(user, password, mfa_token)\n"),
    "auth/session.py": _fake_file("auth/session.py", 300),
    "api/routes.py": _fake_file("api/routes.py", 460),
    "db/models.py": _fake_file("db/models.py", 350),
}


def tool_read_file(path: str) -> str:
    """Bir kaynak dosyanın TAMAMINI okur (büyük çıktı)."""
    if path in _FILES:
        return _FILES[path]
    return _fake_file(path, 260)


def tool_search_code(query: str) -> str:
    """Kod tabanında arama yapar (çok sayıda eşleşme döner)."""
    lines = []
    for f in _FILES:
        for i in range(55):
            lines.append(f"{f}:{i*7+3}: ... {query} ... çağrısı bulundu "
                         f"(bağlam: handler_{i}, modül={f.split('/')[0]})")
    return f"'{query}' için {len(lines)} eşleşme:\n" + "\n".join(lines)


def tool_run_tests(suite: str = "all") -> str:
    """Test paketini koşturur (uzun konsol çıktısı)."""
    out = [f"pytest {suite} — toplama başladı"]
    for i in range(240):
        st = "FAILED" if i in (37, 118) else "PASSED"
        out.append(f"tests/test_module_{i//20}.py::test_case_{i} ... {st} "
                   f"[{i*100//240}%] ({(i%9)+1}ms)")
    out += ["", "=== 2 failed, 238 passed in 12.4s ===",
            "FAILED tests/test_module_1.py::test_case_37 - AssertionError: MFA baypas edilebiliyor",
            "FAILED tests/test_module_5.py::test_case_118 - AssertionError: oturum süresi dolmuyor"]
    return "\n".join(out)


def tool_fetch_docs(topic: str) -> str:
    """Dahili dokümantasyonu çeker (çok büyük metin)."""
    secs = []
    for i in range(70):
        secs.append(f"## {topic} — bölüm {i}\n" + " ".join(
            [f"{topic} politikası maddesi {i}.{j}: sistem davranışı ve gerekçesi."
             for j in range(14)]))
    return f"# {topic} dokümantasyonu\n\n" + "\n\n".join(secs)


TOOLS = {
    "read_file": (tool_read_file, "Bir kaynak dosyanın tamamını okur.",
                  {"path": {"type": "string", "description": "dosya yolu"}}, ["path"]),
    "search_code": (tool_search_code, "Kod tabanında metin arar.",
                    {"query": {"type": "string", "description": "aranacak ifade"}}, ["query"]),
    "run_tests": (tool_run_tests, "Test paketini koşturur.",
                  {"suite": {"type": "string", "description": "paket adı (varsayılan: all)"}}, []),
    "fetch_docs": (tool_fetch_docs, "Dahili dokümantasyonu çeker.",
                   {"topic": {"type": "string", "description": "konu"}}, ["topic"]),
}


# Çalışma anında EKLENEN tool'lar (orchestrator burayı doldurur).
# Buranın asıl amacı: işçi ajana `create_task` vermek → ajan iş yaparken YENİ İŞ keşfedip
# task açabilsin (replanlama). Yani task üretimi de bir TOOL ÇAĞRISIdır; sadece
# "iş tool'u" değil "task-yönetim tool'u"dur.
EXTRA_TOOLS: dict = {}


def _all_tools() -> dict:
    return {**TOOLS, **EXTRA_TOOLS}


def _openai_tool_specs() -> list:
    specs = []
    for name, (_fn, desc, props, req) in _all_tools().items():
        specs.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    return specs


# ═════════════════════════ LANGGRAPH AJAN DÖNGÜSÜ ═════════════════════════

SYSTEM_PROMPT = (
    "Sen bir kod-inceleme ajanısın. Kullanıcının hedefine ulaşmak için sana verilen "
    "tool'ları kullan. Tool çıktıları ÇOK BÜYÜK olabilir — özetlenmiş/kırpılmış içerik "
    "görebilirsin, bu normaldir, elindeki bilgiyle ilerle. Yeterli bulgu topladığında "
    "tool çağırmayı bırak ve NET bir sonuç yaz: kök neden + kanıt + öneri. "
    "En fazla 4 tool çağrısı yap."
)


class AgentState(TypedDict):
    messages: list          # ham dict mesajlar (JSON-serileşebilir)
    strategy: str
    budget: int
    trace: list             # bu adımda ne olduğu (UI/log için)
    compaction_events: list
    finished: bool
    answer: str


def _lc_available() -> bool:
    try:
        import langgraph  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
        return True
    except Exception:
        return False


def _call_llm(messages: list) -> dict:
    """GERÇEK LLM çağrısı (OpenAI-uyumlu, tool-calling ile). Erişim yoksa hata verir."""
    if llm is None or not llm.available():
        raise RuntimeError("LLM erişimi yok (.env: LLM_BASE_URL/LLM_API_KEY/LLM_MODEL_NAME)")
    r = llm.chat(messages, tools=_openai_tool_specs(), max_tokens=600, temperature=0.2)
    return r


def n_reason(state: AgentState) -> dict:
    """Düğüm 1 — REASON: LLM'e sor, tool çağrısı ya da nihai yanıt al."""
    msgs = state["messages"]
    r = _call_llm(msgs)
    content = (r.get("content") or "").strip()
    tcs = r.get("tool_calls") or []
    ai_msg = {"role": "assistant", "content": content, "tool_calls": tcs}
    trace = list(state.get("trace", []))
    if tcs:
        names = ", ".join(tc["function"]["name"] for tc in tcs)
        trace.append(f"REASON → LLM {len(tcs)} tool çağırdı: {names}")
        return {"messages": msgs + [ai_msg], "trace": trace, "finished": False}
    trace.append(f"REASON → LLM nihai yanıtı üretti ({compaction.est(content):,}t)")
    return {"messages": msgs + [ai_msg], "trace": trace,
            "finished": True, "answer": content}


def n_act(state: AgentState) -> dict:
    """Düğüm 2 — ACT: tool'ları GERÇEKTEN koştur, sonuçları mesajlara ekle."""
    msgs = list(state["messages"])
    trace = list(state.get("trace", []))
    last = msgs[-1]
    for tc in last.get("tool_calls", []):
        fname = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        fn = _all_tools().get(fname, (None,))[0]
        if fn is None:
            out = f"[hata] bilinmeyen tool: {fname}"
        else:
            try:
                out = fn(**args)
            except Exception as e:
                out = f"[hata] {fname}: {e}"
        msgs.append({"role": "tool", "name": fname,
                     "tool_call_id": tc.get("id", ""), "content": out})
        trace.append(f"ACT → {fname}({', '.join(f'{k}={v!r}' for k, v in args.items())}) "
                     f"→ {compaction.est(out):,} token HAM çıktı")
    return {"messages": msgs, "trace": trace}


def n_compact(state: AgentState) -> dict:
    """Düğüm 3 — COMPACT: seçilen tool-trace stratejisini uygula."""
    res = compaction.compact(state["strategy"], state["messages"], budget=state["budget"])
    trace = list(state.get("trace", []))
    events = list(state.get("compaction_events", []))
    trace.append(f"COMPACT [{res.strategy}] {res.summary_line()}")
    events.append({"strategy": res.strategy, "before": res.before, "after": res.after,
                   "saved": res.saved, "pct": round(res.pct, 1),
                   "triggered": res.triggered, "log": res.log[:8]})
    return {"messages": res.messages, "trace": trace, "compaction_events": events}


def _route(state: AgentState) -> str:
    return "act" if not state.get("finished") else "__end__"


def build_graph(interrupt: bool = True):
    """LangGraph StateGraph: reason → (act → compact → reason) | END."""
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    g = StateGraph(AgentState)
    g.add_node("reason", n_reason)
    g.add_node("act", n_act)
    g.add_node("compact", n_compact)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _route, {"act": "act", "__end__": END})
    g.add_edge("act", "compact")
    g.add_edge("compact", "reason")
    kw = {"checkpointer": MemorySaver()}
    if interrupt:
        kw["interrupt_after"] = ["compact"]      # her tool turu = checkpoint sınırı
    return g.compile(**kw)


# ═════════════════════════ ADIMLANABİLİR İŞÇİ AJAN ═════════════════════════

class BrainAgentJob:
    """TEK bir task'ı yürüten işçi ajan — tur tur ilerletilebilir ve checkpoint'lenebilir.

    orchestrator.execute_task() bunu kullanır: board'dan gelen bir task'ın
    başlığı/gövdesi hedef olur, ajan turları koşar, state checkpoint'lenir.

    Sözleşme:
        initial_state() -> dict            (JSON-serileşebilir checkpoint)
        step(state, attempt, fail_at) -> (done, state, label)   # 1 tur ilerlet
    """

    def __init__(self, goal: str, strategy: str = "hermes", budget: int = 3_000,
                 job_id: str = "brain-1", max_turns: int = 5):
        self.goal = goal
        self.strategy = strategy
        self.budget = budget
        self.job_id = job_id
        self.max_turns = max_turns

    def spec(self) -> dict:
        return {"goal": self.goal, "strategy": self.strategy,
                "budget": self.budget, "max_turns": self.max_turns}

    def initial_state(self) -> dict:
        return {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": self.goal}],
            "done_steps": [],
            "trace": [],
            "compaction_events": [],
            "answer": "",
            "turn": 0,
        }

    def step(self, state: dict, attempt: int = 0, fail_at: str | None = None):
        """TEK ajan turu ilerlet: reason → (act → compact). Checkpoint'lenebilir.

        fail_at: bu tool adında GEÇİCİ hata üret (retry yolunu tetiklemek için).
        """
        state = json.loads(json.dumps(state, ensure_ascii=False, default=str))
        turn = state.get("turn", 0)
        if turn >= self.max_turns:
            # Tur sınırı: tool KULLANMADAN son bir kez sonucu yazdır (boş yanıt dönmesin)
            msgs = state["messages"] + [{
                "role": "user",
                "content": ("Tool bütçen bitti. Şu ana kadarki bulgularınla ŞİMDİ nihai sonucu yaz: "
                            "kök neden + kanıt + öneri. Yeni tool çağırma.")}]
            try:
                r = llm.chat(msgs, max_tokens=600, temperature=0.2) if (llm and llm.available()) else {}
                final = (r.get("content") or "").strip()
            except Exception as e:
                final = f"[sonuç üretilemedi: {e}]"
            state["messages"] = msgs + [{"role": "assistant", "content": final}]
            state["answer"] = final or "[tur sınırına ulaşıldı]"
            state["trace"] = state.get("trace", []) + [
                f"REASON (tur sınırı) → tool'suz nihai yanıt ({compaction.est(final):,}t)"]
            state.setdefault("done_steps", []).append(f"turn{turn}:final")
            state["turn"] = turn + 1
            return True, state, f"tur sınırı ({self.max_turns}) → nihai yanıt ✓"

        # --- REASON ---
        st: AgentState = {"messages": state["messages"], "strategy": self.strategy,
                          "budget": self.budget, "trace": [],
                          "compaction_events": [], "finished": False, "answer": ""}
        upd = n_reason(st)
        st.update(upd)

        if st.get("finished"):
            state["messages"] = st["messages"]
            state["trace"] = state.get("trace", []) + st["trace"]
            state["answer"] = st.get("answer", "")
            state["turn"] = turn + 1
            state.setdefault("done_steps", []).append(f"turn{turn}:final")
            return True, state, "REASON → nihai yanıt ✓"

        # hangi tool'lar çağrılacak?
        called = [tc["function"]["name"] for tc in st["messages"][-1].get("tool_calls", [])]
        label_tools = ",".join(called) or "?"

        # --- geçici hata enjeksiyonu (retry yolunu göstermek için) ---
        if fail_at and fail_at in called and attempt == 0:
            raise RuntimeError(f"geçici hata: {fail_at} tool'u zaman aşımına uğradı")

        # --- ACT + COMPACT ---
        st.update(n_act(st))
        st.update(n_compact(st))

        state["messages"] = st["messages"]
        state["trace"] = state.get("trace", []) + st["trace"]
        state["compaction_events"] = state.get("compaction_events", []) + st["compaction_events"]
        state["turn"] = turn + 1
        for t in called:
            state.setdefault("done_steps", []).append(f"turn{turn}:{t}")
        return False, state, f"turn{turn}: {label_tools} → koştu + compact"

    # --- doğrudan (task-mgmt'siz) koşu: saf LangGraph ---
    def run_sync(self, goal: str | None = None) -> dict:
        """LangGraph grafiğini uçtan uca koştur (task-management katmanı olmadan)."""
        graph = build_graph(interrupt=False)
        init: AgentState = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": goal or self.goal}],
            "strategy": self.strategy, "budget": self.budget,
            "trace": [], "compaction_events": [], "finished": False, "answer": "",
        }
        out = graph.invoke(init, {"recursion_limit": self.max_turns * 3 + 5,
                                  "configurable": {"thread_id": self.job_id}})
        return {"answer": out.get("answer", ""), "trace": out.get("trace", []),
                "compaction_events": out.get("compaction_events", [])}


def build_job(spec: dict, job_id: str = "brain-1") -> BrainAgentJob:
    """Süreçler arası (Celery worker gibi) işi yeniden kurmak için fabrika."""
    return BrainAgentJob(goal=spec.get("goal", ""),
                         strategy=spec.get("strategy", "hermes"),
                         budget=int(spec.get("budget", 3_000)),
                         job_id=job_id,
                         max_turns=int(spec.get("max_turns", 5)))


# ═════════════════════════ CLI ═════════════════════════

DEFAULT_GOAL = ("auth/login.py dosyasındaki oturum açma akışını incele, testleri koştur ve "
                "MFA ile ilgili güvenlik hatasını kök nedeniyle birlikte raporla.")


def _emit_json(a):
    """Web UI için tek seferde yapılandırılmış sonuç üret (stdout'a JSON)."""
    import time as _t
    t0 = _t.time()
    job = BrainAgentJob(goal=a.goal, strategy=a.strategy, budget=a.budget,
                        max_turns=a.max_turns)
    out = {"strategy": a.strategy, "budget": a.budget,
           "llm": (llm.MODEL if (llm and llm.available()) else None),
           "goal": a.goal, "ok": False, "trace": [], "compaction_events": [],
           "answer": "", "error": None}
    try:
        r = job.run_sync()
        out["trace"] = r["trace"]
        out["compaction_events"] = r["compaction_events"]
        out["answer"] = r["answer"]
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    ce = out["compaction_events"]
    if ce:
        out["totals"] = {
            "ham": sum(e["before"] for e in ce),
            "kalan": sum(e["after"] for e in ce),
            "kazanc": sum(e["saved"] for e in ce),
        }
    out["seconds"] = round(_t.time() - t0, 1)
    print("<<<JSON>>>" + json.dumps(out, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser(
        description="işçi ajan — LangGraph döngüsü + tool-trace compaction "
                    "(task yönetimi için orchestrator.py)")
    ap.add_argument("--strategy", default="hermes", choices=list(STRATEGIES),
                    help="tool-trace compaction stratejisi")
    ap.add_argument("--goal", default=DEFAULT_GOAL)
    ap.add_argument("--budget", type=int, default=3_000, help="context token bütçesi")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="yapılandırılmış JSON çıktı (web UI için)")
    a = ap.parse_args()

    if a.json:
        _emit_json(a)
        return

    print("=" * 84)
    print("İŞÇİ AJAN — LangGraph döngüsü + seçilebilir tool-trace compaction")
    print("  (task üretimi/yönetimi burada DEĞİL → orchestrator.py)")
    print("=" * 84)
    mode = "GERÇEK LLM" if (llm and llm.available()) else "LLM YOK"
    print(f"  LLM      : {mode}" + (f" ({llm.MODEL})" if llm and llm.available() else ""))
    print(f"  strateji : {a.strategy} — {STRATEGY_INFO[a.strategy]['ozet']}")
    print(f"  bütçe    : {a.budget:,} token")
    print("-" * 84)

    job = BrainAgentJob(goal=a.goal, strategy=a.strategy, budget=a.budget,
                        max_turns=a.max_turns)
    out = job.run_sync()
    print("\nAJAN İZİ:")
    for t in out["trace"]:
        print("  " + t)
    print("\nCOMPACTION OLAYLARI:")
    for e in out["compaction_events"]:
        print(f"  {e['strategy']}: {e['before']:,} → {e['after']:,} "
              f"(−{e['saved']:,} · %{e['pct']})")
    print("\nYANIT:\n" + (out["answer"] or "(boş)")[:1500])


if __name__ == "__main__":
    main()
