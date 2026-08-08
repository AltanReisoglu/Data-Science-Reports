"""
langgraph_agent.py — LangGraph ajanı (POC'un VARSAYILAN motoru, generic-canlı yolda).

Ajan döngüsünü LangGraph sürer (reason→act→observe + tool yürütme); biz observe ayağını
`pre_model_hook` ile sıkıştırırız. `strategies/` katmanı framework-agnostik olduğu için
13 stratejinin hiçbiri değişmeden takılır.

İki kullanım:
  1. LangGraphAgent  — chat.py / demo_server.py'nin sürdüğü OTURUM sınıfı (çok-turlu,
     .conv/.strategy/.budget/.set_strategy/.send arayüzü ToolUseAgent ile aynı).
  2. main()          — tek dosyalık standalone demo.

Kanca: create_react_agent(model, tools, pre_model_hook=CompactionHook). pre_model_hook
her LLM çağrısından önce `llm_input_messages` döndürür (state'e yazılmaz) → ham tool sonucu
graf state'inde kalır, modele giden kopya sıkışır, tool_call_id korunur (yoksa API 400).

  pip install langgraph langchain-openai   (zaten kurulu)
  python langgraph_agent.py --strategy ours
"""
from __future__ import annotations

import os
import sys
import itertools

import llm  # .env'i yükler (LLM_BASE_URL/KEY/MODEL) — anahtar yazdırılmaz
from harness import Conversation, ToolResult, MockTools, est
from providers import GenericProvider
from strategies import get as get_strategy

try:
    import warnings
    warnings.filterwarnings("ignore", message=".*create_react_agent.*")
    from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                          ToolMessage)
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    HAVE_LANGGRAPH = True
except ImportError:
    HAVE_LANGGRAPH = False

_THREADS = itertools.count(1)


# --------------------------------------------------------------------------
# messages[] ↔ bizim ToolResult çevirisi (hook ve oturum ortak kullanır).
# --------------------------------------------------------------------------
def _extract_results(messages, provider):
    """LangGraph mesaj listesinden Conversation + {tool_call_id: ToolResult} çıkar."""
    call_args = {}
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            call_args[tc["id"]] = (tc["name"], tc.get("args", {}))
    conv = Conversation()
    by_id = {}
    turn_no = 0
    cur = None
    for m in messages:
        if isinstance(m, HumanMessage):
            turn_no += 1
            cur = conv.new_turn(m.content if isinstance(m.content, str) else str(m.content))
        elif isinstance(m, ToolMessage):
            if cur is None:
                cur = conv.new_turn("")
                turn_no = max(turn_no, 1)
            name, args = call_args.get(m.tool_call_id, (m.name or "", {}))
            content = m.content if isinstance(m.content, str) else str(m.content)
            tr = ToolResult(call_id=m.tool_call_id, name=name,
                            tool_type=provider.tool_type(name),
                            resource=provider.resource_of(name, args),
                            content=content, turn=turn_no, args=args)
            cur.results.append(tr)
            by_id[m.tool_call_id] = tr
    return conv, by_id


# --------------------------------------------------------------------------
# ADAPTÖR — bizim Strategy'yi LangGraph pre_model_hook'una çevirir.
# strateji/bütçe DİNAMİK okunur (oturumda set_strategy ile değişebilir).
# --------------------------------------------------------------------------
class CompactionHook:
    def __init__(self, provider, get_strat, get_budget, verbose=False):
        self.provider = provider
        self.get_strat = get_strat      # callable → aktif Strategy
        self.get_budget = get_budget    # callable → aktif bütçe
        self.verbose = verbose

    def __call__(self, state) -> dict:
        msgs = state["messages"]
        conv, by_id = _extract_results(msgs, self.provider)
        if not by_id:
            return {}
        conv.reset_fates()
        preamble = self.get_strat().compact(conv.all_results(), conv, self.get_budget())
        out = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                tr = by_id.get(m.tool_call_id)
                shown = tr.shown() if tr else (m.content if isinstance(m.content, str) else str(m.content))
                if shown == "":
                    shown = "[içerik compaction ile özete taşındı]"
                out.append(ToolMessage(content=shown, tool_call_id=m.tool_call_id, name=m.name))
            else:
                out.append(m)
        if preamble:
            out.insert(1, SystemMessage(content="[compaction özeti]\n" + preamble))
        if self.verbose:
            raw = conv.raw_tokens(); shown_tok = conv.shown_tokens(preamble)
            print(f"    ┄ [{self.get_strat().name}] {len(by_id)} tool · ham {raw}→{shown_tok} tok "
                  f"(%{round(100*(raw-shown_tok)/raw) if raw else 0})")
        return {"llm_input_messages": out}


# --------------------------------------------------------------------------
# generic tool'lar — MockTools'u LangChain @tool olarak sar (tek MockTools = sürüm kalıcı).
# --------------------------------------------------------------------------
def _build_tools(mt=None):
    mt = mt or MockTools()

    @tool
    def run_terminal(cmd: str) -> str:
        """Bir kabuk komutu çalıştır (npm test/build)."""
        return mt.terminal(cmd)

    @tool
    def read_file(path: str) -> str:
        """Bir kaynak dosyayı oku."""
        return mt.read_file(path)

    @tool
    def web_extract(url: str) -> str:
        """Bir web sayfasının metnini çıkar."""
        return mt.web_extract(url)

    @tool
    def take_snapshot(page: str) -> str:
        """Bir tarayıcı sayfasının accessibility snapshot'ını al."""
        return mt.take_snapshot(page)

    @tool
    def grep(query: str) -> str:
        """Kod tabanında desen ara."""
        return mt.grep(query)

    @tool
    def write_file(path: str, content: str = "") -> str:
        """Bir dosyaya yaz/düzenle (mutasyon)."""
        return mt.write_file(path)

    return [run_terminal, read_file, web_extract, take_snapshot, grep, write_file]


def _make_model():
    return ChatOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"),
                      model=os.getenv("LLM_MODEL_NAME"), temperature=0.2)


# --------------------------------------------------------------------------
# LangGraphAgent — chat.py / demo_server.py'nin sürdüğü OTURUM sınıfı.
# ToolUseAgent ile AYNI arayüz: conv, strategy, budget, last_preamble, set_strategy, send.
# --------------------------------------------------------------------------
class LangGraphAgent:
    is_live_agent = True
    engine = "langgraph"

    def __init__(self, strategy, budget: int = 1500, verbose: bool = False, provider=None):
        if not HAVE_LANGGRAPH:
            raise RuntimeError("langgraph kurulu değil (pip install langgraph langchain-openai)")
        self.strategy = strategy
        self.budget = budget
        self.provider = provider or GenericProvider()  # generic (LangGraph yolu)
        self.conv = Conversation()
        self.last_preamble = ""
        self._mt = MockTools()
        hook = CompactionHook(self.provider, lambda: self.strategy, lambda: self.budget, verbose)
        self.app = create_react_agent(_make_model(), tools=_build_tools(self._mt),
                                      pre_model_hook=hook, checkpointer=MemorySaver())
        self._config = {"configurable": {"thread_id": f"poc-{next(_THREADS)}"}}

    def set_strategy(self, strategy):
        self.strategy = strategy

    def send(self, user: str) -> dict:
        result = self.app.invoke({"messages": [HumanMessage(content=user)]}, self._config)
        messages = result["messages"]
        # tüm geçmişten trace'i yeniden kur + aktif stratejiyle sıkıştır (gösterim için)
        self.conv, _ = _extract_results(messages, self.provider)
        self.conv.reset_fates()
        self.last_preamble = self.strategy.compact(self.conv.all_results(), self.conv, self.budget)
        answer = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                c = m.content
                answer = c if isinstance(c, str) else str(c)
                if answer.strip():
                    break
        return self._summary(answer)

    def _summary(self, answer: str) -> dict:
        results = self.conv.all_results()
        fates: dict[str, int] = {}
        for r in results:
            fates[r.fate] = fates.get(r.fate, 0) + 1
        raw = self.conv.raw_tokens(); shown = self.conv.shown_tokens(self.last_preamble)
        return {"answer": answer or "(model yanıt üretmedi)", "raw_tokens": raw,
                "shown_tokens": shown, "saved_pct": round(100*(raw-shown)/raw) if raw else 0,
                "units": len(results), "fates": fates, "preamble": self.last_preamble}


# --------------------------------------------------------------------------
# standalone demo.
# --------------------------------------------------------------------------
def build_agent(strategy_name: str, budget: int = 1500):
    """Tek atışlık standalone ajan (main için)."""
    strat = get_strategy(strategy_name)
    hook = CompactionHook(GenericProvider(), lambda: strat, lambda: budget, verbose=True)
    return create_react_agent(_make_model(), tools=_build_tools(),
                              pre_model_hook=hook, checkpointer=MemorySaver())


DEMO = ["src/server.py dosyasını oku ve npm test çalıştır, sonra src/server.py'yi tekrar oku"]


def main() -> None:
    if not HAVE_LANGGRAPH:
        print("LangGraph kurulu değil. Kurulum:\n    pip install langgraph langchain-openai")
        return
    if not llm.available():
        print(f"LLM yapılandırılmadı ({llm.why_unavailable()}). .env'de LLM_* dolu mu?")
        return
    argv = sys.argv[1:]
    name = argv[argv.index("--strategy") + 1] if "--strategy" in argv else "ours"
    agent = LangGraphAgent(get_strategy(name), verbose=False)
    print(f"LangGraph create_react_agent · pre_model_hook=CompactionHook({name})")
    print("(framework döngüyü sürüyor; observe ayağını bizim strateji sıkıştırıyor)\n")
    for msg in DEMO:
        print(f"sen › {msg}")
        out = agent.send(msg)
        fates = ", ".join(f"{k}×{v}" for k, v in out["fates"].items() if k != "TAM")
        print(f"    ┄ ham {out['raw_tokens']}→{out['shown_tokens']} tok "
              f"(%{out['saved_pct']}) · {fates or '—'}")
        print(f"asistan › {out['answer']}\n")


if __name__ == "__main__":
    main()
