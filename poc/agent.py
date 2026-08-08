"""
agent.py — GERÇEK tool-use ajanı (bizim internal LLM) + compaction köprüsü.

Manuel tool-calling döngüsü (framework yok — Anthropic'in "manual loop"unun OpenAI-uyumlu
muadili). Model tool'ları KENDİ seçer; sonuçlar SEÇİLİ compaction stratejisinden GEÇMİŞ
hâliyle modele geri döner. Köprü kilit: `_render_messages` her tool mesajının GÖVDESİNİ
kaderine göre yeniden yazar (özet/maske/gizle/sil) ama `tool_call_id` eşleşmesini ASLA
bozmaz — yoksa OpenAI-uyumlu API 400 verir. Böylece "trace compaction" gerçekten modelin
gördüğü bağlamı küçültür.

ChatSession ile aynı arayüzü sunar (conv, strategy, budget, last_preamble, set_strategy,
send) → chat.py ikisini de sürebilir.
"""
from __future__ import annotations

import json

import llm
from harness import Conversation, ToolResult, est
from providers import GenericProvider

MAX_STEPS = 8


class ToolUseAgent:
    is_live_agent = True
    engine = "manuel"

    def __init__(self, strategy, budget: int = 1500, verbose: bool = True,
                 provider=None) -> None:
        self.strategy = strategy
        self.budget = budget
        self.verbose = verbose
        self.provider = provider or GenericProvider()  # generic | product
        self.conv = Conversation()
        self.messages: list[dict] = [{"role": "system", "content": self.provider.SYSTEM}]
        self.by_id: dict[str, ToolResult] = {}   # tool_call_id → ToolResult
        self.last_preamble = ""

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy

    # --- köprü: modelin GÖRDÜĞÜ (sıkışık) messages[] ---------------------
    def _render_messages(self) -> list[dict]:
        """Ham messages[]'i sıkışık görünüme çevir: tool gövdeleri kaderine göre,
        preamble system olarak başa. Yapı (tool_call_id) korunur."""
        wire: list[dict] = []
        for m in self.messages:
            if m.get("role") == "tool":
                tr = self.by_id.get(m.get("tool_call_id"))
                shown = tr.shown() if tr else m.get("content", "")
                if shown == "":  # toplu özete taşındı — boş içerik göndermeyelim
                    shown = "[içerik compaction ile özete taşındı — üstteki compaction özetine bakın]"
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                             "name": m.get("name", ""), "content": shown})
            else:
                wire.append(m)
        if self.last_preamble:
            wire.insert(1, {"role": "system",
                            "content": "[compaction özeti]\n" + self.last_preamble})
        return wire

    # --- bir tool çağrısını çalıştır + trace'e işle ----------------------
    def _exec_tool_call(self, tc: dict, turn) -> None:
        fn = tc["function"]
        name = fn["name"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        out = self.provider.run(name, args)
        tr = ToolResult(call_id=tc["id"], name=name, tool_type=self.provider.tool_type(name),
                        resource=self.provider.resource_of(name, args), content=out,
                        turn=len(self.conv.turns), args=args)
        turn.results.append(tr)
        self.by_id[tc["id"]] = tr
        # ham tool mesajını ekle (yapı kaynağı); görünüm _render_messages'te sıkışır
        self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                              "name": name, "content": out})
        if self.verbose:
            print(f"    ⚙ {name}({self.provider.resource_of(name, args)}) → {est(out)} tok ham")

    # --- ana döngü -------------------------------------------------------
    def send(self, user: str) -> dict:
        self.messages.append({"role": "user", "content": user})
        turn = self.conv.new_turn(user)
        answer = ""
        for step in range(MAX_STEPS):
            wire = self._render_messages()
            force_answer = step == MAX_STEPS - 1
            resp = llm.chat(wire, tools=self.provider.SCHEMAS,
                            tool_choice="none" if force_answer else "auto")
            tool_calls = resp.get("tool_calls")
            if tool_calls and not force_answer:
                # asistanın tool_calls mesajını ham olarak sakla (yapı için)
                self.messages.append({"role": "assistant",
                                      "content": resp.get("content") or "",
                                      "tool_calls": tool_calls})
                for tc in tool_calls:
                    self._exec_tool_call(tc, turn)
                # compaction: tüm trace, seçili strateji (köprü sonraki turda etkir)
                self.conv.reset_fates()
                self.last_preamble = self.strategy.compact(
                    self.conv.all_results(), self.conv, self.budget)
                continue
            # tool yok → nihai yanıt
            answer = resp.get("content") or ""
            self.messages.append({"role": "assistant", "content": answer})
            break
        turn.answer = answer
        return self._summary(answer)

    def _summary(self, answer: str) -> dict:
        results = self.conv.all_results()
        fates: dict[str, int] = {}
        for r in results:
            fates[r.fate] = fates.get(r.fate, 0) + 1
        raw = self.conv.raw_tokens()
        shown = self.conv.shown_tokens(self.last_preamble)
        return {
            "answer": answer or "(model yanıt üretmedi)",
            "raw_tokens": raw, "shown_tokens": shown,
            "saved_pct": round(100 * (raw - shown) / raw) if raw else 0,
            "units": len(results), "fates": fates, "preamble": self.last_preamble,
        }
