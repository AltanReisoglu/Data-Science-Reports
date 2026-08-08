"""
harness.py — POC ortak iskeleti.

Her rakip sistemin (Hermes, Codex, Cline, ...) tool-trace compaction mantığını
AYNI trace üzerinde çalıştırabilmek için paylaşılan altyapı:

  * ToolResult / Turn / Conversation — OpenAI-tarzı messages[] ile eşleşen model.
  * est()                          — deterministik token tahmini (~4 karakter = 1 token).
  * MockTools                      — tip-çeşitli, GERÇEKÇE-BÜYÜK çıktı üreten sahte tool'lar
                                     (her sistemin tip-özel yolu ancak farklı tiplerle tetiklenir).
  * ScriptedBrain                  — kullanıcı mesajını tool çağrı planına çeviren deterministik ajan.
  * ChatSession                    — çok-turlu sohbet; her turda seçili strateji trace'i sıkıştırır.

Hiçbir dış bağımlılık yok — sadece stdlib. Böylece her ortamda çalışır.
Not: LLM tool-seçimi/yanıtı burada scripted; amaç compaction'ı karşılaştırmak,
LLM'in tool seçimini değil. (İsteğe bağlı canlı LLM için strategies/base.py'ye bak.)
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Token tahmini — bütün sistemler byte/token bütçesiyle çalışır; tek ölçü kullanalım.
# --------------------------------------------------------------------------
def est(text: str) -> int:
    """Kaba token tahmini: ~4 karakter = 1 token. Deterministik, bağımlılıksız."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Veri modeli — messages[] köprüsü ile birebir.
# --------------------------------------------------------------------------
# Tool tipleri — her sistem farklı tiplere özel davranır:
#   terminal      → Hermes tek-satır, Codex orta-kes, SWE "omitted"
#   read_file     → Cline dedup, Roo fold, Claude Code clear
#   web_extract   → Hermes web şablonu, Headroom metin compressor
#   take_snapshot → gemini-cli supersede (bayat snapshot)
#   grep          → Headroom SearchCompressor
#   write_file    → mutasyon (bizim staleness'i tetikler)
TOOL_TYPES = ("terminal", "read_file", "web_extract", "take_snapshot", "grep", "write_file")

_MUTATING = {"write_file"}


@dataclass
class ToolResult:
    """Tek bir tool çağrısının sonucu — bir 'tool' rollü mesaja karşılık gelir."""
    call_id: str
    name: str            # tool adı (ör. read_file)
    tool_type: str       # TOOL_TYPES'tan biri
    resource: str        # dosya yolu / url / komut / sorgu — dedup/staleness anahtarı
    content: str         # HAM çıktı (sıkıştırmadan önce)
    turn: int            # kaçıncı kullanıcı turunda üretildi
    args: dict = field(default_factory=dict)

    # --- strateji tarafından doldurulan görüntüleme alanları (compaction sonucu) ---
    fate: str = "TAM"           # TAM|KES|ÖZET|MASKE|GİZLE|SİL|KATLA|SUPERSEDE|DEDUP|CRUSH
    view: Optional[str] = None  # modelin GERÇEKTE gördüğü gövde (None → ham content)
    note: str = ""              # kader hakkında tek satırlık açıklama (debug/gösterim)

    @property
    def is_mutation(self) -> bool:
        return self.tool_type in _MUTATING

    def raw_tokens(self) -> int:
        return est(self.content)

    def shown(self) -> str:
        """Modele giden gövde: strateji bir görüntü koyduysa o, yoksa ham."""
        return self.content if self.view is None else self.view

    def shown_tokens(self) -> int:
        return est(self.shown())

    def reset_fate(self) -> None:
        self.fate, self.view, self.note = "TAM", None, ""


@dataclass
class Turn:
    user: str
    results: list[ToolResult] = field(default_factory=list)
    answer: str = ""


class Conversation:
    """Turların dizisi. Tool sonuçları turlar boyunca BİRİKİR."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self._ids = itertools.count(1)

    def new_turn(self, user: str) -> Turn:
        t = Turn(user=user)
        self.turns.append(t)
        return t

    def next_call_id(self) -> str:
        return f"call_{next(self._ids):03d}"

    def all_results(self) -> list[ToolResult]:
        return [r for t in self.turns for r in t.results]

    def reset_fates(self) -> None:
        for r in self.all_results():
            r.reset_fate()

    # --- token muhasebesi ---
    def raw_tokens(self) -> int:
        return sum(r.raw_tokens() for r in self.all_results())

    def shown_tokens(self, preamble: str = "") -> int:
        base = est(preamble) if preamble else 0
        return base + sum(r.shown_tokens() for r in self.all_results())

    def user_turn_count(self) -> int:
        return len(self.turns)


# --------------------------------------------------------------------------
# Mock tool'lar — gerçekçe, tip-farklı, BÜYÜK çıktılar (sıkıştırılacak hammadde).
# --------------------------------------------------------------------------
class MockTools:
    """Deterministik sahte tool'lar. Her çağrı aynı girdi için aynı çıktıyı verir,
    ama write_file bir 'sürüm' artırır → sonraki read_file 'bayat'ı tetikler."""

    def __init__(self) -> None:
        # dosya sürüm sayacı: write_file çağrılınca artar (bizim staleness için)
        self._version: dict[str, int] = {}

    def bump(self, path: str) -> int:
        self._version[path] = self._version.get(path, 0) + 1
        return self._version[path]

    def version(self, path: str) -> int:
        return self._version.get(path, 0)

    # --- tip başına üreticiler ---
    def terminal(self, cmd: str) -> str:
        lines = [f"$ {cmd}"]
        if "test" in cmd:
            for i in range(1, 44):
                lines.append(f"  PASS  tests/unit/case_{i:02d}.spec.ts  ({i*3} ms)")
            lines += ["", "Test Suites: 43 passed, 43 total",
                      "Tests:       128 passed, 128 total", "Time:        6.42 s",
                      "Ran all test suites.", "exit 0"]
        elif "build" in cmd:
            for i in range(1, 60):
                lines.append(f"  [webpack] compiled module chunk_{i:03d}.js  ({120+i} KiB)")
            lines += ["", "webpack 5.91 compiled successfully in 8123 ms", "exit 0"]
        else:
            for i in range(1, 30):
                lines.append(f"  out[{i:02d}]: processing record {1000+i} ... ok")
            lines.append("exit 0")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        v = self.version(path)
        head = f"# {path}   (sürüm v{v})\n"
        body = []
        body.append("import os, sys, json\nfrom dataclasses import dataclass\n")
        for i in range(1, 46):
            if i % 12 == 0:
                body.append(f"\ndef handler_{i}(req, ctx):")
                body.append(f"    \"\"\"v{v} — {path} satır {i}.\"\"\"")
                body.append(f"    return process(req, limit={i*7})")
            elif i % 12 == 6:
                body.append(f"\nclass Service_{i}:")
                body.append(f"    PORT = {8000+i}")
                body.append("    def start(self): ...")
            else:
                body.append(f"value_{i} = compute({i}, factor={i%5})   # satır {i}")
        return head + "\n".join(body)

    def web_extract(self, url: str) -> str:
        lines = [f"URL: {url}", "=" * 40]
        for i in range(1, 38):
            lines.append(f"[p{i:02d}] Lorem ipsum dolor sit amet, consectetur "
                         f"adipiscing elit, section {i} of the page body text.")
        lines += ["", "Related links:"] + [f"  - {url}/sub/{j}" for j in range(1, 9)]
        return "\n".join(lines)

    def take_snapshot(self, page: str) -> str:
        lines = [f"# accessibility snapshot: {page}", "role=document"]
        for i in range(1, 52):
            lines.append(f"  [{i:03d}] role=button name=\"action-{i}\" state="
                         f"{'focused' if i == 3 else 'enabled'} ref=node_{i}")
        return "\n".join(lines)

    def grep(self, query: str) -> str:
        lines = [f"rg '{query}'  (matches)"]
        for i in range(1, 34):
            lines.append(f"src/module_{i%9}/file_{i}.py:{i*4}:    "
                         f"result matching '{query}'  → context tail {i}")
        lines.append("33 matches across 9 files")
        return "\n".join(lines)

    def write_file(self, path: str) -> str:
        v = self.bump(path)
        return (f"WROTE {path}  (yeni sürüm v{v})\n"
                f"  +{12} satır, -{4} satır  ·  128 bytes yazıldı  ·  ok")


# --------------------------------------------------------------------------
# ScriptedBrain — kullanıcı mesajından deterministik tool planı.
# --------------------------------------------------------------------------
@dataclass
class PlanStep:
    tool_type: str
    resource: str
    name: str


_FILE_RE = re.compile(r"[\w./-]+\.(?:py|ts|js|tsx|md|json|yaml|go|rs)")
_URL_RE = re.compile(r"https?://[\w./-]+")


class ScriptedBrain:
    """Kullanıcı cümlesini tool çağrılarına çevirir (LLM yerine deterministik planlayıcı).

    Amaç: her sistemin tip-özel mantığını GERÇEK bir sohbette tetiklemek.
    Örn: aynı dosyayı iki kez 'oku' → Cline dedup; 'snapshot' iki kez → gemini supersede;
    'düzenle' sonra 'oku' → bizim staleness.
    """

    def plan(self, text: str) -> list[PlanStep]:
        low = text.lower()
        steps: list[PlanStep] = []
        files = _FILE_RE.findall(text)
        urls = _URL_RE.findall(text)

        if any(k in low for k in ("test", "çalıştır", "run", "build")):
            cmd = "npm run build" if "build" in low else "npm test"
            steps.append(PlanStep("terminal", cmd, "run_terminal"))
        if any(k in low for k in ("düzenle", "edit", "yaz", "write", "fix", "düzelt")):
            tgt = files[0] if files else "src/app.py"
            steps.append(PlanStep("write_file", tgt, "write_file"))
        for f in files:
            steps.append(PlanStep("read_file", f, "read_file"))
        for u in urls:
            steps.append(PlanStep("web_extract", u, "web_extract"))
        if any(k in low for k in ("snapshot", "sayfa", "page", "ekran", "browser", "tarayıcı")):
            page = urls[0] if urls else "https://app.local/dashboard"
            steps.append(PlanStep("take_snapshot", page, "take_snapshot"))
        if any(k in low for k in ("ara", "search", "grep", "bul", "find", "nerede")):
            m = re.search(r"['\"]([^'\"]+)['\"]", text)
            q = m.group(1) if m else (low.split()[-1] if low.split() else "TODO")
            steps.append(PlanStep("grep", q, "grep"))

        if not steps:
            # varsayılan: iki dosya oku + bir arama (birikim olsun)
            steps = [PlanStep("read_file", "src/server.py", "read_file"),
                     PlanStep("read_file", "src/config.py", "read_file"),
                     PlanStep("grep", "PORT", "grep")]
        return steps


# --------------------------------------------------------------------------
# ChatSession — çok-turlu; her turda seçili strateji trace'i sıkıştırır.
# --------------------------------------------------------------------------
class ChatSession:
    def __init__(self, strategy, budget: int = 1500) -> None:
        self.strategy = strategy
        self.budget = budget
        self.conv = Conversation()
        self.tools = MockTools()
        self.brain = ScriptedBrain()
        self.last_preamble = ""

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy

    def send(self, user: str) -> dict:
        """Bir kullanıcı mesajını işle: planla → tool'ları çalıştır → compaction → yanıt.

        Döndürür: bu turun özeti (yanıt + ham/sıkışık token + kader dağılımı)."""
        turn = self.conv.new_turn(user)
        for step in self.brain.plan(user):
            out = self._run_tool(step)
            turn.results.append(
                ToolResult(call_id=self.conv.next_call_id(), name=step.name,
                           tool_type=step.tool_type, resource=step.resource,
                           content=out, turn=len(self.conv.turns),
                           args={"resource": step.resource}))

        # --- compaction: her turda seçili strateji AYNI trace üzerinde çalışır ---
        self.conv.reset_fates()
        self.last_preamble = self.strategy.compact(
            self.conv.all_results(), self.conv, self.budget)

        answer = self._answer(turn)
        turn.answer = answer
        return self._summary(turn, answer)

    def _run_tool(self, step: PlanStep) -> str:
        t = self.tools
        return {
            "terminal": lambda: t.terminal(step.resource),
            "read_file": lambda: t.read_file(step.resource),
            "web_extract": lambda: t.web_extract(step.resource),
            "take_snapshot": lambda: t.take_snapshot(step.resource),
            "grep": lambda: t.grep(step.resource),
            "write_file": lambda: t.write_file(step.resource),
        }[step.tool_type]()

    def _answer(self, turn: Turn) -> str:
        kinds = ", ".join(sorted({r.tool_type for r in turn.results}))
        return (f"{len(turn.results)} tool çağırdım ({kinds}); "
                f"sonuçları '{self.strategy.name}' mantığıyla sıkıştırıp yanıtladım.")

    def _summary(self, turn: Turn, answer: str) -> dict:
        results = self.conv.all_results()
        fates: dict[str, int] = {}
        for r in results:
            fates[r.fate] = fates.get(r.fate, 0) + 1
        raw = self.conv.raw_tokens()
        shown = self.conv.shown_tokens(self.last_preamble)
        return {
            "answer": answer,
            "raw_tokens": raw,
            "shown_tokens": shown,
            "saved_pct": round(100 * (raw - shown) / raw) if raw else 0,
            "units": len(results),
            "fates": fates,
            "preamble": self.last_preamble,
        }
