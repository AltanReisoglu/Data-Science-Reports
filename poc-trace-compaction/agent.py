"""
agent.py — OpenAI-uyumlu ajan döngüsü + trace compaction katmanı.

ek-a §7 manuel döngüsü. Trace katmanı, model çağrısı ile tool yürütmesi
ARASINDAKİ boşluğa girer (§6.7 diyagramı):
  model → [OLAY KAYDET → YÜRÜT → LEDGER → TETİKLE/EVICT] → messages'e ekle

Endpoint OpenAI-uyumlu olduğu için openai SDK base_url ile kullanılır
(model Gemma; Anthropic SDK değil).
"""
from __future__ import annotations
import json

from config import (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME,
                    TRACE_TOKEN_BUDGET, TRACE_PROTECT_WINDOW, estimate_tokens)
from trace import Trace
from ledger import ExecutionLedger
from compactor import TraceCompactor, _task_keywords
from episode_graph import EpisodeGraph
from playbook import Playbook
from ptc import PTCSandbox
from tools import SCHEMAS, DISPATCH, read_file, list_dir, grep

SYSTEM = (
    "Sen bir kod ajanısın. sample_repo üzerinde çalışıyorsun. "
    "Verilen görevi tool'ları kullanarak tamamla. Aynı dosyayı gereksiz yere "
    "tekrar okuma; grep ile hedefli ara. Bitince kısa bir özet yaz."
)


class TracingAgent:
    """Trace compaction'lı ajan. compaction=False ile taban çizgisi ölçülür."""

    def __init__(self, compaction: bool = True, max_turns: int = 12,
                 use_llm_summary: bool = False, schemas=None, dispatch=None,
                 tool_meta=None, system: str = None):
        self.compaction = compaction
        self.max_turns = max_turns
        self.schemas = schemas if schemas is not None else SCHEMAS
        self.dispatch = dispatch if dispatch is not None else DISPATCH
        self.system = system or SYSTEM
        self.trace = Trace()
        self.ledger = ExecutionLedger(tool_meta=tool_meta)   # genel sözleşme (domain-bağımsız)
        self.episodes = EpisodeGraph()   # CWL: ajanın delimiter ile tiplediği yapı
        self.playbook = Playbook()       # ACE (K4): evict'ten korunan öğrenilmiş ders
        # PTC (K3): sandbox'a yalnızca salt-okuma tool'ları enjekte edilir
        self.ptc = PTCSandbox({"read_file": read_file, "list_dir": list_dir,
                               "grep": grep})
        summarize_fn = self._llm_summarize if use_llm_summary else None
        self.compactor = TraceCompactor(TRACE_TOKEN_BUDGET, TRACE_PROTECT_WINDOW,
                                        summarize_fn=summarize_fn,
                                        playbook=self.playbook)
        self.metrics = {"turns": 0, "tool_calls": 0, "evicted": 0,
                        "compaction_passes": 0, "peak_trace_tokens": 0,
                        "playbook_bullets": 0, "ptc_inner_saved": 0}
        self._client = None
        self.messages = None   # çok-turlu chat için kalıcı mesaj dizisi

    # --- LLM istemcisi (OpenAI-uyumlu) -----------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai kurulu değil. Kurulum: pip install -r requirements.txt")
            if not LLM_API_KEY:
                raise RuntimeError(
                    "LLM_API_KEY boş. .env dosyasını doldurun (bkz. .env.example).")
            self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        return self._client

    def _llm_summarize(self, ev) -> str:
        """Opsiyonel: bir tool çıktısını LLM ile özetle (görev-koşullu)."""
        client = self._get_client()
        out = ev.payload.get("output", "")
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME, temperature=0, max_tokens=80,
            messages=[{"role": "system",
                       "content": "Tool çıktısını tek satırda özetle. Dosya yolu, "
                                  "satır no, sayı gibi kritik değerleri BİREBİR koru."},
                      {"role": "user", "content": out[:2000]}])
        return r.choices[0].message.content.strip()

    # --- trace katmanı: boşluğa giren iş (§6.7) --------------------------

    def _handle_delimiter(self, args: dict) -> str:
        """CWL delimiter çağrısını episode grafiğine uygula."""
        action = args.get("action")
        # seq olarak trace'in şu anki uzunluğunu kullan (sınır işareti)
        seq = self.trace._seq
        if action == "start":
            self.episodes.start(
                name=args.get("name", f"ep{seq}"),
                type=args.get("type", "expl"), seq=seq,
                dependencies=args.get("dependencies"))
            return f"OK: '{args.get('name')}' ({args.get('type')}) episode açıldı"
        elif action == "end":
            self.episodes.end(seq, description=args.get("description", ""))
            return "OK: episode kapandı"
        return "delimiter: action start|end olmalı"

    def _record_and_maybe_compact(self, name, args, output, status, intent_ref):
        """Tool sonucunu trace+ledger'a işle, gerekiyorsa sıkıştır."""
        verbatim = name in ("grep", "run_tests", "run_code")  # sonuç kritik (yol/port/bulgu)
        meta = self.ledger.tool_meta                          # domain-farkında verbatim
        if meta and name in meta and meta[name].get("verbatim"):
            verbatim = True
        ev = self.trace.add_tool(name, args, output, status=status,
                                 intent_ref=intent_ref, verbatim=verbatim)
        self.ledger.record(name, args, output, ev.seq)
        self.episodes.attach(ev.seq)               # CWL: aktif episode'a bağla
        self.metrics["tool_calls"] += 1

        if self.compaction:
            res = self.compactor.compact(self.trace, self.ledger,
                                         episode_graph=self.episodes)
            if res["triggered"]:
                self.metrics["compaction_passes"] += 1
                self.metrics["evicted"] += res["evicted"]
            self.metrics["playbook_bullets"] = self.playbook.stats()["active"]
        self.metrics["peak_trace_tokens"] = max(
            self.metrics["peak_trace_tokens"], self.trace.total_tokens())
        return None

    # --- ana döngü (ek-a §7) ---------------------------------------------

    def run(self, task: str) -> str:
        """Tek-atışlık: yeni bir sohbet başlatıp tek soruyu yanıtlar."""
        self.messages = [{"role": "system", "content": self.system}]
        return self.send(task)

    def send(self, user_msg: str) -> str:
        """ÇOK-TURLU chat: bir kullanıcı mesajını yanıtla. Trace/ledger/playbook/
        mesajlar turlar boyunca KORUNUR — trace birikir, compaction arkada işler.
        """
        client = self._get_client()
        if getattr(self, "messages", None) is None:
            self.messages = [{"role": "system", "content": self.system}]
        # göreve-koşullu sıkıştırma (K5): compactor en son soruyu bilir
        self.compactor.task = user_msg
        self.compactor.keywords = _task_keywords(user_msg)
        self.messages.append({"role": "user", "content": user_msg})

        for _ in range(self.max_turns):
            self.metrics["turns"] += 1
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME, temperature=0,
                tools=self.schemas, messages=self._render_messages(self.messages))
            msg = resp.choices[0].message

            if msg.content:
                self.trace.add_reasoning(msg.content)
            intent_ref = self.trace.events[-1].seq if (
                self.trace.events and self.trace.events[-1].type == "reasoning") else None

            if not msg.tool_calls:
                self.trace.add_answer(msg.content or "")
                self.messages.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or ""

            # asistan turunu OLDUĞU GİBİ ekle (ek-a: tool_calls kaybolmasın)
            self.messages.append({"role": "assistant", "content": msg.content or "",
                                  "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                # CWL delimiter: dosya işi yapmaz, episode grafiğini günceller
                if name == "delimiter":
                    output = self._handle_delimiter(args)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": output})
                    continue

                # PTC (§12.10): kodu sandbox'ta çalıştır. İçerideki N tool çağrısı
                # trace'e GİRMEZ — yalnızca print sonucu TEK olay olarak girer (uzaysal).
                if name == "run_code":
                    res = self.ptc.run(args.get("code", ""))
                    output = res["output"]
                    self.metrics["ptc_inner_saved"] += res["inner_calls"]
                    self._record_and_maybe_compact(
                        "run_code", {"inner_calls": res["inner_calls"]},
                        output, res["status"], intent_ref)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": output})
                    continue

                try:
                    output = str(self.dispatch[name](**args))
                    status = "ok"
                except Exception as e:                       # hata → is_error kanalı
                    output = f"Hata: {e}"
                    status = "error"

                self._record_and_maybe_compact(name, args, output, status, intent_ref)
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": output})

        return "(max_turns aşıldı)"

    def _render_messages(self, messages: list[dict]) -> list[dict]:
        """Trace katmanı sıkıştırdıysa, tool sonuçlarını sıkışık hâlleriyle gönder.

        evict edilmiş tool olaylarının içeriği, messages[]'teki ilgili tool
        mesajında özetle değiştirilir — böylece modelin gördüğü bağlam küçülür.
        """
        if not self.compaction:
            return messages
        # ACE (K4): playbook'u bağlamın ÜSTÜNE enjekte et. Trace evict edilse de
        # buradaki ders durur — öğrenilmiş bilgi sıkıştırmadan bağımsız yaşar.
        pb = self.playbook.render()
        if pb:
            head = messages[0]
            merged = dict(head)
            merged["content"] = head["content"] + "\n\n" + pb
            messages = [merged] + messages[1:]
        # Not: evict edilmiş tool sonuçlarının messages[]'e geri yazılması bu
        # POC'de basitleştirilmiştir; sıkışma trace.total_tokens ile ölçülür.
        return messages
