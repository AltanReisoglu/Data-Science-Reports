"""Her koşu tipinin grafı ve sıra diyagramı sağlam mı.

Bu dosya tek bir hata sınıfını kovalıyor: **grafın kendi kendine tutarsız
olması**. Bir kenarın olmayan bir düğüme bakması, bir sıra adımının olmayan bir
şeride ok çizmesi, ya da ileri kenarlarda kalan bir döngünün yerleşimi sonsuza
kadar sağa uzatması. Üçü de ekranda sessizce bozuk bir resim üretiyor —
hiçbiri istisna fırlatmıyor.

Ölçülmüş kaynak: RoundRobin üçüncü ajandan sonra birinciye dönüyor. Dönüş
kenarı ileri sayıldığında altı konuşmalık bir koşu yirmi sütuna çıktı ve graf
karta sığdırılınca kılcal çizgilere döndü. Model çağrısı yapılmadan
yakalanabilecek bir hataydı; bu dosya artık onu yakalıyor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runlog  # noqa: E402
import stages  # noqa: E402
import teams  # noqa: E402


def _stage(sid: str, **meta):
    entry = stages.CATALOGUE.get(sid)
    base = entry.as_dict() if entry else {"id": sid, "title": sid}
    return {"type": "stage", **base, "meta": meta}


def _chat_run() -> runlog.Run:
    run = runlog.Run(id="c", kind="chat", question="soru")
    run.event(_stage("context", tokens=900, budget=12000))
    run.event(_stage("model", streaming=True))
    run.event({"type": "tool", "name": "scan_facts"})
    run.event(_stage("gate", tool="scan_facts", blocked=False))
    run.event(_stage("tool_exec", tool="scan_facts"))
    run.event({"type": "tool_result", "name": "scan_facts"})
    run.event(_stage("done", llm_calls=2, tokens=900, tool_calls=1))
    run.event({"type": "done", "text": "cevap", "stop_reason": ""})
    return run


def _scan_run() -> runlog.Run:
    run = runlog.Run(id="s", kind="scan", question="tarama")
    for sid in stages.SCAN_FLOW:
        run.event(_stage(sid))
    run.event({"type": "done", "text": "", "stop_reason": ""})
    return run


def _team_run(kind: str) -> runlog.Run:
    """Döngüsel bir konuşma sırası — RoundRobin'in gerçekte yaptığı."""
    run = runlog.Run(id="t-" + kind, kind="team", question="soru")
    run.variant = kind
    run.event(_stage("team_build", kind=kind))
    cast = ["Planner", "Researcher", "Critic"]
    if kind == "magenticone":
        # Takım kendi yöneticisini yaratıyor; kadroda yok.
        cast = ["MagenticOneOrchestrator", "Planner", "MagenticOneOrchestrator",
                "Researcher", "MagenticOneOrchestrator", "Critic"]
    for i, who in enumerate(cast * 3, start=1):
        run.event(_stage("speaker", who=who, turn=i))
    if kind == "swarm":
        run.event(_stage("handoff", who="Planner", to="Researcher"))
    run.event(_stage("team_done", stop_reason="max message"))
    return run


def _cron_run() -> runlog.Run:
    """Zamanlama turu. Üst bandı BOŞ, ve bu bilinçli: bu yolda AutoGen'in
    hiçbir parçası koşmuyor. Testin buradaki işi, boş bandın yerleşimi
    bozmadığını doğrulamak — sıfır düğümlü bir bant, sıfıra bölme demek."""
    run = runlog.Run(id="k", kind="cron", question="her sabah 9da tarama yap")
    run.event(_stage("cron_parse", text="her sabah 9da tarama yap", action="add"))
    run.event(_stage("cron_gate", method="cron.add", held=False, when="her gün 09:00"))
    run.event(_stage("cron_done", action="add", ok=True, when="her gün 09:00"))
    return run


def _team_tool_run() -> runlog.Run:
    """Tool çağıran takım: ajan başına tool düğümleri ve dönüş okları."""
    run = _team_run("selector")
    run.event(_stage("team_tool", who="Researcher", tool="search_docs"))
    run.event(_stage("team_tool", who="Researcher", tool="search_docs"))
    run.event(_stage("team_tool", who="Critic", tool="scan_facts"))
    return run


def _cases():
    yield "chat", _chat_run()
    yield "scan", _scan_run()
    yield "cron", _cron_run()
    yield "team:selector tool'lu", _team_tool_run()
    for kind in teams.KINDS:
        yield "team:" + kind, _team_run(kind)


class GraphShapeTests(unittest.TestCase):
    def test_every_edge_points_at_a_real_node(self):
        for label, run in _cases():
            with self.subTest(label):
                graph = run.graph()
                ids = {n["id"] for n in graph["nodes"]}
                for e in graph["edges"]:
                    self.assertIn(e["src"], ids, f"{label}: {e}")
                    self.assertIn(e["dst"], ids, f"{label}: {e}")

    def test_forward_edges_never_form_a_cycle(self):
        """Yerleşim döngüsüz varsayıyor; döngü sütun sayısını patlatıyor."""
        for label, run in _cases():
            with self.subTest(label):
                graph = run.graph()
                nodes = {n["id"] for n in graph["nodes"]}
                forward = [(e["src"], e["dst"]) for e in graph["edges"]
                           if not e.get("back") and not e.get("cross")]
                depth = {n: 0 for n in nodes}
                for _ in range(len(forward) + 1):
                    for src, dst in forward:
                        depth[dst] = max(depth[dst], depth[src] + 1)
                self.assertLessEqual(max(depth.values()), len(nodes),
                                     f"{label}: ileri kenarlarda döngü var")

    def test_the_drawing_stays_within_a_readable_width(self):
        """Sütun sayısı düğüm sayısını aşarsa graf karta sığmıyor.

        `akis.js` sütunu 320 px sayıyor; on sütunu geçen bir graf zaten
        okunmuyor, ve o noktada bir yerleşim hatası var demektir.
        """
        for label, run in _cases():
            with self.subTest(label):
                graph = run.graph()
                for band in (0, 1):
                    ids = {n["id"] for n in graph["nodes"] if n.get("band", 0) == band}
                    forward = [(e["src"], e["dst"]) for e in graph["edges"]
                               if not e.get("back") and not e.get("cross")
                               and e["src"] in ids and e["dst"] in ids]
                    depth = {n: 0 for n in ids}
                    for _ in range(len(forward) + 1):
                        for src, dst in forward:
                            depth[dst] = max(depth[dst], depth[src] + 1)
                    columns = (max(depth.values()) + 1) if depth else 0
                    self.assertLessEqual(columns, 10,
                                         f"{label} bant {band}: {columns} sütun")

    def test_every_sequence_step_points_at_a_real_lifeline(self):
        for label, run in _cases():
            with self.subTest(label):
                seq = run.sequence()
                lanes = {l["id"] for l in seq["lanes"]}
                for step in seq["steps"]:
                    self.assertIn(step["src"], lanes, f"{label}: {step}")
                    self.assertIn(step["dst"], lanes, f"{label}: {step}")

    def test_group_boxes_reference_steps_that_exist(self):
        for label, run in _cases():
            with self.subTest(label):
                seq = run.sequence()
                for g in seq["groups"]:
                    self.assertGreaterEqual(g["from"], 0, label)
                    self.assertLess(g["to"], len(seq["steps"]), label)
                    self.assertLessEqual(g["from"], g["to"], label)

    def test_every_report_is_serialisable_and_complete(self):
        """Ekranın çizdiği her alan gerçekten geliyor mu."""
        import json

        for label, run in _cases():
            with self.subTest(label):
                report = run.report()
                json.dumps(report, ensure_ascii=False)   # fırlatmamalı
                for key in ("design", "graph", "sequence", "teams", "patterns",
                            "messages", "components", "topics", "timeline",
                            "details", "totals"):
                    self.assertIn(key, report, f"{label}: {key} yok")

    def test_the_counter_never_contradicts_the_drawing(self):
        """Grafta tool kutusu varsa sayaç sıfır diyemez — ve tersi.

        Bu hata **üç kez** çıktı, her seferinde aynı şekilde: sayaç bir olay
        türünü sayıyor, çağrı başka bir yoldan geliyor, ve ekran kendi
        diyagramıyla çelişiyor.

        1. `done.tool_calls` yalnız koşanı sayıyordu; kapı tuttuğunda sıfır.
        2. O sayaç `BaseTool` olaylarını sayıyordu; **MCP** onu yaymıyor.
        3. Takım koşusu `tool_exec` değil `team_tool` yayıyor — graf
           "Critic çağırdı, 4 kez" derken sayaç `TOOL · KOŞTU 0` diyordu.

        Bir sayacı düzeltmek bir kez; bu bekçi, dördüncüsünü yazılırken
        yakalasın diye duruyor. Model çağırmadan koşuyor.
        """
        for label, run in _cases():
            with self.subTest(label):
                tools = [n for n in run.graph()["nodes"] if n["kind"] == "tool"]
                ran = run.totals()["tools_ran"]
                if tools:
                    self.assertGreater(
                        ran, 0,
                        f"{label}: grafta {len(tools)} tool kutusu var ama "
                        f"sayaç {ran} diyor",
                    )
                else:
                    self.assertEqual(
                        ran, 0,
                        f"{label}: sayaç {ran} tool koştu diyor ama grafta "
                        f"tool kutusu yok",
                    )

    def test_exactly_one_team_is_claimed_per_run(self):
        for label, run in _cases():
            with self.subTest(label):
                used = [t["id"] for t in run.teams() if t["used"]]
                self.assertLessEqual(len(used), 1, f"{label}: {used}")
                if label.startswith("team:"):
                    # Etiket "team:<tip>" ya da "team:<tip> <not>" olabiliyor;
                    # nottan önceki kelime takım tipi.
                    self.assertEqual(used, [label.split(":", 1)[1].split(" ")[0]])
                elif label == "chat":
                    self.assertEqual(used, [])


if __name__ == "__main__":
    unittest.main()
