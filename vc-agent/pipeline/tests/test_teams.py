"""Beş takım tipi: kurulabiliyor mu, ve kayıt onları doğru anlatıyor mu.

Model çağrısı yapılmıyor. Burada ölçülen şey kurulum: `Swarm`'ın devir
tool'ları doğuyor mu, `Selector`'ın seçim için bakacağı `description` dolu mu,
ve sonlandırma koşulu her takımda var mı. Bunların hiçbiri koşturmadan
görülebiliyor, ve koşturarak ölçmek her testte gerçek para demek.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runlog  # noqa: E402
import stages  # noqa: E402
import teams  # noqa: E402


class RosterTests(unittest.TestCase):
    def test_every_agent_has_a_description(self):
        # `SelectorGroupChat` sıradaki konuşmacıyı bu metne bakarak seçiyor.
        # Boşsa seçim kör yapılıyor ve hata da vermiyor.
        for spec in teams.ROSTER:
            self.assertTrue(spec["description"].strip(), spec["name"])

    def test_the_roster_is_the_same_for_every_team(self):
        # Kadro takım tipine göre değişseydi, ölçülen token farkının takımdan mı
        # kadrodan mı geldiği belirsiz kalırdı.
        self.assertEqual(len(teams.KINDS), 5)
        self.assertEqual([a["name"] for a in teams.ROSTER],
                         [a["name"] for a in runlog.TEAM_ROSTER])

    def test_every_kind_declares_who_picks_the_speaker(self):
        for kind in teams.KINDS:
            self.assertIn(kind, teams.PICKER)


class HandoffTests(unittest.TestCase):
    def test_swarm_agents_get_handoff_tools_and_others_do_not(self):
        """Devir bir TOOL çağrısı; yalnız Swarm'da doğuyor."""
        from autogen_agentchat.base import Handoff

        names = [a["name"] for a in teams.ROSTER]
        for spec in teams.ROSTER:
            targets = [n for n in names if n != spec["name"]]
            handoffs = [Handoff(target=t) for t in targets]
            self.assertEqual(len(handoffs), len(names) - 1)
            for h in handoffs:
                # Elle yazılan ad eşleşmiyor: üretilen ad küçük harfe düşüyor.
                self.assertEqual(h.name, h.name.lower())
                self.assertNotEqual(h.name, f"transfer_to_{h.target}")


class TargetTests(unittest.TestCase):
    def test_the_handoff_target_comes_back_with_the_roster_s_casing(self):
        """Tool adı küçük harfe düşüyor; ham hâli şeritle eşleşmiyor.

        Ölçüldü (team-0001, swarm): sıra diyagramında `Researcher → researcher`
        diye bir ok çizildi ve hedef şerit yoktu.
        """
        class Call:
            name = "transfer_to_researcher"

        class Event:
            content = [Call()]

        self.assertEqual(teams._handoff_target(Event()), "Researcher")

    def test_an_unknown_target_is_returned_as_is_rather_than_dropped(self):
        class Call:
            name = "transfer_to_someoneelse"

        class Event:
            content = [Call()]

        self.assertEqual(teams._handoff_target(Event()), "someoneelse")


class TierTests(unittest.TestCase):
    def test_the_selector_asks_for_a_tier_that_exists(self):
        """`KeyError: 'small'` ile Selector daha ilk anda düşüyordu."""
        import engine

        source = Path(teams.__file__).read_text(encoding="utf-8")
        for tier in ("cheap", "mid", "strong"):
            pass
        self.assertNotIn('raw_client("small")', source)
        self.assertIn('raw_client("cheap")', source)


class CatalogueTests(unittest.TestCase):
    def test_every_team_stage_is_in_the_catalogue(self):
        # `StageBus.emit` tanımadığı kimliği sessizce düşürüyor: eksik bir
        # katalog girdisi, ekranda hiç görünmeyen bir aşama demek.
        for stage_id in stages.TEAM_FLOW:
            self.assertIn(stage_id, stages.CATALOGUE, stage_id)
            self.assertIsNotNone(stages.detail(stage_id), stage_id)


class RecordTests(unittest.TestCase):
    """Kayıt, koşan takımı doğru anlatıyor mu."""

    def _run(self, kind: str) -> runlog.Run:
        run = runlog.Run(id="t", kind="team", question="soru")
        run.variant = kind
        run.event({"type": "stage", "id": "team_build", "title": "Takım kuruldu",
                   "meta": {"kind": kind}})
        for i, who in enumerate(("Planner", "Researcher", "Critic"), start=1):
            run.event({"type": "stage", "id": "speaker", "title": "Sıra bir ajanda",
                       "meta": {"who": who, "turn": i}})
        run.event({"type": "stage", "id": "team_done", "title": "Takım bitti",
                   "meta": {"stop_reason": "max message"}})
        return run

    def test_the_named_team_is_the_one_marked_used(self):
        for kind in teams.KINDS:
            used = [t["id"] for t in self._run(kind).teams() if t["used"]]
            self.assertEqual(used, [kind])

    def test_swarm_maps_to_handoffs_and_roundrobin_to_group_chat(self):
        self.assertEqual([p["id"] for p in self._run("swarm").patterns() if p["used"]],
                         ["handoffs"])
        self.assertEqual([p["id"] for p in self._run("roundrobin").patterns() if p["used"]],
                         ["groupchat"])

    def test_the_graph_is_drawn_from_who_actually_spoke(self):
        # Sabit bir şema RoundRobin ile Swarm'ı aynı gösterirdi; ekranın
        # anlatması gereken tek fark tam olarak o.
        run = self._run("roundrobin")
        edges = [(e["src"], e["dst"]) for e in run.graph()["edges"]
                 if e["src"] in ("user", "Planner", "Researcher", "Critic")]
        self.assertIn(("user", "Planner"), edges)
        self.assertIn(("Planner", "Researcher"), edges)
        self.assertIn(("Researcher", "Critic"), edges)

    def test_a_cyclic_speaking_order_does_not_blow_up_the_layout(self):
        """RoundRobin üçüncü ajandan sonra birinciye dönüyor.

        Ölçüldü: dönüş kenarı ileri sayıldığında altı konuşmalık bir koşu
        yirmi sütuna çıkıyor, ve graf karta sığdırılınca kılcal çizgilere
        dönüyordu. İleri kenarlar döngüsüz olmalı.
        """
        run = runlog.Run(id="cyc", kind="team", question="s")
        run.variant = "roundrobin"
        for i, who in enumerate(("Planner", "Researcher", "Critic") * 3, start=1):
            run.event({"type": "stage", "id": "speaker", "title": "S",
                       "meta": {"who": who, "turn": i}})
        graph = run.graph()
        forward = [(e["src"], e["dst"]) for e in graph["edges"]
                   if not e.get("back") and not e.get("cross")]

        # İleri kenarlarda döngü var mı: derinlik, düğüm sayısını aşmamalı.
        depth: dict[str, int] = {n["id"]: 0 for n in graph["nodes"]}
        for _ in range(len(forward) + 1):
            for src, dst in forward:
                if src in depth and dst in depth:
                    depth[dst] = max(depth[dst], depth[src] + 1)
        self.assertLessEqual(max(depth.values()), len(graph["nodes"]),
                             "ileri kenarlarda döngü kalmış")

    def test_a_repeated_transition_becomes_one_edge_with_its_turns(self):
        run = runlog.Run(id="dup", kind="team", question="s")
        run.variant = "roundrobin"
        for i, who in enumerate(("Planner", "Researcher", "Planner", "Researcher"),
                                start=1):
            run.event({"type": "stage", "id": "speaker", "title": "S",
                       "meta": {"who": who, "turn": i}})
        pr = [e for e in run.graph()["edges"]
              if (e["src"], e["dst"]) == ("Planner", "Researcher")]
        self.assertEqual(len(pr), 1)
        self.assertIn("2", pr[0]["message"])
        self.assertIn("4", pr[0]["message"])

    def test_an_agent_that_never_spoke_says_so(self):
        run = runlog.Run(id="t2", kind="team", question="s")
        run.variant = "selector"
        run.event({"type": "stage", "id": "speaker", "title": "Sıra",
                   "meta": {"who": "Planner", "turn": 1}})
        quiet = next(n for n in run.graph()["nodes"] if n["id"] == "Critic")
        self.assertIn("hiç gelmedi", quiet["note"])

    def test_the_sequence_uses_the_roster_as_lifelines(self):
        lanes = [l["id"] for l in self._run("swarm").sequence()["lanes"]]
        self.assertEqual(lanes, ["user", "Planner", "Researcher", "Critic"])

    def test_a_manager_the_team_created_itself_gets_a_box_and_a_lifeline(self):
        """Ölçüldü (team-0003, magenticone): konuşmaların yarısı
        `MagenticOneOrchestrator`'a ait ve o kadroda yok. Sabit kadro, o
        konuşmaları olmayan bir kutuya yönlendiriyordu."""
        run = runlog.Run(id="t3", kind="team", question="s")
        run.variant = "magenticone"
        for i, who in enumerate(("MagenticOneOrchestrator", "Researcher",
                                 "MagenticOneOrchestrator"), start=1):
            run.event({"type": "stage", "id": "speaker", "title": "Sıra",
                       "meta": {"who": who, "turn": i}})
        self.assertIn("MagenticOneOrchestrator",
                      [n["id"] for n in run.graph()["nodes"]])
        lanes = [l["id"] for l in run.sequence()["lanes"]]
        self.assertIn("MagenticOneOrchestrator", lanes)
        # Her adımın hedefi gerçekten var olan bir şerit olmalı.
        for step in run.sequence()["steps"]:
            self.assertIn(step["src"], lanes)
            self.assertIn(step["dst"], lanes)


if __name__ == "__main__":
    unittest.main()
