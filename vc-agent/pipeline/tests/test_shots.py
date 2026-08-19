"""Kamera karesi: adlandırma, saklama, ve asıl mesele — yol kaçışı.

Bu modülün tek gerçek riski `path_for`. Bir web sunucusunda kimliğe göre dosya
veren kodun sessizce yanlış yazılması, `../../.ssh/id_rsa` okunması demek. O
yüzden testlerin çoğu orada yoğunlaşıyor ve saldırı biçimlerini tek tek deniyor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shots  # noqa: E402


class IdentityTests(unittest.TestCase):
    def test_new_id_is_thirty_two_hex(self):
        for _ in range(20):
            self.assertRegex(shots.new_id(), r"^[0-9a-f]{32}$")

    def test_ids_do_not_repeat(self):
        self.assertEqual(len({shots.new_id() for _ in range(200)}), 200)


class PathEscapeTests(unittest.TestCase):
    """Kimlik hiçbir zaman bir yol parçası olmuyor."""

    def test_a_valid_id_lands_inside_the_directory(self):
        target = shots.path_for("a" * 32)
        self.assertTrue(target.is_relative_to(shots.SHOTS.resolve()))
        self.assertEqual(target.suffix, ".jpg")

    def test_traversal_is_refused_before_a_path_is_built(self):
        for hostile in (
            "../../../../etc/passwd",
            "..%2f..%2fetc%2fpasswd",
            "/etc/passwd",
            "a" * 31 + "/",
            "../" + "a" * 29,
            "\x00" + "a" * 31,
        ):
            with self.subTest(hostile=hostile):
                with self.assertRaises(shots.ShotError):
                    shots.path_for(hostile)

    def test_wrong_length_or_alphabet_is_refused(self):
        for bad in ("", "abc", "A" * 32, "g" * 32, "a" * 31, "a" * 33, "a" * 32 + ".jpg"):
            with self.subTest(bad=bad):
                with self.assertRaises(shots.ShotError):
                    shots.path_for(bad)

    def test_exists_says_false_rather_than_raising_on_a_hostile_id(self):
        # The route asks `exists` before `read`; a raise there would turn a
        # probe into a 500 and tell the prober their input reached the code.
        self.assertFalse(shots.exists("../../etc/passwd"))


class CommandTests(unittest.TestCase):
    def test_the_command_writes_where_we_said(self):
        shot_id = "b" * 32
        command = shots.command(shot_id)
        self.assertIn(str(shots.path_for(shot_id)), command)
        self.assertIn("-frames:v 1", command)

    def test_the_sentence_carries_the_command_verbatim(self):
        # What the approval card shows must be what runs. If the sentence ever
        # paraphrased the command, the gate would be signing something else.
        shot_id = "c" * 32
        self.assertIn(shots.command(shot_id), shots.sentence(shot_id))

    def test_the_device_is_ours_to_choose_not_the_models(self):
        self.assertIn("/dev/video1", shots.command("d" * 32, device="/dev/video1"))


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self._real = shots.SHOTS
        self._tmp = Path(__file__).resolve().parent / "_shots_tmp"
        shots.SHOTS = self._tmp

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)
        shots.SHOTS = self._real

    def _write(self, n: int) -> list[str]:
        import time

        self._tmp.mkdir(parents=True, exist_ok=True)
        ids = []
        for i in range(n):
            shot_id = f"{i:032x}"
            (self._tmp / f"{shot_id}.jpg").write_bytes(b"\xff\xd8\xff")
            # mtime ordering is what `recent` and `prune` sort on.
            time.sleep(0.01)
            ids.append(shot_id)
        return ids

    def test_prune_keeps_the_newest_and_drops_the_rest(self):
        ids = self._write(8)
        dropped = shots.prune(keep=3)
        self.assertEqual(dropped, 5)
        left = {row["id"] for row in shots.recent(limit=99)}
        self.assertEqual(left, set(ids[-3:]))

    def test_recent_is_newest_first(self):
        ids = self._write(4)
        self.assertEqual([row["id"] for row in shots.recent()], list(reversed(ids)))

    def test_clear_removes_everything(self):
        self._write(3)
        self.assertEqual(shots.clear(), 3)
        self.assertEqual(shots.recent(), [])

    def test_recent_on_a_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(shots.recent(), [])

    def test_read_refuses_a_frame_that_is_not_there(self):
        with self.assertRaises(shots.ShotError):
            shots.read("e" * 32)


if __name__ == "__main__":
    unittest.main()
