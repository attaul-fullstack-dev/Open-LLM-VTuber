"""Conversation-history safety net (deterministic tests, temp dirs only).

The stores under test hold a real companion's shared past and have no version
control, so this module exists purely so that a mistake is survivable. These
tests pin the three properties that make it safe to run unattended on every
server start:

- a snapshot can be taken and restores byte-for-byte after a wipe;
- pruning only ever removes this module's own old snapshots, never a live store;
- nothing is written inside the live stores, and every failure is swallowed.
"""

import json
import os
import shutil
import tempfile
import unittest

from src.open_llm_vtuber import history_backup


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        history_backup.BACKUP_ROOT = "backups"
        os.makedirs(os.path.join("chat_history", "c1"), exist_ok=True)
        os.makedirs("episodic", exist_ok=True)
        os.makedirs("character_state", exist_ok=True)
        with open(os.path.join("chat_history", "c1", "a.json"), "w", encoding="utf-8") as h:
            json.dump([{"role": "metadata", "title": "keep me"}], h)
        with open(os.path.join("episodic", "c1.json"), "w", encoding="utf-8") as h:
            json.dump([{"event_text": "an event"}], h)
        with open(os.path.join("character_state", "c1.json"), "w", encoding="utf-8") as h:
            json.dump({"relationship_status": "married"}, h)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_snapshot_creates_a_copy_without_touching_the_live_store(self):
        before = open(os.path.join("chat_history", "c1", "a.json"), encoding="utf-8").read()
        path = history_backup.take_snapshot(reason="unit")
        self.assertTrue(path)
        self.assertTrue(os.path.isdir(path))
        # the live store is byte-identical afterwards
        self.assertEqual(
            open(os.path.join("chat_history", "c1", "a.json"), encoding="utf-8").read(),
            before,
        )
        self.assertTrue(os.path.isfile(os.path.join(path, "chat_history", "c1", "a.json")))
        self.assertTrue(os.path.isfile(os.path.join(path, "character_state", "c1.json")))
        self.assertTrue(os.path.isfile(os.path.join(path, "SNAPSHOT.txt")))

    def test_restore_recovers_a_wiped_store(self):
        path = history_backup.take_snapshot(reason="unit")
        shutil.rmtree("chat_history")
        self.assertFalse(os.path.isdir("chat_history"))
        restored = history_backup.restore_snapshot(os.path.basename(path))
        self.assertIn("chat_history", restored)
        self.assertIn("character_state", restored)
        recovered = json.load(open(os.path.join("chat_history", "c1", "a.json"), encoding="utf-8"))
        self.assertEqual(recovered[0]["title"], "keep me")
        self.assertEqual(
            json.load(open(os.path.join("character_state", "c1.json"), encoding="utf-8"))[
                "relationship_status"
            ],
            "married",
        )

    def test_restore_moves_the_previous_store_aside(self):
        path = history_backup.take_snapshot(reason="unit")
        history_backup.restore_snapshot(os.path.basename(path))
        aside = [n for n in os.listdir(".") if n.startswith("chat_history.replaced-")]
        self.assertTrue(aside, "a mistaken restore must itself be recoverable")

    def test_restore_rejects_an_unknown_snapshot(self):
        with self.assertRaises(FileNotFoundError):
            history_backup.restore_snapshot("snapshot-does-not-exist")

    def test_prune_keeps_the_configured_number_and_never_the_live_store(self):
        for i in range(history_backup.MAX_SNAPSHOTS + 6):
            history_backup.take_snapshot(reason=f"flood{i}")
        snapshots = [
            n for n in os.listdir(history_backup.BACKUP_ROOT)
            if n.startswith("snapshot-")
        ]
        self.assertLessEqual(len(snapshots), history_backup.MAX_SNAPSHOTS)
        self.assertGreaterEqual(len(snapshots), 1)
        # the live store survived the flood untouched
        self.assertTrue(os.path.isfile(os.path.join("chat_history", "c1", "a.json")))

    def test_snapshot_names_are_unique_within_the_same_second(self):
        first = history_backup.take_snapshot(reason="a")
        second = history_backup.take_snapshot(reason="b")
        self.assertNotEqual(first, second)

    def test_empty_environment_is_a_no_op(self):
        for store in history_backup.PROTECTED_STORES:
            if os.path.isdir(store):
                shutil.rmtree(store)
        self.assertEqual(history_backup.take_snapshot(reason="empty"), "")

    def test_failure_is_swallowed_not_raised(self):
        history_backup.BACKUP_ROOT = os.path.join(os.getcwd(), "backups")
        # a file where a directory is expected makes copytree fail
        with open(history_backup.BACKUP_ROOT, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        self.assertEqual(history_backup.take_snapshot(reason="broken"), "")
        # the live store is still there and still intact
        self.assertTrue(os.path.isfile(os.path.join("chat_history", "c1", "a.json")))

    def test_prune_refuses_to_step_outside_the_backup_root(self):
        history_backup.take_snapshot(reason="unit")
        history_backup.BACKUP_ROOT = "backups"
        history_backup._prune(keep=0)
        self.assertTrue(os.path.isfile(os.path.join("chat_history", "c1", "a.json")))


if __name__ == "__main__":
    unittest.main()
