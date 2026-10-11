"""History write safety (deterministic, isolated).

``store_message`` reads the existing transcript, appends one row, and writes
the file back atomically. If the existing file cannot be read (corrupt
truncated write, concurrent writer, disk error), the only non-destructive
behaviour is to refuse the write and keep the file untouched: falling through
with an empty list would collapse the whole conversation into the single new
message.

All tests run in a temporary working directory; production `chat_history/`
is never touched.
"""

import json
import os
import tempfile
import unittest

CONF = "writechar"
HIST = "h-write-safe"


def history_path():
    return os.path.join("chat_history", CONF, f"{HIST}.json")


def read_rows():
    with open(history_path(), encoding="utf-8") as handle:
        return json.load(handle)


class StoreMessageWriteSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        # store_message appends to an existing file; the history directory
        # is created by the history-creation path in production.
        os.makedirs(os.path.join("chat_history", CONF), exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_unreadable_file_is_preserved_not_collapsed(self):
        from src.open_llm_vtuber.chat_history_manager import store_message

        store_message(CONF, HIST, "human", "pesan pertama")
        store_message(CONF, HIST, "ai", "jawaban pertama")
        self.assertEqual(len(read_rows()), 2)

        # corrupt the file the way a truncated write would
        with open(history_path(), "w", encoding="utf-8") as handle:
            handle.write("{not valid json!!!")
        raw = open(history_path(), encoding="utf-8").read()

        store_message(CONF, HIST, "ai", "jawaban kedua")

        self.assertEqual(
            open(history_path(), encoding="utf-8").read(),
            raw,
            "an unreadable transcript must be left byte-identical",
        )

    def test_normal_appends_still_work(self):
        from src.open_llm_vtuber.chat_history_manager import store_message

        store_message(CONF, HIST, "human", "halo", name="User", avatar="u.png")
        store_message(CONF, HIST, "ai", "hai juga")
        rows = read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["role"], "human")
        self.assertEqual(rows[0]["content"], "halo")
        self.assertEqual(rows[0]["name"], "User")
        self.assertEqual(rows[0]["avatar"], "u.png")

    def test_missing_ids_are_refused(self):
        from src.open_llm_vtuber.chat_history_manager import store_message

        store_message("", HIST, "human", "x")
        store_message(CONF, "", "human", "x")
        store_message(CONF, HIST, "human", "ok")
        self.assertEqual(len(read_rows()), 1)


if __name__ == "__main__":
    unittest.main()
