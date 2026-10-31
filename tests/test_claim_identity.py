"""Claim-owner identity and startup recovery (isolated, no provider calls).

The recovery rule is intentionally narrow: a claim is cleared only when its
process token proves the owning process is dead (reboot, vanished PID, reused
PID). A live process, a legacy owner without a token, or unreadable ``/proc``
data all fall back to the 600 s lease, so recovery can never steal an active
claim from a second instance.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.open_llm_vtuber import claim_identity
from src.open_llm_vtuber.character_state import (
    load_character_state,
    recover_stale_claims,
    save_character_state,
)

CONF = "claim-identity-synthetic"
JKT = "Asia/Jakarta"


class OwnerIdentityTest(unittest.TestCase):
    def test_live_process_token_is_never_dead(self):
        owner = claim_identity.claim_owner("client:abc")
        self.assertIn("|", owner)
        self.assertFalse(claim_identity.owner_is_dead(owner))

    def test_legacy_owner_without_token_is_never_dead(self):
        self.assertFalse(claim_identity.owner_is_dead("old-client:abc"))
        self.assertFalse(claim_identity.owner_is_dead(""))

    def test_different_boot_proves_the_owner_is_dead(self):
        owner = "linux:00000000000000000000000000000000:4026531836:999999:1|old:abc"
        self.assertTrue(claim_identity.owner_is_dead(owner))

    def test_vanished_pid_is_dead(self):
        boot = claim_identity._boot_id()
        token = f"linux:{boot}:{claim_identity._pid_namespace()}:123:456"
        owner = f"{token}|old:abc"
        with patch.object(claim_identity, "_starttime", return_value=None), patch.object(
            claim_identity, "_pid_alive", return_value=False
        ):
            self.assertTrue(claim_identity.owner_is_dead(owner))

    def test_reused_pid_is_dead(self):
        boot = claim_identity._boot_id()
        token = f"linux:{boot}:{claim_identity._pid_namespace()}:123:456"
        owner = f"{token}|old:abc"
        with patch.object(claim_identity, "_starttime", return_value=999):
            self.assertTrue(claim_identity.owner_is_dead(owner))

    def test_unreadable_proc_fails_safe(self):
        boot = claim_identity._boot_id()
        token = f"linux:{boot}:{claim_identity._pid_namespace()}:123:456"
        owner = f"{token}|old:abc"
        with patch.object(claim_identity, "_starttime", return_value=None), patch.object(
            claim_identity, "_pid_alive", return_value=True
        ):
            self.assertFalse(claim_identity.owner_is_dead(owner))

    def test_current_token_is_never_dead(self):
        boot = claim_identity._boot_id()
        token = f"linux:{boot}:{claim_identity._pid_namespace()}:123:456"
        owner = f"{token}|client:abc"
        self.assertFalse(
            claim_identity.owner_is_dead(owner, current_token=token)
        )


class StaleClaimRecoveryTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.conf_uid = CONF

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _seed(self, *, status="pending", claimed_by=None, claim_delta=600):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": "ingetin aku minum obat ya",
            "due_at": (base - timedelta(minutes=5)).isoformat(),
            "created_at": base.isoformat(),
            "status": status,
            "source": "conversation",
            "kind": "reminder",
            "tz": JKT,
            "delivery_text": "minum obat",
        }
        if claimed_by:
            row["claimed_by"] = claimed_by
            row["claim_until"] = (
                base + timedelta(seconds=claim_delta)
            ).isoformat()
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        return row

    def _row(self, row_id):
        return next(
            item
            for item in load_character_state(self.conf_uid).future_intentions
            if item["id"] == row_id
        )

    def test_recovery_clears_only_provably_dead_pending_claims(self):
        dead = self._seed(
            claimed_by=(
                "linux:00000000000000000000000000000000:4026531836:999999:1"
                "|old:abc"
            )
        )
        legacy = self._seed(claimed_by="old-version:abc")
        live = self._seed(claimed_by=claim_identity.claim_owner("client:live"))
        done = self._seed(
            status="done",
            claimed_by=(
                "linux:00000000000000000000000000000000:4026531836:999999:1"
                "|old:abc"
            ),
        )

        cleared = recover_stale_claims(self.conf_uid)

        self.assertEqual(cleared, 1)
        dead_after = self._row(dead["id"])
        self.assertEqual(dead_after["status"], "pending")
        self.assertNotIn("claimed_by", dead_after)
        self.assertNotIn("claim_until", dead_after)

        # A legacy owner and a live process keep their lease-bound claim.
        self.assertEqual(self._row(legacy["id"])["claimed_by"], "old-version:abc")
        self.assertEqual(
            self._row(live["id"])["claimed_by"],
            claim_identity.claim_owner("client:live"),
        )
        # Done rows are terminal and are never rewritten by recovery.
        self.assertEqual(self._row(done["id"])["status"], "done")

    def test_recovery_is_idempotent(self):
        dead = self._seed(
            claimed_by=(
                "linux:00000000000000000000000000000000:4026531836:999999:1"
                "|old:abc"
            )
        )
        self.assertEqual(recover_stale_claims(self.conf_uid), 1)
        self.assertEqual(recover_stale_claims(self.conf_uid), 0)
        self.assertNotIn("claimed_by", self._row(dead["id"]))

    def test_write_failure_reports_retryable(self):
        self._seed(
            claimed_by=(
                "linux:00000000000000000000000000000000:4026531836:999999:1"
                "|old:abc"
            )
        )
        with patch(
            "src.open_llm_vtuber.character_state.save_character_state",
            return_value=False,
        ):
            self.assertIsNone(recover_stale_claims(self.conf_uid))


if __name__ == "__main__":
    unittest.main()
