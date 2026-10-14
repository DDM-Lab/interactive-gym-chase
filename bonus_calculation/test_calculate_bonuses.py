"""Payment and recording edge cases using small synthetic rounds."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from calculate_bonuses import calculate_payments


class PaymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / "cramped_room_frame_skip").mkdir()

    def write_round(self, subject, number, reward=1, *, partial=False, reset=True):
        folder = self.data / "cramped_room_frame_skip" / f"cramped_room_frame_skip_round_{number:02d}"
        folder.mkdir(exist_ok=True)
        path = folder / f"{subject}.csv"
        fields = [
            "t", "episode_num", "infos.0.delivery_reward", "infos.1.delivery_reward",
            "terminateds.__all__", "truncateds.__all__",
        ]
        rows = [
            [0, 0, reward, reward, False, False],
            [1, 0, 0, 0, False, False],
            [2, 0, 0, 0, False, True],
        ]
        if partial:
            rows = rows[:2]
        elif reset:
            rows.append([3, 1, "", "", "", ""])
        with path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def complete_subject(self, subject="human", reward=1, *, reset=True):
        for number in range(1, 21):
            self.write_round(subject, number, reward, reset=reset)

    def payments(self):
        return {row["subject_id"]: row for row in calculate_payments(self.data, 3)}

    def test_complete_without_metadata_shared_reward_counted_once(self):
        self.complete_subject()
        # Tutorial deliveries must not contribute to payment.
        tutorial = self.data / "overcooked_tutorial"
        tutorial.mkdir()
        (tutorial / "human.csv").write_text("infos.0.delivery_reward\n999\n")
        result = self.payments()["human"]
        self.assertEqual(result["completion_status"], "completed")
        self.assertFalse(result["completion_metadata_present"])
        self.assertEqual(result["total_delivery_reward"], "20")
        self.assertEqual(result["base_payment"], "4.00")
        self.assertEqual(result["bonus"], "0.40")
        self.assertEqual(result["total_payment"], "4.40")

    def test_zero_deliveries_still_earns_base_and_reset_optional(self):
        self.complete_subject(reward=0, reset=False)
        result = self.payments()["human"]
        self.assertEqual(result["bonus"], "0.00")
        self.assertEqual(result["total_payment"], "4.00")

    def test_incomplete_high_reward_and_metadata_do_not_earn_payment(self):
        for number in range(1, 20):
            self.write_round("human", number, 10)
        end = self.data / "end_completion_code_scene"
        end.mkdir()
        (end / "human_metadata.json").write_text("{}")
        result = self.payments()["human"]
        self.assertTrue(result["completion_metadata_present"])
        self.assertEqual(result["rounds_recorded"], 19)
        self.assertEqual(result["completion_status"], "incomplete")
        self.assertEqual(result["total_delivery_reward"], "190")
        for key in ("base_payment", "bonus", "total_payment"):
            self.assertEqual(result[key], "0.00")

    def test_all_folders_present_but_partial_round_is_incomplete(self):
        self.complete_subject()
        self.write_round("human", 20, partial=True)
        result = self.payments()["human"]
        self.assertEqual(result["rounds_recorded"], 20)
        self.assertEqual(result["rounds_completed"], 19)
        self.assertIn("round 20:", result["recording_issues"])
        self.assertEqual(result["total_payment"], "0.00")

    def test_duplicate_gap_and_missing_terminal_do_not_count_as_full(self):
        for name, change in (
            ("duplicate", lambda text: text.replace("1,0,0,0,False,False", "0,0,0,0,False,False")),
            ("gap", lambda text: text.replace("1,0,0,0,False,False\n", "")),
            ("terminal", lambda text: text.replace("False,True", "False,False")),
        ):
            with self.subTest(name=name):
                self.complete_subject(name)
                path = self.write_round(name, 20)
                path.write_text(change(path.read_text()))
                result = self.payments()[name]
                self.assertEqual(result["total_payment"], "0.00")
                self.assertTrue(result["recording_issues"])

    def test_start_only_subject_included_without_suffix_stripping_id(self):
        start = self.data / "overcooked_start_scene"
        start.mkdir()
        (start / "human_with_underscores_metadata.json").write_text("{}")
        result = self.payments()["human_with_underscores"]
        self.assertEqual(result["rounds_recorded"], 0)
        self.assertEqual(result["total_payment"], "0.00")

    def test_invalid_rewards_fail_instead_of_silently_underpaying(self):
        for reward in ("NaN", "Infinity", "-1", "oops", "0.5"):
            with self.subTest(reward=reward):
                self.write_round("human", 1, reward)
                with self.assertRaises(ValueError):
                    self.payments()

    def test_missing_reward_column_fails(self):
        path = self.write_round("human", 1)
        path.write_text(path.read_text().replace("infos.0.delivery_reward", "other"))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.payments()


if __name__ == "__main__":
    unittest.main()
