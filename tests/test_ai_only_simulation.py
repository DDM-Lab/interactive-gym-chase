"""Integration checks with the real shipped model and CoGrid 0.3.2 environment."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_rl_policies.environment import make_environment
from run_rl_policies.experiment import ExperimentConfig, run_experiment
from run_rl_policies.game_player import GamePlayer
from run_rl_policies.onnx_policy import ONNXPolicy


class AIOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = ONNXPolicy()

    def test_reported_skip_is_focal_only_and_survives_reset_and_step(self):
        env = make_environment((1, 1), max_steps=10)
        self.addCleanup(env.close)
        noop = {aid: env.action_set.index("no-op") for aid in (0, 1)}
        for focal in (0, 1):
            baseline = []
            env.reset(seed=42)
            baseline.append(env.reset(seed=42)[0])
            baseline.append(env.step(noop)[0])
            target_indices = set()
            source_indices = set()
            for reported in (5, 10, 15):
                skips = [1, 1]
                skips[focal] = reported
                changed = [env.reset(seed=42, options={"frame_skip": skips})[0]]
                changed.append(env.step(noop)[0])
                for before, after in zip(baseline, changed):
                    np.testing.assert_array_equal(before[1-focal], after[1-focal])
                    delta = after[focal] - before[focal]
                    self.assertEqual(np.count_nonzero(delta), 2)
                    source = np.flatnonzero(delta == -1)
                    target = np.flatnonzero(delta == 1)
                    self.assertEqual(len(source), 1)
                    self.assertEqual(len(target), 1)
                    source_indices.add(int(source[0]))
                    target_indices.add(int(target[0]))
                np.testing.assert_array_equal(
                    env._env_state.extra_state["overcooked.frame_skip_countdown"], [0, 0]
                )
            self.assertEqual(len(source_indices), 1)
            self.assertEqual(len(target_indices), 3)
        # Constructor settings must be restored after an episode-specific override.
        env.reset(seed=42)
        np.testing.assert_array_equal(env._env_state.extra_state["overcooked.frame_skip"], [1, 1])

    def test_model_contract_and_independent_cadences(self):
        env = make_environment((5, 10), max_steps=41)
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=42)
        for aid in (0, 1):
            self.assertEqual(obs[aid].shape, (896,))
            self.assertEqual(self.policy.logits(obs[aid]).shape, (7,))
        with self.assertRaisesRegex(ValueError, "896 features"):
            self.policy.logits(np.zeros(100))
        player = GamePlayer(env, self.policy, (2, 5), (5, 10))
        buffer = io.StringIO()
        summary = player.run_episode(0, 0, 42, csv.DictWriter(buffer, fieldnames=None))
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(rows), 41)
        self.assertEqual((summary["agent_0_decisions"], summary["agent_1_decisions"]), (21, 9))
        for i, row in enumerate(rows):
            for aid, skip in enumerate((2, 5)):
                self.assertEqual(int(row[f"agent_{aid}_decided"]), int(i % skip == 0))
                if i % skip:
                    self.assertEqual(int(row[f"agent_{aid}_action"]), env.action_set.index("no-op"))
            self.assertEqual(row["AI_speed"], "")
        self.assertEqual(float(rows[-1]["cumulative_reward"]), summary["team_reward"])
        self.assertTrue(summary["truncated"])

    def test_runs_reproduce_actions_rewards_and_write_complete_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                participants=2, episodes=2, max_steps=35,
                actual_frame_skips=(2, 5), reported_frame_skips=(5, 10),
                output_dir=Path(tmp),
            )
            runs = [run_experiment(config), run_experiment(config)]
            self.assertNotEqual(*runs)
            for participant in (0, 1):
                trajectories = []
                for run in runs:
                    path = next(run.glob(f"cramped_room_{participant}_*game_data.csv"))
                    with path.open(newline="", encoding="utf-8") as file:
                        rows = list(csv.DictReader(file))
                    self.assertEqual(len(rows), 70)
                    self.assertEqual({int(row["seed"]) for row in rows}, {42+participant*2, 43+participant*2})
                    # Wall-clock fields intentionally differ between runs.
                    ignored = {"timestamp", "episode_elapsed_time"}
                    trajectories.append([{k: v for k, v in row.items() if k not in ignored} for row in rows])
                self.assertEqual(*trajectories)
            metadata = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["completed_episodes"], 4)
            self.assertEqual(metadata["config"]["reported_frame_skips"], [5, 10])
            self.assertEqual(len(metadata["model_sha256"]), 64)
            with (runs[0] / "episode_summary.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 4)

    def test_invalid_settings_fail_before_running(self):
        for kwargs in (
            {"participants": 0}, {"episodes": 0}, {"max_steps": 0},
            {"actual_frame_skips": (0, 5)}, {"reported_frame_skips": (2, 5)},
            {"seed": -1}, {"seed": 2**32 - 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
