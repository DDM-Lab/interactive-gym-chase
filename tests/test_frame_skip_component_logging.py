"""Run with CoGrid 0.3.2: python -m unittest discover -s tests."""

import dataclasses
from pathlib import Path
import runpy
import unittest

import numpy as np


INIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "interactive_gym/examples/cogrid/pyodide_overcooked/env_initialization"
    / "cramped_room_frame_skip_environment_initialization.py"
)
COMPONENTS = (
    "delivery_reward",
    "delivery_act_reward",
    "onion_in_pot_reward",
    "soup_in_dish_reward",
)


class ComponentLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = runpy.run_path(str(INIT_PATH))["env"]

    def setUp(self):
        self.env.reset(seed=42)

    def prepare_interaction(self, agent_id, held, *, delivery=False, count=0, timer=30):
        """Set real array state so the chosen agent faces the pot or delivery."""
        state = self.env._env_state
        pos, direction, inventory = (
            state.agent_pos.copy(), state.agent_dir.copy(), state.agent_inv.copy()
        )
        pos[agent_id] = [2, 3] if delivery else [1, 2]
        direction[agent_id] = 1 if delivery else 3
        inventory[agent_id, 0] = self.env._reward_config["type_ids"][held]
        extra = dict(state.extra_state)
        contents = extra["overcooked.pot_contents"].copy()
        contents[:] = -1
        contents[0, :count] = self.env._reward_config["type_ids"]["onion"]
        extra["overcooked.pot_contents"] = contents
        extra["overcooked.pot_timer"] = np.array([timer], dtype=np.int32)
        self.env._env_state = dataclasses.replace(
            state, agent_pos=pos, agent_dir=direction, agent_inv=inventory,
            extra_state=extra,
        )
        self.env._sync_objects_from_state()

    def interact(self, agent_id):
        actions = {aid: self.env.action_set.index("no-op") for aid in (0, 1)}
        actions[agent_id] = self.env.action_set.index("pick_up_or_drop")
        return self.env.step(actions)

    def test_onion_placement_both_agents_and_full_pot(self):
        for aid in (0, 1):
            with self.subTest(agent=aid):
                self.env.reset(seed=42)
                self.prepare_interaction(aid, "onion")
                _, rewards, _, _, infos = self.interact(aid)
                self.assertEqual(infos[aid]["onion_in_pot_reward"], 1.0)
                self.assertEqual(infos[1-aid]["onion_in_pot_reward"], 0.0)
                self.assertEqual(rewards, {0: 0.0, 1: 0.0})
                self.assertEqual(self.env._env_state.agent_inv[aid, 0], -1)
        self.env.reset(seed=42)
        self.prepare_interaction(0, "onion", count=3)
        self.assertEqual(self.interact(0)[4][0]["onion_in_pot_reward"], 0.0)

    def test_plating_requires_cooked_full_pot(self):
        for count, timer, expected in ((3, 0, 1.0), (3, 10, 0.0), (0, 0, 0.0)):
            with self.subTest(count=count, timer=timer):
                self.env.reset(seed=42)
                self.prepare_interaction(1, "plate", count=count, timer=timer)
                _, rewards, _, _, infos = self.interact(1)
                self.assertEqual(infos[1]["soup_in_dish_reward"], expected)
                self.assertEqual(infos[0]["soup_in_dish_reward"], 0.0)
                self.assertEqual(rewards, {0: 0.0, 1: 0.0})
                if expected:
                    self.assertEqual(
                        self.env._env_state.agent_inv[1, 0],
                        self.env._reward_config["type_ids"]["onion_soup"],
                    )

    def test_delivery_shared_score_and_individual_credit(self):
        for aid in (0, 1):
            with self.subTest(agent=aid):
                self.env.reset(seed=42)
                self.prepare_interaction(aid, "onion_soup", delivery=True)
                _, rewards, _, _, infos = self.interact(aid)
                self.assertEqual(rewards, {0: 1.0, 1: 1.0})
                self.assertEqual(infos[aid]["delivery_act_reward"], 1.0)
                self.assertEqual(infos[1-aid]["delivery_act_reward"], 0.0)
                for components in infos.values():
                    self.assertEqual(components["delivery_reward"], 1.0)

    def test_noop_does_not_log_interaction_and_schema_matches_archive(self):
        self.prepare_interaction(0, "onion_soup", delivery=True)
        _, _, _, _, infos = self.env.step({0: 6, 1: 6})
        for components in infos.values():
            self.assertEqual([components[key] for key in COMPONENTS], [0.0]*4)
            self.assertEqual(
                list(components), list(COMPONENTS) + ["row", "col", "direction", "layout_id"]
            )

    def test_reset_does_not_add_component_sample(self):
        _, infos = self.env.reset(seed=42)
        for components in infos.values():
            self.assertFalse(set(COMPONENTS).intersection(components))


if __name__ == "__main__":
    unittest.main()
