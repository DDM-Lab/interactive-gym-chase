"""Synchronous, headless episode execution and per-step CSV logging."""

import time

import numpy as np

COMPONENTS = ("delivery_reward", "delivery_act_reward", "onion_in_pot_reward", "soup_in_dish_reward")


class GamePlayer:
    def __init__(self, env, policy, actual_frame_skips=(5, 5), reported_frame_skips=(1, 1), fps=30):
        self.env = env
        self.policy = policy
        self.actual_frame_skips = actual_frame_skips
        self.reported_frame_skips = reported_frame_skips
        self.fps = fps
        self.agent_ids = tuple(env.agent_ids)
        self.noop = env.action_set.index("no-op")

    def run_episode(self, participant, episode, seed, writer):
        obs, _ = self.env.reset(seed=seed)
        # Independent action streams avoid coupling agent 1's randomness to agent 0's cadence.
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        started = time.perf_counter()
        cumulative = 0.0
        decisions = [0, 0]
        steps = 0
        while True:
            decided = [steps % skip == 0 for skip in self.actual_frame_skips]
            actions = {
                aid: self.policy.action(obs[aid], rngs[i]) if decided[i] else self.noop
                for i, aid in enumerate(self.agent_ids)
            }
            obs, rewards, terms, truncs, infos = self.env.step(actions)
            steps += 1
            team_reward = float(rewards[self.agent_ids[0]])
            if any(float(rewards[aid]) != team_reward for aid in self.agent_ids):
                raise RuntimeError("Expected the environment's common team reward.")
            cumulative += team_reward
            row = {
                "timestamp": time.time(), "participant": participant,
                "episode": episode, "seed": seed, "step": steps,
                "episode_elapsed_time": time.perf_counter() - started,
                "simulated_elapsed_time": steps / self.fps,
                # Legacy field is meaningful only when both agents have the same cadence.
                "AI_speed": self.actual_frame_skips[0] if len(set(self.actual_frame_skips)) == 1 else "",
            }
            for i, aid in enumerate(self.agent_ids):
                decisions[i] += int(decided[i])
                prefix = f"agent_{aid}_"
                row.update({
                    prefix + "action": actions[aid],
                    prefix + "shared_reward": float(rewards[aid]),
                    prefix + "actual_frame_skip": self.actual_frame_skips[i],
                    prefix + "reported_frame_skip": self.reported_frame_skips[i],
                    prefix + "decided": int(decided[i]),
                    prefix + "terminated": bool(terms[aid]),
                    prefix + "truncated": bool(truncs[aid]),
                })
                for name in COMPONENTS:
                    row[prefix + name] = float(infos[aid][name])
            row["cumulative_reward"] = cumulative
            if writer.fieldnames is None:
                writer.fieldnames = list(row)
                writer.writeheader()
            writer.writerow(row)
            if all(terms[aid] or truncs[aid] for aid in self.agent_ids):
                break
            if steps >= self.env.config["max_steps"]:
                raise RuntimeError("Environment failed to terminate at its configured step limit.")
        return {
            "participant": participant, "episode": episode, "seed": seed,
            "steps": steps, "team_reward": cumulative,
            "agent_0_decisions": decisions[0], "agent_1_decisions": decisions[1],
            "terminated": all(terms[aid] for aid in self.agent_ids),
            "truncated": any(truncs[aid] for aid in self.agent_ids),
            "wall_clock_seconds": time.perf_counter() - started,
            "simulated_seconds": steps / self.fps,
        }
