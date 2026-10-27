"""Participant/episode orchestration, output ownership, and run provenance."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

from run_rl_policies.environment import INIT_PATH, ROOT, make_environment
from run_rl_policies.game_player import GamePlayer
from run_rl_policies.onnx_policy import DEFAULT_MODEL, ONNXPolicy


@dataclass(frozen=True)
class ExperimentConfig:
    participants: int = 5
    episodes: int = 20
    max_steps: int = 1350
    actual_frame_skips: tuple[int, int] = (5, 5)
    reported_frame_skips: tuple[int, int] = (1, 1)
    seed: int = 42
    fps: int = 30
    output_dir: Path = ROOT / "data/ai_only"
    model_path: Path = DEFAULT_MODEL

    def __post_init__(self):
        for name in ("participants", "episodes", "max_steps", "fps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.actual_frame_skips) != 2 or any(
            not isinstance(k, int) or k < 1 for k in self.actual_frame_skips
        ):
            raise ValueError("Provide two positive integer actual frame skips")
        if len(self.reported_frame_skips) != 2 or any(
            k not in (1, 5, 10, 15) for k in self.reported_frame_skips
        ):
            raise ValueError("Reported frame skips must each be 1, 5, 10, or 15")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - self.participants * self.episodes:
            raise ValueError("Episode seeds must fit in unsigned 32-bit integers")


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(config: ExperimentConfig) -> Path:
    policy = ONNXPolicy(config.model_path)
    env = make_environment(config.reported_frame_skips, config.max_steps)
    try:
        obs, _ = env.reset(seed=config.seed)
        agent_ids = tuple(env.agent_ids)
        if agent_ids != (0, 1):
            raise ValueError(f"Expected agent IDs (0, 1), got {agent_ids}")
        for aid in agent_ids:
            logits = policy.logits(obs[aid])
            if logits.shape != (len(env.action_set),):
                raise ValueError("Policy action count does not match the environment action set")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
        run_dir = Path(config.output_dir).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1, "status": "running", "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {key: str(value.resolve()) if isinstance(value, Path) else value
                       for key, value in asdict(config).items()},
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_branch": _git("branch", "--show-current"),
            "repository_status": _git("status", "--porcelain"),
            "model_sha256": _sha256(policy.model_path),
            "environment_source": str(INIT_PATH.relative_to(ROOT)),
            "environment_sha256": _sha256(INIT_PATH),
            "python": sys.version, "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
            "policy": policy.metadata(), "action_set": list(env.action_set),
            "features": list(env.config["features"]),
            "observation_shapes": {str(aid): list(obs[aid].shape) for aid in agent_ids},
            "seed_scheme": "base_seed + participant_index * episodes + episode_index; zero-based indices",
            "action_sampling": "softmax; independent NumPy generators per agent, spawned from episode seed",
            "timing": "synchronous; decisions at zero-based steps 0, k, 2k; no-ops between decisions",
            "episode_numbering": "zero-based; CSV step is one-based after transition",
            "completed_episodes": 0,
        }
        metadata_path = run_dir / "metadata.json"

        def save_metadata():
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        save_metadata()
        print(f"Output directory: {run_dir}", flush=True)
        player = GamePlayer(env, policy, config.actual_frame_skips, config.reported_frame_skips, config.fps)
        try:
            with (run_dir / "episode_summary.csv").open("x", newline="", encoding="utf-8") as summary_file:
                summary_writer = csv.DictWriter(summary_file, fieldnames=None)
                for participant in range(config.participants):
                    filename = f"cramped_room_{participant}_agent0_frame_skip_agent1_frame_skip_game_data.csv"
                    with (run_dir / filename).open("x", newline="", encoding="utf-8") as data_file:
                        writer = csv.DictWriter(data_file, fieldnames=None)
                        for episode in range(config.episodes):
                            seed = config.seed + participant * config.episodes + episode
                            summary = player.run_episode(participant, episode, seed, writer)
                            if summary_writer.fieldnames is None:
                                summary_writer.fieldnames = list(summary)
                                summary_writer.writeheader()
                            summary_writer.writerow(summary)
                            data_file.flush()
                            summary_file.flush()
                            metadata["completed_episodes"] += 1
                            save_metadata()
                            print(
                                f"Participant {participant + 1}/{config.participants}, "
                                f"episode {episode + 1}/{config.episodes}: "
                                f"reward={summary['team_reward']:g}, steps={summary['steps']}",
                                flush=True,
                            )
            metadata["status"] = "complete"
        except BaseException as error:
            metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            metadata["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_metadata()
        return run_dir
    finally:
        env.close()
