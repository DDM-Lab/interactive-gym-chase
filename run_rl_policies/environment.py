"""Reuse the browser experiment's environment without importing its server."""

from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / (
    "interactive_gym/examples/cogrid/pyodide_overcooked/env_initialization/"
    "cramped_room_frame_skip_environment_initialization.py"
)


@lru_cache(maxsize=1)
def _definitions():
    if version("cogrid") != "0.3.2":
        raise RuntimeError(
            "AI-only simulations require cogrid==0.3.2. Install "
            "run_rl_policies/requirements.txt in a separate environment."
        )
    # This is also how tests/test_frame_skip_component_logging.py loads it.
    definitions = runpy.run_path(str(INIT_PATH))
    definitions["env"].close()
    return definitions


def make_environment(reported_frame_skips=(1, 1), max_steps=1350):
    definitions = _definitions()
    allowed = definitions["FRAME_SKIP_VALUES"]
    if len(reported_frame_skips) != 2 or any(k not in allowed for k in reported_frame_skips):
        raise ValueError(f"Provide two reported frame skips from {allowed}.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    config = deepcopy(definitions["overcooked_config"])
    config.update(num_agents=2, max_steps=max_steps)
    return definitions["InteractiveGymFrameSkipOvercooked"](
        config=config,
        agent_frame_skips=tuple(reported_frame_skips),
        agent_class=definitions["OvercookedAgent"],
    )
