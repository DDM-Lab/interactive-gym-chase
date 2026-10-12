"""Cramped Room for CoGrid 0.3.x, with the frame-skip observation used in training.

This file is executed by Pyodide and must produce an ``env`` at module scope.
``scenes.cramped_room_frame_skip_init_code()`` prepends overrides for the
settings below, so one file serves both the tutorial and the AI scenes. Because
code is prepended (and Pyodide prepends its own), this file can't use
``from __future__`` imports.

The policy was trained with ``examples/frame_skip/train_frame_skip_jax.py`` in
the CoGrid repo. Its observation is the stock ``Overcooked-CrampedRoom-V0``
feature set plus a one-hot of the agent's *own* frame skip over
``FRAME_SKIP_VALUES``. ``FrameSkipOneHot`` and ``FrameSkipOvercookedEnv`` below
are copied from ``examples/frame_skip/frame_skip_env.py`` so the observation is
built exactly as it was in training.

Interactive Gym owns the AI's decision cadence (``GymScene.frame_skip``): it
queries the policy every ``k`` steps and submits no-ops in between. The env only
reports a frame skip in the observation, ``AI_OBSERVED_FRAME_SKIP``, which need
not equal ``k``. It does not mask actions, because ONNX
inference runs asynchronously in the browser, so the AI's action can land a step
after the one it was requested on, and env-side masking would discard it.
"""

import copy
import dataclasses
import typing

import numpy as np
from cogrid import backend
from cogrid.cogrid_env import CoGridEnv
from cogrid.core import actions as grid_actions
from cogrid.core import objects as grid_object
from cogrid.core.features import Feature, register_feature_type
from cogrid.core.pipeline.step import envstate_to_dict
from cogrid.envs import registry
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.rewards import DeliveryReward
from cogrid.feature_space.features import get_all_agent_obs

# ---- Settings (scenes.py prepends overrides for these) ----

# 1 for the solo tutorial, 2 for human + AI.
NUM_AGENTS = globals().get("NUM_AGENTS", 2)
# Frame skip reported in the AI's (agent 1's) observation; must be one of
# FRAME_SKIP_VALUES. The AI's actual cadence is the scene's
# ``.policies(frame_skip=...)``, which this file never sees.
AI_OBSERVED_FRAME_SKIP = globals().get("AI_OBSERVED_FRAME_SKIP", 1)
MAX_STEPS = globals().get("MAX_STEPS", 1350)

# Must match FRAME_SKIP_VALUES in the training script: it sets the one-hot
# width and order, and so the policy's input.
FRAME_SKIP_VALUES = (1, 5, 10, 15)
# The human (agent 0) acts every step; this only affects its own (unused) obs.
HUMAN_FRAME_SKIP = 1


# ---- Frame skip (copied from cogrid examples/frame_skip/frame_skip_env.py) ----

FRAME_SKIP_FEATURE = "frame_skip"
FRAME_SKIP_KEY = "frame_skip"  # extra_state key (stored as "<scope>.frame_skip")
COUNTDOWN_KEY = "frame_skip_countdown"  # steps until each agent's next decision
DEFAULT_FRAME_SKIP_VALUES = (1, 5, 10, 15)


def _frame_skip_values(env_config):
    """Read the tuple of allowed frame-skip values from an env config."""
    if env_config is not None and "frame_skip_values" in env_config:
        return tuple(int(k) for k in env_config["frame_skip_values"])
    return DEFAULT_FRAME_SKIP_VALUES


@register_feature_type(FRAME_SKIP_FEATURE, scope="overcooked")
class FrameSkipOneHot(Feature):
    """One-hot of the focal agent's current frame skip over the configured values."""

    per_agent = True
    focal_only = True
    obs_dim = len(DEFAULT_FRAME_SKIP_VALUES)

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        """Return the number of allowed frame-skip values."""
        return len(_frame_skip_values(env_config))

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        """Build ``fn(state, agent_idx) -> (len(values),) int32`` one-hot."""
        values = _frame_skip_values(env_config)

        def fn(state, agent_idx):
            from cogrid.backend import xp

            return (xp.array(values) == state.frame_skip[agent_idx]).astype(xp.int32)

        return fn


class FrameSkipOvercookedEnv(CoGridEnv):
    """CoGrid Overcooked env where each agent acts every ``k_i`` steps.

    Trimmed copy of the training env for the numpy backend with fixed frame
    skips: no random sampling or mid-episode resampling.
    """

    def __init__(self, config, frame_skip_values, **kwargs):
        config = copy.deepcopy(config)
        config["frame_skip_values"] = tuple(frame_skip_values)
        if FRAME_SKIP_FEATURE not in config["features"]:
            config["features"] = list(config["features"]) + [FRAME_SKIP_FEATURE]

        self.frame_skip_values = _frame_skip_values(config)
        super().__init__(config, **kwargs)
        self._noop_idx = self.action_set.index(grid_actions.Actions.Noop)

    def _build_layout_arrays(self):
        """Add frame-skip placeholders so they become part of extra_state."""
        layout_arrays, spawn_positions, action_set_name = super()._build_layout_arrays()
        n_agents = self.config["num_agents"]
        layout_arrays[FRAME_SKIP_KEY] = backend.xp.ones(n_agents, dtype=np.int32)
        layout_arrays[COUNTDOWN_KEY] = backend.xp.zeros(n_agents, dtype=np.int32)
        return layout_arrays, spawn_positions, action_set_name

    def _build_pipeline(self, layout_arrays, spawn_positions, action_set_name, seed):
        """Wrap the base step fn with frame-skip action masking."""
        obs = super()._build_pipeline(layout_arrays, spawn_positions, action_set_name, seed)
        self._step_fn = self._wrap_step_fn(self._step_fn)
        return obs

    def _with_frame_skip(self, state, frame_skip, countdown=None):
        """Return ``state`` with new frame skips (and countdowns) in extra_state."""
        extra = dict(state.extra_state)
        extra[f"{self.scope}.{FRAME_SKIP_KEY}"] = frame_skip
        if countdown is not None:
            extra[f"{self.scope}.{COUNTDOWN_KEY}"] = countdown
        return dataclasses.replace(state, extra_state=extra)

    def _observe(self, state):
        return get_all_agent_obs(self._feature_fn, envstate_to_dict(state), self._agent_id_order)

    def _wrap_step_fn(self, base_step_fn):
        agent_ids = self._agent_id_order
        noop = self._noop_idx

        def step_fn(key, state, actions):
            xp = backend.xp
            frame_skip = state.extra_state[f"{self.scope}.{FRAME_SKIP_KEY}"]
            countdown = state.extra_state[f"{self.scope}.{COUNTDOWN_KEY}"]

            # Agents with countdown 0 act this step; others are no-op'd.
            decide = countdown == 0
            masked = {aid: xp.where(decide[i], actions[aid], noop) for i, aid in enumerate(agent_ids)}

            obs, state, rewards, terms, truncs, _ = base_step_fn(key, state, masked)

            # Deciding agents wait k - 1 steps, others tick down.
            countdown = xp.where(decide, frame_skip - 1, countdown - 1)
            state = self._with_frame_skip(state, frame_skip, countdown.astype(np.int32))

            infos = {aid: {"decided": decide[i]} for i, aid in enumerate(agent_ids)}
            return obs, state, rewards, terms, truncs, infos

        return step_fn

    def reset(self, *, seed=42, options=None):
        """Reset; pass ``options={"frame_skip": (k0, k1, ...)}`` to set the skips."""
        obs, infos = super().reset(seed=seed, options=options)
        fixed = (options or {}).get("frame_skip")
        if fixed is not None:
            values = self.frame_skip_values
            if len(fixed) != self.config["num_agents"] or not all(k in values for k in fixed):
                raise ValueError(
                    f"options['frame_skip'] must give one value per agent from {values}, "
                    f"got {fixed}"
                )
            frame_skip = backend.xp.array(fixed, dtype=np.int32)
            self._env_state = self._with_frame_skip(self._env_state, frame_skip)
            obs_dict = self._observe(self._env_state)
            obs = {aid: np.array(obs_dict[aid]) for aid in self._agent_id_order}
        return obs, infos


# ---- Rendering ----


@dataclasses.dataclass
class Sprite:
    """
    Context for a sprite object to render it. We maintain
    this for each object that we are tracking in the game
    state and passing along to render.
    """

    uuid: str
    x: int
    y: int
    height: int
    width: int
    image_name: str | None = None  # texture name
    frame: str | int | None = None
    object_size: int | None = None
    angle: int | None = None
    depth: int = 1
    animation: str | None = None
    object_type: str = "sprite"
    tween: bool = False
    tween_duration: int = 50
    permanent: bool = False

    def as_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Text:
    uuid: str
    text: str
    x: float | int
    y: float | int
    size: int = 16
    color: str = "#000000"
    font: str = "Arial"
    depth: int = -1
    object_type: str = "text"
    permanent: bool = False

    def as_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


TILE_SIZE = 45
# overcooked_cramped_room_v0 is 5 x 4 tiles in CoGrid 0.3.x.
WIDTH = 5 * TILE_SIZE
HEIGHT = 4 * TILE_SIZE
DIR_TO_CARDINAL_DIRECTION = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
}
PLAYER_COLORS = {0: "blue", 1: "green"}


def get_x_y(pos: tuple[int, int], game_height: int, game_width: int) -> tuple[float, float]:
    col, row = pos
    x = row * TILE_SIZE / game_width
    y = col * TILE_SIZE / game_height
    return x, y


def sprite_id(obj: grid_object.GridObj, pos: tuple[int, int], suffix: str = "") -> str:
    """Stable sprite ID. CoGrid 0.3 objects have no uuid and are rebuilt from
    array state, but each (type, cell) pair is unique and stable."""
    return f"{obj.object_id}-{pos[0]}-{pos[1]}{suffix}"


def generate_static_objects(env: CoGridEnv) -> list[Sprite]:
    """Counters, delivery zones, and dispensers; drawn once per episode."""
    objs = []
    for obj in env.grid.grid:
        if obj is None:
            continue
        x, y = get_x_y(obj.pos, HEIGHT, WIDTH)

        if isinstance(
            obj,
            (grid_object.Counter, grid_object.Wall, overcooked_grid_objects.Pot),
        ):
            objs.append(
                Sprite(
                    sprite_id(obj, obj.pos, "-tile"),
                    x=x,
                    y=y,
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    image_name="terrain",
                    frame="counter.png",
                    permanent=True,
                    depth=-2,
                )
            )

        static_frame = None
        if isinstance(obj, overcooked_grid_objects.DeliveryZone):
            static_frame = "serve.png"
        elif isinstance(obj, overcooked_grid_objects.PlateStack):
            static_frame = "dishes.png"
        elif isinstance(obj, overcooked_grid_objects.OnionStack):
            static_frame = "onions.png"
        elif isinstance(obj, overcooked_grid_objects.Pot):
            static_frame = "pot.png"

        if static_frame is not None:
            objs.append(
                Sprite(
                    sprite_id(obj, obj.pos, "-static"),
                    x=x,
                    y=y,
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    image_name="terrain",
                    frame=static_frame,
                    permanent=True,
                )
            )
    return objs


def generate_agent_sprites(env: CoGridEnv) -> list[Sprite]:
    objs = []
    for i, agent_obj in enumerate(env.grid.grid_agents.values()):
        x, y = get_x_y(agent_obj.pos, HEIGHT, WIDTH)
        held_object_name = ""
        if agent_obj.inventory:
            held_obj = agent_obj.inventory[0]
            if isinstance(held_obj, overcooked_grid_objects.Onion):
                held_object_name = "-onion"
            elif isinstance(held_obj, overcooked_grid_objects.OnionSoup):
                held_object_name = "-soup-onion"
            elif isinstance(held_obj, overcooked_grid_objects.Plate):
                held_object_name = "-dish"

        dir = DIR_TO_CARDINAL_DIRECTION[agent_obj.dir]

        objs.append(
            Sprite(
                f"agent-{i}-sprite",
                x=x,
                y=y,
                height=TILE_SIZE,
                width=TILE_SIZE,
                image_name="chefs",
                tween=True,
                tween_duration=75,
                frame=f"{dir}{held_object_name}.png",
            )
        )

        objs.append(
            Sprite(
                f"agent-{i}-hat-sprite",
                x=x,
                y=y,
                height=TILE_SIZE,
                width=TILE_SIZE,
                image_name="chefs",
                frame=f"{dir}-{PLAYER_COLORS[i]}hat.png",
                tween=True,
                tween_duration=75,
                depth=2,
            )
        )
    return objs


def generate_objects(env: CoGridEnv) -> list[Sprite | Text]:
    objs = []
    for obj in env.grid.grid:
        if obj is None:
            continue

        if obj.obj_placed_on is not None:
            # obj_placed_on is rebuilt on each state sync and has no pos of
            # its own, so draw it at its container's position.
            objs += temp_object_creation(obj=obj.obj_placed_on, pos=obj.pos)

        objs += temp_object_creation(obj=obj)

    return objs


def temp_object_creation(obj: grid_object.GridObj, pos=None) -> list[Sprite | Text]:
    pos = pos if pos is not None else obj.pos
    if pos is None:
        return []
    x, y = get_x_y(pos, HEIGHT, WIDTH)

    if isinstance(obj, overcooked_grid_objects.Pot):
        if not obj.objects_in_pot:
            return []
        status = "cooked" if obj.cooking_timer == 0 else "cooking"
        if status == "cooking":
            frame = f"soup-onion-{len(obj.objects_in_pot)}-cooking.png"
        else:
            frame = "soup-onion-cooked.png"

        pot_sprite = [
            Sprite(
                sprite_id(obj, pos, "-contents"),
                x=x,
                y=y,
                height=TILE_SIZE,
                width=TILE_SIZE,
                image_name="objects",
                frame=frame,
                depth=-1,
            )
        ]

        if status == "cooking" and len(obj.objects_in_pot) == 3:
            pot_sprite.append(
                Text(
                    uuid=sprite_id(obj, pos, "-time-left"),
                    text=f"{obj.cooking_timer:02d}",
                    x=x,
                    y=y,
                    size=14,
                    color="red",
                )
            )

        return pot_sprite

    frame = None
    if isinstance(obj, overcooked_grid_objects.Onion):
        frame = "onion.png"
    elif isinstance(obj, overcooked_grid_objects.Plate):
        frame = "dish.png"
    elif isinstance(obj, overcooked_grid_objects.OnionSoup):
        frame = "soup-onion-dish.png"

    if frame is None:
        return []

    return [
        Sprite(
            sprite_id(obj, pos),
            x=x,
            y=y,
            height=TILE_SIZE,
            width=TILE_SIZE,
            image_name="objects",
            frame=frame,
            depth=-1,
        )
    ]


class InteractiveGymRenderMixin:
    """Sprite rendering and position infos for Interactive Gym.

    ``render_mode`` stays None so CoGrid doesn't render inside ``step``; we sync
    the grid objects from the array state ourselves before rendering.
    """

    def reset(self, *, seed=42, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        self._sync_objects_from_state()
        return obs, self._add_position_infos(infos)

    def step(self, actions):
        # The archived human-AI experiment logged unweighted reward components
        # from the pre-step grid, while only delivery contributed to the score.
        # Calculate before stepping: CoGrid 0.3 rebuilds these grid objects.
        component_infos = self._get_component_infos(actions)
        obs, rewards, terminateds, truncateds, infos = super().step(actions)
        self._sync_objects_from_state()
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        for agent_id, components in component_infos.items():
            infos[agent_id].update(components)
        return obs, rewards, terminateds, truncateds, self._add_position_infos(infos)

    def _get_component_infos(self, actions):
        """Preserve the archive's four component names and unit coefficients.

        These use the archived reward predicates (interaction, held item, and
        facing cell), rather than adding shaping rewards to the environment.
        Reset infos intentionally have no components, matching the old logger.
        """
        infos = {
            agent_id: {
                "delivery_reward": 0.0,
                "delivery_act_reward": 0.0,
                "onion_in_pot_reward": 0.0,
                "soup_in_dish_reward": 0.0,
            }
            for agent_id in self._agent_id_order
        }
        for agent_id, action in self._action_idx_to_str(actions).items():
            if action != grid_actions.Actions.PickupDrop:
                continue
            agent = self.grid.grid_agents[agent_id]
            if not agent.inventory:
                continue
            held = agent.inventory[0]
            facing = self.grid.get(*agent.front_pos)
            if isinstance(facing, overcooked_grid_objects.DeliveryZone):
                if isinstance(held, overcooked_grid_objects.OnionSoup):
                    infos[agent_id]["delivery_act_reward"] = 1.0
            elif isinstance(facing, overcooked_grid_objects.Pot):
                if isinstance(held, overcooked_grid_objects.Onion):
                    if len(facing.objects_in_pot) < facing.capacity:
                        infos[agent_id]["onion_in_pot_reward"] = 1.0
                elif isinstance(held, overcooked_grid_objects.Plate):
                    # In 0.3 an empty pot may have timer 0; it is not ready soup.
                    if (
                        len(facing.objects_in_pot) == facing.capacity
                        and facing.cooking_timer == 0
                    ):
                        infos[agent_id]["soup_in_dish_reward"] = 1.0

        common_delivery = sum(
            components["delivery_act_reward"] for components in infos.values()
        )
        for components in infos.values():
            components["delivery_reward"] = common_delivery
        return infos

    def _add_position_infos(self, infos):
        """Add the agent positions and directions to the infos dictionary."""
        for agent_id, agent in self.grid.grid_agents.items():
            row, col = agent.pos
            infos[agent_id]["row"] = int(row)
            infos[agent_id]["col"] = int(col)
            infos[agent_id]["direction"] = int(agent.dir)
            infos[agent_id]["layout_id"] = self.current_layout_id
        return infos

    def render(self):
        render_objects = []

        if self.t == 0:
            render_objects += generate_static_objects(env=self)

        render_objects += generate_agent_sprites(env=self)
        render_objects += generate_objects(env=self)

        return [obj.as_dict() for obj in render_objects]


class InteractiveGymOvercooked(InteractiveGymRenderMixin, CoGridEnv):
    """Solo Overcooked for the tutorial (no policy, so no frame-skip feature)."""


class InteractiveGymFrameSkipOvercooked(InteractiveGymRenderMixin, FrameSkipOvercookedEnv):
    """Human + AI Overcooked whose observations report fixed per-agent frame skips."""

    def __init__(self, config, agent_frame_skips, **kwargs):
        super().__init__(config, frame_skip_values=FRAME_SKIP_VALUES, **kwargs)
        self.agent_frame_skips = tuple(agent_frame_skips)

    def reset(self, *, seed=42, options=None):
        options = {"frame_skip": self.agent_frame_skips, **(options or {})}
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = super().step(actions)

        # Zero the countdowns so FrameSkipOvercookedEnv never masks an action;
        # Interactive Gym's frame_skip sets the cadence instead (see module docstring).
        frame_skip = self._env_state.extra_state[f"{self.scope}.{FRAME_SKIP_KEY}"]
        n_agents = self.config["num_agents"]
        self._env_state = self._with_frame_skip(
            self._env_state, frame_skip, np.zeros(n_agents, dtype=np.int32)
        )
        return obs, rewards, terminateds, truncateds, infos


# ---- Build the env ----

# Start from the config the policy was trained on so the features (and their
# order) match; only the reward (shown as the score) and episode length change.
overcooked_config = copy.deepcopy(registry.ENVIRONMENT_REGISTRY["Overcooked-CrampedRoom-V0"].keywords["config"])
overcooked_config["rewards"] = [DeliveryReward(coefficient=1.0, common_reward=True)]
overcooked_config["max_steps"] = MAX_STEPS
overcooked_config["num_agents"] = NUM_AGENTS

if NUM_AGENTS == 1:
    # Partner-relative features aren't defined for a solo kitchen, and nothing
    # reads the tutorial's observations.
    overcooked_config["features"] = ["agent_dir", "agent_position"]
    env = InteractiveGymOvercooked(config=overcooked_config, agent_class=OvercookedAgent)
else:
    env = InteractiveGymFrameSkipOvercooked(
        config=overcooked_config,
        agent_frame_skips=(HUMAN_FRAME_SKIP, AI_OBSERVED_FRAME_SKIP),
        agent_class=OvercookedAgent,
    )
