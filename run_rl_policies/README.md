# AI-only Cramped Room simulations

This adapts the participant/episode loop and wide CSV outputs from
`DDM-Lab/interactive-gym`, branch `grace_comp_analysis`, to the frame-skip
policy in this repository. Both chefs are AI agents using
`interactive_gym/server/static/assets/overcooked/models/frame_skip_cramped_room_00.onnx`.

## Set up once

Use a separate Python environment: the root `requirements.txt` pins legacy
CoGrid 0.0.15, while this agent uses CoGrid 0.3.2. Python 3.12 was used for
validation. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-ai-only
.\.venv-ai-only\Scripts\python.exe -m pip install -r run_rl_policies/requirements.txt
```

For an environment created with `uv` without pip, install dependencies using:

```powershell
uv pip install --python .venv-ai-only/Scripts/python.exe -r run_rl_policies/requirements.txt
```

In VS Code, use **Python: Select Interpreter** and select
`.venv-ai-only/Scripts/python.exe`. The commands below explicitly select it
and do not require activating the environment.

## Run

```powershell
.\.venv-ai-only\Scripts\python.exe -m run_rl_policies.chapt2_collect_AI_human_speed
```

Defaults: 5 simulated participants, 20 episodes each, 1,350 steps per episode,
base seed 42, headless. Both agents actually act every 5 steps and are told
their frame skip is 1. The default output directory is `<repository>/data/ai_only`.
There is no Flask server, browser, human player, or real-time pacing.

A small check with unequal agent speeds:

```powershell
.\.venv-ai-only\Scripts\python.exe -m run_rl_policies.chapt2_collect_AI_human_speed --participants 1 --episodes 2 --max-steps 100 --actual-frame-skips 2 5 --reported-frame-skips 5 10 --output-dir data/ai_only/check
```

`--actual-frame-skips K0 K1` controls decision cadence independently for each
agent. Positive integers are accepted, with no-ops on intervening steps.
Decisions happen on zero-based steps `0, K, 2K, ...`; actions are not repeated.
`--reported-frame-skips K0 K1` controls each policy's focal frame-skip one-hot;
its values must be 1, 5, 10, or 15. These settings remain independent and are
restored on every reset. Actual cadence is applied only by the runner; the
shared environment adapter disables its own additional action masking.

Other options: `--participants`, `--episodes`, `--max-steps`, `--seed`,
`--output-dir`, `--model-path`, and `--fps`. Paths supplied by the user are
relative to the current directory; default model and output paths are relative
to the repository. `--fps` defaults to 30 and only converts steps to nominal
simulated seconds. It never changes the environment dynamics or sleeps.
Run the module from the repository root, or put the repository on `PYTHONPATH`.

The ONNX adapter validates the feed-forward model's float32 observation input
and named `logits` output against the environment. The shipped model has 896
observation features, 7 action logits, a separate unused value output, and no
recurrent state. One inference session is shared; each agent has a separate
seeded action-sampling generator, recreated each episode. Recurrent legacy
SP/IBC policies require another adapter and are not supported by this runner.

## Output and reproducibility

Each invocation creates a unique UTC-stamped run directory containing:

- One `cramped_room_<participant>_agent0_frame_skip_agent1_frame_skip_game_data.csv`
  per participant, containing all of that participant's episodes.
- `episode_summary.csv`, containing one row per completed episode.
- `metadata.json`, containing configuration, package versions, repository
  commit and working-tree status, environment/model hashes, model tensor
  metadata, action ordering, seed scheme, and completion status.

Participant and episode indices start at zero. CSV `step` starts at one and
describes the completed transition. Seeds are
`base_seed + participant_index * episodes + episode_index`. Separate agent
sampling streams are derived using `numpy.random.SeedSequence(seed).spawn(2)`.
Changing the configured number of episodes changes later participants' seeds.

The per-step CSV retains the original action, shared reward, four reward
components, termination flags, and cumulative reward fields. It adds participant,
seed, per-agent actual/reported skips, and `agent_<id>_decided`. Decision flags
distinguish deliberately selected no-ops from waiting for the next decision.
The legacy `AI_speed` column is filled only when both actual skips agree;
otherwise it is blank. Use the per-agent fields for analysis.

`cumulative_reward` includes the current transition and counts shared delivery
reward once. Per-agent component columns describe individual contributions or
shared delivery as in the human–AI logger. `episode_elapsed_time` is wall-clock
runtime, while `simulated_elapsed_time` is `step / fps`. Wall-clock fields will
differ across repeated seeded runs. CSVs flush after each episode, files close
on errors, and metadata records complete/failed/interrupted status. Abrupt process
termination can leave status `running` and a partial last episode.

The runner loads the exact environment initialization source used by the browser
experiment, caches its definitions, and creates fresh environments from its
configuration. It therefore uses the new CoGrid layout and feature composition,
not repo 1's older behavior/full-map observations. Results are not automatically
comparable to the older environment. Browser inference is asynchronous; this
synchronous runner matches intended cadence, not exact browser action timing.

## Validate

```powershell
.\.venv-ai-only\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use the real model/environment to check inference, independent cadence,
reported one-hots across resets and steps, deterministic trajectories, episode
outputs, and existing component-reward behavior.
