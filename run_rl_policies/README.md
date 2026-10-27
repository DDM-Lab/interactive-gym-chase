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

CLI defaults: 100 simulated participants, 20 episodes each, 1,350 steps per episode,
base seed 42, headless. Both agents actually act every 5 steps and are told
their frame skip is 1. The CLI saves to this local Google Drive folder by default:

```text
G:\.shortcut-targets-by-id\1zFPkx_p8sPRshZUcZ95mHkYUPR3dh1-i\2025GraceRoessling\2025ControllableCollaborationChaseGrace\Experiments\2026-ControllableCollaboration-Temporal-Allignment\Data\Temporally Aligned Agent, AI-only condition
```

Override it with `--output-dir` to save elsewhere. Direct Python use of
`ExperimentConfig` still defaults to `<repository>/data/ai_only`.
On Windows, output I/O automatically uses extended-length paths so the long
Google Drive shortcut path works with run subfolders and participant filenames.
The terminal displays the ordinary path; Python's returned run path may start
with `\\?\` (or `\\?\UNC\` for a network share).
There is no Flask server, browser, human player, or real-time pacing.

To run participants concurrently, add `--workers`:

```powershell
.\.venv-ai-only\Scripts\python.exe -m run_rl_policies.chapt2_collect_AI_human_speed --participants 100 --episodes 20 --workers 4
```

`--workers` is a positive integer and defaults to 16. Use `--workers 1` for
sequential execution. Direct Python use of `ExperimentConfig` defaults to 1.
The effective worker count is capped at the number of participants. Each process
loads its own model and environment once, then processes whole participant
sessions. Episodes within a participant remain sequential. Start with 2 or 4
workers and compare elapsed time for your workload; spawning processes adds
overhead, so very small runs may be faster with one worker.

Seeds and non-timing CSV values are unchanged by worker count. Each participant
has its own gameplay CSV; only the parent process writes metadata and the shared
episode summary. Progress can appear out of participant order, but final summary
rows are sorted by participant and episode. Metadata records the requested and
effective worker counts. On an error, pending tasks are cancelled and running
tasks stop at an episode boundary before the run is marked failed. A keyboard
interrupt marks it interrupted; already written data is retained.

When calling `run_experiment` from another Python script with multiple workers,
put that call inside `if __name__ == "__main__":` so spawned processes can import
the script safely. The provided module entry point already has this guard.

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

Other options: `--participants`, `--episodes`, `--max-steps`, `--workers`, `--seed`,
`--output-dir`, `--model-path`, and `--fps`. Paths supplied by the user are
relative to the current directory. The default model path is relative to the
repository, and the CLI's default output path is the absolute Google Drive path
above. `--fps` defaults to 30 and only converts steps to nominal
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
They also compare sequential and parallel trajectories, verify worker reuse and
worker-count limits, and exercise an actual spawned-worker failure.
