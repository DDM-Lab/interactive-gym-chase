# Human-AI bonus calculation

This standalone script calculates payments for
`interactive_gym/examples/cogrid/overcooked_human_ai_sp_crampedroom.py`.
It uses only Python's standard library; no experiment server or pandas is needed.

From the repository root, run:

```powershell
python bonus_calculation/calculate_bonuses.py --data-dir data
```

The result is `bonus_calculation/subject_payments.csv`, next to this script,
regardless of the input data directory or working directory. To choose another
location:

```powershell
python bonus_calculation/calculate_bonuses.py --data-dir "C:\path\to\experiment\data" --output "C:\path\to\subject_payments.csv"
```

Each run recalculates and replaces the output CSV. Source recordings are read only.
Use a separate data directory for each experiment run; this script groups files
by participant ID and cannot distinguish reused IDs across separate sessions.

## Payment rules

- This is for **human-AI teams only**. Each CSV filename identifies the human;
  no payment row is created for the AI.
- A completed session requires all **20 main rounds** to be fully recorded.
  Completion-scene metadata is reported but is **not required**.
- Completed session: **$4.00 base payment + $0.02 per team delivery reward**.
  There is no minimum reward threshold. Zero deliveries still earns the $4 base.
- Incomplete session: **$0 base, $0 bonus, $0 total**, regardless of deliveries.
- Sum `infos.0.delivery_reward` once across main-round CSVs. The delivery reward
  is shared by the human and AI; adding `infos.1.delivery_reward` would double it.
  Individual action rewards, shaping rewards, and tutorial deliveries are excluded.

The human-AI branch of archived `calculate_bonuses.py` at commit `04cc9d73`
provided the completion-based bonus formula. This script replaces its aggregated
category-folder input and observed maximum-episode check with the current raw
round-folder layout and a fixed 20-round requirement. It explicitly adds the $4
base payment. It does not implement the archive's human-only team or quitter rules.

## Input and completion checks

```text
data/
  cramped_room_frame_skip/
    cramped_room_frame_skip_round_01/<subject_id>.csv
    ...
    cramped_room_frame_skip_round_20/<subject_id>.csv
  end_completion_code_scene/<subject_id>_metadata.json  (optional)
```

Round numbers come from folder names, **not `episode_num`**, which resets inside
each scene. A full round has ordered, unique gameplay timesteps `0..1349`,
nonnegative integer delivery rewards for every timestep, and a terminal flag
(`terminateds.__all__` or `truncateds.__all__`) on timestep 1349. The logger's
optional trailing reset row at timestep 1350 has no delivery reward and is ignored
for scoring. If the experiment's round length changes, use `--steps-per-round`.

Missing or partial rounds make the session incomplete. Gaps, duplicates, extra
gameplay, and missing terminal flags appear in `recording_issues`. Missing required
columns or malformed reward/timestep values stop the script before replacing the
output, rather than producing a payment from unreadable data.

The report includes participants discovered in start, tutorial, survey, and
completion files, even if they recorded no main rounds. It contains:

- `subject_id`, `completion_status`, `rounds_recorded`, `rounds_completed`;
- `missing_rounds`, `incomplete_rounds` (semicolon-separated round numbers);
- `completion_metadata_present`, `recording_issues`;
- `total_delivery_reward` (including recorded rewards for incomplete sessions);
- `base_payment`, `bonus`, `total_payment` (USD, two decimal places).

## Verification

```powershell
python -m unittest discover -s bonus_calculation -p "test_*.py"
```
