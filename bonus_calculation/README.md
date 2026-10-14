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
completion files, even if they recorded no main rounds. Its columns are:

```text
subject_id,completion_status,rounds_recorded,rounds_completed,completion_metadata_present,total_delivery_reward,base_payment,bonus,total_payment,session_start_datetime,session_completion_datetime,completion_time,recording_issues
```

`total_delivery_reward` includes recorded rewards for incomplete sessions.
Payment amounts are USD with two decimal places. `completion_time` is total
elapsed time between the recorded start and completion timestamps, in
`HH:MM:SS` format (including tutorials, breaks, and survey time). It is blank if
either timestamp is missing. If completion precedes the recorded start, the
duration is left blank and the discrepancy is flagged in `recording_issues`.
This does not stop the report or change payment eligibility: payments depend on
the round recordings. Durations longer than a day retain total hours.

A participant revisiting their URL can overwrite their start-scene metadata
under the same subject ID after completing the experiment. The script preserves
these recorded timestamps rather than inventing an earlier session start.

`session_start_datetime` comes from the `timestamp` in
`overcooked_start_scene/<subject_id>_metadata.json` (normally
`YYYY-MM-DD HH:MM:SS`). Both timestamp columns display 12-hour time with AM/PM
(for example, `2026-10-13 08:34:38 PM`). Sorting uses the original timestamps
before display formatting. The report preserves the server's recorded local time; the
metadata does not include a timezone. If the start metadata or its timestamp is
missing, the field is blank and payment eligibility is unaffected.

Rows are ordered by `session_completion_datetime`, newest completion first.
This timestamp comes from `end_completion_code_scene/<subject_id>_metadata.json`.
Completed sessions without this timestamp follow dated completions. Incomplete
sessions follow all completed sessions, ordered by newest start time. Completion
metadata remains optional for payment eligibility.

## Verification

```powershell
python -m unittest discover -s bonus_calculation -p "test_*.py"
```
