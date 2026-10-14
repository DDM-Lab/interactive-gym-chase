"""Repair two identified client-logger resets in the October 14 recordings.

The raw CSVs are copied to data/recording_repair_backups before replacement.
Only the two subject IDs below are touched. Each output row contains one actual
gameplay step; the logger's reset rows and abandoned startup steps stay in the
backups. Run from the repository root with ``python bonus_calculation/repair_subject_recordings.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from calculate_bonuses import read_round


COMBINED_SUBJECT = "66b153eb1d692e08cd3c7503"
RESET_SUBJECT = "615ddab1e4f013092538b6c5"
STEPS = 1350
TIMING_COLUMNS = {"t", "timestamp", "episode_num"}


def round_path(data_dir: Path, number: int, subject: str) -> Path:
    return (
        data_dir / "cramped_room_frame_skip"
        / f"cramped_room_frame_skip_round_{number:02d}" / f"{subject}.csv"
    )


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    return list(reader.fieldnames or []), list(reader), raw


def make_round(
    fields: list[str], rows: list[dict[str, str]],
    time_start: int, gameplay_start: int,
) -> list[dict[str, str]]:
    """Align one 1350-step gameplay array with its own timestamps."""
    times = rows[time_start : time_start + STEPS + 1]
    gameplay = rows[gameplay_start : gameplay_start + STEPS]
    if len(times) != STEPS + 1 or len(gameplay) != STEPS:
        raise ValueError("Expected one complete 1350-step run")
    if [int(row["t"]) for row in times] != list(range(STEPS + 1)):
        raise ValueError("The run has missing or unordered timestep markers")
    if any(
        not row["infos.0.delivery_reward"] or not row["actions.0"]
        or not row["actions.1"]
        for row in gameplay
    ):
        raise ValueError("The run has missing gameplay fields")
    terminals = [
        i for i, row in enumerate(gameplay)
        if row["terminateds.__all__"] == "True"
        or row["truncateds.__all__"] == "True"
    ]
    if terminals != [STEPS - 1]:
        raise ValueError(f"Expected terminal gameplay step 1349, got {terminals}")

    result = []
    for i, source in enumerate(gameplay):
        row = {field: source[field] for field in fields if field not in TIMING_COLUMNS}
        row.update(t=str(i), timestamp=times[i + 1]["timestamp"], episode_num="0")
        result.append(row)
    return result


def prepare(data_dir: Path):
    """Build and validate all outputs before changing any recording."""
    replacements = {}
    originals = {}

    combined = round_path(data_dir, 14, COMBINED_SUBJECT)
    missing = round_path(data_dir, 13, COMBINED_SUBJECT)
    if missing.exists():
        raise FileExistsError(f"Expected round 13 CSV to be absent: {missing}")
    fields, rows, raw = load_csv(combined)
    if len(rows) != 2 * (STEPS + 1):
        raise ValueError(f"Expected two runs in {combined}")
    replacements[missing] = (fields, make_round(fields, rows, 0, 0))
    replacements[combined] = (fields, make_round(fields, rows, STEPS + 1, STEPS))
    originals[combined] = raw

    for number in range(1, 21):
        path = round_path(data_dir, number, RESET_SUBJECT)
        fields, rows, raw = load_csv(path)
        timesteps = [int(row["t"]) for row in rows]
        try:
            second_reset = timesteps.index(0, 1)
        except ValueError as error:
            raise ValueError(f"Expected a second reset in {path}") from error
        if timesteps[:second_reset] != list(range(second_reset)):
            raise ValueError(f"Unexpected startup segment in {path}")
        if len(rows) != second_reset + STEPS + 1:
            raise ValueError(f"Unexpected row count in {path}")
        # The first reset has one more t/timestamp item than gameplay items.
        first_gameplay_steps = second_reset - 1
        if any(
            float(row["infos.0.delivery_reward"]) != 0
            for row in rows[:first_gameplay_steps]
        ):
            raise ValueError(f"Startup segment contains deliveries: {path}")
        replacements[path] = (
            fields, make_round(fields, rows, second_reset, first_gameplay_steps),
        )
        originals[path] = raw

    # Both reward totals must be unchanged by the repair.
    for subject, expected in ((COMBINED_SUBJECT, 19), (RESET_SUBJECT, 153)):
        paths = [path for path in replacements if path.stem == subject]
        reward = sum(
            float(row["infos.0.delivery_reward"])
            for path in paths for row in replacements[path][1]
        )
        if reward != expected:
            raise ValueError(f"Unexpected repaired reward total for {subject}: {reward}")
    return replacements, originals


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.stem}.", suffix=".tmp", delete=False,
    ) as temp:
        writer = csv.DictWriter(temp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(temp.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    data_dir = args.data_dir
    replacements, originals = prepare(data_dir)
    backup_dir = data_dir / "recording_repair_backups"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    manifest = {}
    for source, raw in originals.items():
        destination = backup_dir / source.relative_to(data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        manifest[str(source.relative_to(data_dir))] = hashlib.sha256(raw).hexdigest()
    (backup_dir / "sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    for path, (fields, rows) in replacements.items():
        write_csv(path, fields, rows)
        result = read_round(path)
        if not result.complete:
            raise ValueError(f"Repaired round failed validation: {path}: {result.issue}")
    print(f"Repaired {len(replacements)} round CSVs; backed up {len(originals)} originals")
    print(f"Original files and SHA-256 manifest: {backup_dir.resolve()}")


if __name__ == "__main__":
    main()
