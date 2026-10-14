"""Calculate completion-based payments for the 20-round human-AI experiment."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

EXPECTED_ROUNDS = 20
STEPS_PER_ROUND = 1350
BASE_PAYMENT = Decimal("4.00")
BONUS_PER_DELIVERY = Decimal("0.02")
REWARD_COLUMN = "infos.0.delivery_reward"
ROUND_PATTERN = re.compile(r"cramped_room_frame_skip_round_(\d{2})")
FIELDNAMES = [
    "subject_id", "completion_status", "rounds_recorded", "rounds_completed",
    "missing_rounds", "incomplete_rounds", "completion_metadata_present",
    "total_delivery_reward", "base_payment", "bonus", "total_payment",
    "recording_issues",
]


@dataclass
class RoundResult:
    delivery_reward: Decimal
    complete: bool
    issue: str


def read_round(path: Path, steps_per_round: int = STEPS_PER_ROUND) -> RoundResult:
    """Validate a full round, including the logger's optional trailing reset row."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        required = {"t", REWARD_COLUMN, "terminateds.__all__", "truncateds.__all__"}
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing columns {sorted(required - columns)}")
        rows = list(reader)

    total = Decimal("0")
    timesteps = []
    gameplay_steps = []
    terminal_steps = []
    for line, row in enumerate(rows, start=2):
        try:
            timestep = int(row["t"])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}:{line}: invalid timestep") from error
        timesteps.append(timestep)
        raw_reward = row[REWARD_COLUMN]
        if raw_reward is None:
            raise ValueError(f"{path}:{line}: malformed CSV row")
        if raw_reward.strip():
            try:
                reward = Decimal(raw_reward)
            except InvalidOperation as error:
                raise ValueError(f"{path}:{line}: invalid delivery reward") from error
            if not reward.is_finite() or reward < 0 or reward != reward.to_integral_value():
                raise ValueError(f"{path}:{line}: delivery reward must be a nonnegative integer")
            total += reward
            gameplay_steps.append(timestep)
            if any(
                (row.get(column) or "").strip().lower() == "true"
                for column in ("terminateds.__all__", "truncateds.__all__")
            ):
                terminal_steps.append(timestep)

    # In this logger, gameplay occupies t=0..1349. A blank reset row at
    # t=1350 may follow. episode_num resets per scene and is not a round ID.
    expected = list(range(steps_per_round))
    issues = []
    if timesteps not in (expected, expected + [steps_per_round]):
        issues.append("missing, duplicate, out-of-order, or unexpected timesteps")
    if gameplay_steps != expected:
        issues.append("incomplete gameplay reward samples")
    if terminal_steps != [steps_per_round - 1]:
        issues.append("missing or unexpected terminal gameplay row")
    return RoundResult(total, not issues, "; ".join(issues))


def collect_subjects(data_dir: Path) -> set[str]:
    """Include participants who only reached the start scene or tutorial."""
    subjects = set()
    for folder in (
        "overcooked_start_scene", "overcooked_tutorial",
        "overcooked_frame_skip_tutorial", "cramped_room_options_scene_0",
        "end_completion_code_scene",
    ):
        directory = data_dir / folder
        for path in directory.glob("*.csv"):
            subjects.add(path.stem)
        for suffix in ("_metadata.json", "_globals.json"):
            for path in directory.glob(f"*{suffix}"):
                subjects.add(path.name.removesuffix(suffix))
    return subjects


def calculate_payments(
    data_dir: Path, steps_per_round: int = STEPS_PER_ROUND,
) -> list[dict]:
    """Count shared team deliveries once and pay only the human participant."""
    if steps_per_round < 1:
        raise ValueError("steps_per_round must be positive")
    gameplay_dir = data_dir / "cramped_room_frame_skip"
    if not gameplay_dir.is_dir():
        raise ValueError(f"Gameplay directory not found: {gameplay_dir}")
    subjects = collect_subjects(data_dir)
    recordings: dict[str, dict[int, RoundResult]] = {}
    for folder in sorted(gameplay_dir.iterdir()):
        if not folder.is_dir():
            continue
        match = ROUND_PATTERN.fullmatch(folder.name)
        if not match:
            continue
        round_num = int(match[1])
        if not 1 <= round_num <= EXPECTED_ROUNDS:
            raise ValueError(f"Unexpected round folder: {folder}")
        for path in sorted(folder.glob("*.csv")):
            subjects.add(path.stem)
            recordings.setdefault(path.stem, {})[round_num] = read_round(
                path, steps_per_round,
            )

    if not subjects:
        raise ValueError(f"No participants found in {data_dir}")
    expected = set(range(1, EXPECTED_ROUNDS + 1))
    payments = []
    for subject_id in sorted(subjects):
        rounds = recordings.get(subject_id, {})
        completed = {number for number, result in rounds.items() if result.complete}
        is_complete = completed == expected
        total_reward = sum((result.delivery_reward for result in rounds.values()), Decimal("0"))
        base = BASE_PAYMENT if is_complete else Decimal("0")
        bonus = total_reward * BONUS_PER_DELIVERY if is_complete else Decimal("0")
        payments.append({
            "subject_id": subject_id,
            "completion_status": "completed" if is_complete else "incomplete",
            "rounds_recorded": len(rounds),
            "rounds_completed": len(completed),
            "missing_rounds": ";".join(map(str, sorted(expected - rounds.keys()))),
            "incomplete_rounds": ";".join(map(str, sorted(rounds.keys() - completed))),
            "completion_metadata_present": (
                data_dir / "end_completion_code_scene" / f"{subject_id}_metadata.json"
            ).is_file(),
            "total_delivery_reward": format(total_reward, ".0f"),
            "base_payment": format(base, ".2f"),
            "bonus": format(bonus, ".2f"),
            "total_payment": format(base + bonus, ".2f"),
            "recording_issues": " | ".join(
                f"round {number}: {result.issue}"
                for number, result in sorted(rounds.items()) if result.issue
            ),
        })
    return payments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path,
        help="Default: subject_payments.csv in the script's bonus_calculation folder",
    )
    parser.add_argument(
        "--steps-per-round", type=int, default=STEPS_PER_ROUND,
        help="Expected gameplay timesteps per round (default: 1350)",
    )
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent / "subject_payments.csv"
    if output.resolve().is_relative_to((args.data_dir / "cramped_room_frame_skip").resolve()):
        parser.error("Output must be outside the gameplay directory")
    try:
        payments = calculate_payments(args.data_dir, args.steps_per_round)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(payments)
    complete_count = sum(row["completion_status"] == "completed" for row in payments)
    total_payment = sum((Decimal(row["total_payment"]) for row in payments), Decimal("0"))
    print(f"Completed: {complete_count}/{len(payments)} participants")
    print(f"Total payment: ${total_payment:.2f}")
    print(f"CSV: {output.resolve()}")
    issue_count = sum(bool(row["recording_issues"]) for row in payments)
    if issue_count:
        print(f"Review recording_issues for {issue_count} participant(s).")


if __name__ == "__main__":
    main()
