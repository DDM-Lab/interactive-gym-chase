"""Run with: python -m run_rl_policies.chapt2_collect_AI_human_speed."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Headless AI-only Cramped Room frame-skip simulations.")
    parser.add_argument("--participants", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per simulated participant")
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--actual-frame-skips", type=int, nargs=2, default=(5, 5), metavar=("AGENT_0", "AGENT_1"))
    parser.add_argument("--reported-frame-skips", type=int, nargs=2, choices=(1, 5, 10, 15), default=(1, 1), metavar=("AGENT_0", "AGENT_1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30, help="Conversion from environment steps to simulated seconds; no wall-clock pacing")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <repository>/data/ai_only")
    parser.add_argument("--model-path", type=Path, help="Defaults to the repository's frame_skip_cramped_room_00.onnx")
    args = parser.parse_args(argv)

    # --help works before the simulation dependencies are installed.
    from run_rl_policies.experiment import ExperimentConfig, run_experiment

    kwargs = vars(args)
    kwargs["actual_frame_skips"] = tuple(args.actual_frame_skips)
    kwargs["reported_frame_skips"] = tuple(args.reported_frame_skips)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        config = ExperimentConfig(**kwargs)
    except ValueError as error:
        parser.error(str(error))
    return run_experiment(config)


if __name__ == "__main__":
    main()
