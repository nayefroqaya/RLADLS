from src.train import train_in_domain
from src.evaluate import evaluate_checkpoint
from src.utils import load_config


VALID_EXPERIMENT_CASES = {
    "in_domain",
}

VALID_RUN_MODES = {
    "train",
    "eval_checkpoint",
    "evaluate",
    "eval",
}


def run_experiment_from_config(config_path: str = "config.yaml"):
    """
    YAML-controlled FTADLS runner.

    This version supports only in-domain experiments.

    Use config.yaml to choose whether to train or evaluate:

    run:
      mode: train

    or:

    run:
      mode: eval_checkpoint
      checkpoint_path: outputs/indomain_ftadls_dqn.pt

    experiment:
      case: in_domain
    """

    cfg = load_config(config_path)

    experiment_cfg = cfg.get("experiment", {})

    case = str(
        experiment_cfg.get("case", "in_domain")
    ).strip().lower()

    if case not in VALID_EXPERIMENT_CASES:
        valid_cases = ", ".join(
            sorted(VALID_EXPERIMENT_CASES)
        )

        raise ValueError(
            f"Unknown experiment.case='{case}' in {config_path}. "
            f"This version supports only: {valid_cases}"
        )

    run_cfg = cfg.get("run", {})

    mode = str(
        run_cfg.get("mode", "train")
    ).strip().lower()

    if mode not in VALID_RUN_MODES:
        valid_modes = ", ".join(
            sorted(VALID_RUN_MODES)
        )

        raise ValueError(
            f"Unknown run.mode='{mode}' in {config_path}. "
            f"Valid modes are: {valid_modes}"
        )

    print()
    print("=" * 70)
    print("YAML-CONTROLLED FTADLS RUN")
    print("=" * 70)
    print(f"Experiment case : {case}")
    print(f"Run mode        : {mode}")
    print("=" * 70)

    if mode == "train":
        print()
        print("Running in-domain training + validation + test...")

        return train_in_domain(
            config_path=config_path
        )

    if mode in {"eval_checkpoint", "evaluate", "eval"}:
        checkpoint_path = run_cfg.get(
            "checkpoint_path",
            None
        )

        if checkpoint_path is None:
            output_cfg = cfg.get("output", {})

            output_dir = output_cfg.get(
                "output_dir",
                "outputs"
            )

            checkpoint_name = output_cfg.get(
                "checkpoint_name",
                "indomain_ftadls_dqn.pt"
            )

            checkpoint_path = f"{output_dir}/{checkpoint_name}"

        print()
        print("Running checkpoint evaluation only...")
        print(f"Checkpoint path: {checkpoint_path}")

        return evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path
        )

    raise RuntimeError(
        f"Unhandled run mode: {mode}"
    )