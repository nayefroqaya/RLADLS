from src.train import train_in_domain
from src.cross_dataset import run_cross_dataset_adaptation
from src.utils import load_config


VALID_CASES = {
    "in_domain",
    "cross_dataset_adaptation",
    "cross_dataset",
    "both",
}


def run_experiment_from_config(config_path: str = "config.yaml"):
    """
    Run the experiment selected in config.yaml.

    Supported values:
      experiment.case: in_domain
      experiment.case: cross_dataset_adaptation
      experiment.case: cross_dataset
      experiment.case: both

    Notes:
      - cross_dataset and cross_dataset_adaptation are aliases here.
      - Cross-dataset mode trains on source datasets, evaluates direct transfer,
        adapts using the configured target normal fraction, and evaluates again.
    """
    cfg = load_config(config_path)
    experiment_cfg = cfg.get("experiment", {})
    case = str(experiment_cfg.get("case", "in_domain")).strip().lower()

    if case not in VALID_CASES:
        valid = ", ".join(sorted(VALID_CASES))
        raise ValueError(
            f"Unknown experiment.case='{case}' in {config_path}. "
            f"Valid cases are: {valid}"
        )

    print()
    print("=" * 70)
    print(f"Selected experiment case from YAML: {case}")
    print("=" * 70)

    if case == "in_domain":
        return train_in_domain(config_path)

    if case in {"cross_dataset", "cross_dataset_adaptation"}:
        return run_cross_dataset_adaptation(config_path)

    if case == "both":
        print("\nRunning case 1/2: in-domain experiment")
        in_domain_result = train_in_domain(config_path)

        print("\nRunning case 2/2: cross-dataset adaptation experiment")
        cross_dataset_result = run_cross_dataset_adaptation(config_path)

        return {
            "in_domain": in_domain_result,
            "cross_dataset_adaptation": cross_dataset_result,
        }
