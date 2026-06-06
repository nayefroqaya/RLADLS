from src.train import train_in_domain
from src.utils import load_config


VALID_CASES = {
    "in_domain",
}


def run_experiment_from_config(config_path: str = "config.yaml"):
    """
    Run only the in-domain experiment selected in config.yaml.

    Supported value:
      experiment.case: in_domain

    Cross-dataset experiments are intentionally disabled in this version.
    """
    cfg = load_config(config_path)
    experiment_cfg = cfg.get("experiment", {})
    case = str(experiment_cfg.get("case", "in_domain")).strip().lower()

    if case not in VALID_CASES:
        raise ValueError(
            f"Unknown experiment.case='{case}' in {config_path}. "
            "This version supports only: in_domain"
        )

    print()
    print("=" * 70)
    print("Selected experiment case from YAML: in_domain")
    print("Cross-dataset mode is disabled in this version.")
    print("=" * 70)

    return train_in_domain(config_path)
