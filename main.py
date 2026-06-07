import argparse

from src.experiment import run_experiment_from_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file."
    )

    args = parser.parse_args()

    run_experiment_from_config(args.config)