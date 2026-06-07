import argparse

from src.evaluate import evaluate_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint file."
    )

    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )