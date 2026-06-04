import argparse

from src.cross_dataset import run_cross_dataset_adaptation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_cross_dataset_adaptation(args.config)
