import argparse
import warnings
import colorama
from src.experiment import run_experiment_from_config
from load_datalog import LogdataRead
from utility import Utilities


if __name__ == "__main__":

    # Part A :

    # ---------------- Section A  : run config for data spliting (60%-10%-30%):----------------

    warnings.filterwarnings('ignore')
    colorama.init()

    GREEN = colorama.Fore.GREEN
    GRAY = colorama.Fore.LIGHTBLACK_EX
    RESET = colorama.Fore.RESET
    YELLOW = colorama.Fore.YELLOW

    DATASET = 'BGL'
    DATASETS_FOLDER = 'datasets'
    Round = '1'  # 1 dataset for first run. 2 for second run. 3 for third run.
    Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    # path to save files
    # ALL_DATASET_CSV_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data second paper
    ALL_DATASET_CSV_PATH = f'../../LogSLM/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'  # data second paper

    # Object classes :
    logdata_read_obj = LogdataRead()
    utilities_obj = Utilities()

    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    print(f"{GREEN}Reading the file was done successfully{RESET}")

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, Round,
                                                                                  Mix_or_stable)
    exit()









    # Part B :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file."
    )

    args = parser.parse_args()

    run_experiment_from_config(args.config)