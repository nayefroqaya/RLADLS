from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy
import random

import pandas as pd
from tqdm import tqdm


@dataclass
class LogSequence:
    sequence_id: str
    sequence: List[int]
    label: int  # true label: 0 = normal, 1 = anomaly

    # Reward label is what the RL environment uses for training reward.
    # For normal target adaptation, reward_label=0 for selected target normal data.
    reward_label: Optional[int] = None
    is_labeled: bool = True
    anomaly_score: float = 0.0
    reward_weight: float = 1.0
    dataset_name: str = ""
    split_name: str = ""


def clone_sequence_with_reward(
    sequence: LogSequence,
    reward_label: Optional[int],
    is_labeled: bool,
    reward_weight: float = 1.0
) -> LogSequence:
    copied = copy.deepcopy(sequence)
    copied.reward_label = reward_label
    copied.is_labeled = is_labeled
    copied.reward_weight = reward_weight
    return copied


def normalize_label(value, normal_label: str, anomaly_label: str) -> int:
    text = str(value).strip().lower()

    normal_values = {
        str(normal_label).lower(),
        "normal",
        "0",
        "false",
        "benign",
        "-"
    }

    anomaly_values = {
        str(anomaly_label).lower(),
        "anomaly",
        "abnormal",
        "1",
        "true"
    }

    if text in normal_values:
        return 0

    if text in anomaly_values:
        return 1

    raise ValueError(
        f"Unknown label value: {value}. "
        "Please check normal_label and anomaly_label in config.yaml."
    )


def dataset_group_col(cfg: dict, dataset_name: str):
    dataset_lower = dataset_name.lower()

    per_dataset = cfg.get("dataset_group_cols", {})
    if dataset_name in per_dataset:
        return per_dataset[dataset_name]

    if dataset_lower == "hdfs":
        return cfg["data"].get("group_col", "Node_block_id")

    return None


def split_paths_for_dataset(cfg: dict, dataset_name: str) -> dict:
    base_dir = Path(cfg["paths"]["data_dir"])
    dataset_subdir = cfg["paths"]["dataset_path_template"].format(
        dataset=dataset_name
    )
    dataset_dir = base_dir / dataset_subdir

    return {
        "train": dataset_dir / cfg["split_files"]["train"],
        "val": dataset_dir / cfg["split_files"]["val"],
        "test": dataset_dir / cfg["split_files"]["test"],
    }


def get_split_paths(cfg: dict) -> dict:
    return split_paths_for_dataset(
        cfg=cfg,
        dataset_name=str(cfg["data"]["dataset_name"])
    )


def load_pickle_dataframe(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")

    print(f"Loading {split_name} dataframe from: {path}")
    df = pd.read_pickle(path)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame in {path}, but got {type(df)}"
        )

    print(f"Loaded {split_name}: shape={df.shape}")
    return df


def load_raw_splits_for_dataset(cfg: dict, dataset_name: str) -> dict:
    paths = split_paths_for_dataset(cfg, dataset_name)

    print()
    print(f"Split file paths for {dataset_name}:")
    print(f"  train: {paths['train']}")
    print(f"  val  : {paths['val']}")
    print(f"  test : {paths['test']}")

    return {
        "train": load_pickle_dataframe(paths["train"], f"{dataset_name}/train"),
        "val": load_pickle_dataframe(paths["val"], f"{dataset_name}/validation"),
        "test": load_pickle_dataframe(paths["test"], f"{dataset_name}/test"),
    }


def build_template_mapping_from_dataframes(
    dataframes: List[pd.DataFrame],
    template_col: str
) -> Dict[str, int]:
    print("Building shared template vocabulary...")

    all_templates = pd.concat(
        [df[template_col] for df in dataframes],
        axis=0
    ).astype(str)

    unique_templates = sorted(
        tqdm(
            all_templates.unique().tolist(),
            desc="Collecting unique templates"
        )
    )
    # get embedding the template ID :
    template_to_id = {
        template: index + 1
        for index, template in enumerate(
            tqdm(unique_templates, desc="Mapping templates to IDs")
        )
    }

    print(f"Template vocabulary size: {len(template_to_id)}")
    return template_to_id


def build_template_mapping_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    template_col: str
) -> Dict[str, int]:
    return build_template_mapping_from_dataframes(
        dataframes=[train_df, val_df, test_df],
        template_col=template_col
    )


def prepare_dataframe(
    df: pd.DataFrame,
    template_to_id: Dict[str, int],
    time_col: str,
    template_col: str,
    label_col: str,
    normal_label: str,
    anomaly_label: str,
    split_name: str
) -> pd.DataFrame:
    print(f"Preparing {split_name} dataframe...")

    required_columns = [time_col, template_col, label_col]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {split_name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[template_col, label_col]).copy()

    print(f"{split_name}: mapping templates to IDs...")
    df["template_id"] = df[template_col].astype(str).progress_map(template_to_id.get)

    if df["template_id"].isna().any():
        missing_templates = df.loc[
            df["template_id"].isna(),
            template_col
        ].astype(str).unique()[:10]

        raise ValueError(
            f"Some templates in {split_name} were not found in template_to_id. "
            f"Examples: {missing_templates}"
        )

    df["template_id"] = df["template_id"].astype(int)

    print(f"{split_name}: normalizing labels...")
    df["label_id"] = df[label_col].progress_apply(
        lambda x: normalize_label(
            value=x,
            normal_label=normal_label,
            anomaly_label=anomaly_label
        )
    )

    print(f"{split_name}: sorting by timestamp...")
    df = df.sort_values(time_col).reset_index(drop=True)

    print(f"Finished preparing {split_name}: shape={df.shape}")
    return df


def dataframe_to_grouped_sequences(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    split_name: str,
    dataset_name: str
) -> List[LogSequence]:
    print(f"Building grouped sequences for {split_name} using {group_col}...")

    if group_col is None or group_col not in df.columns:
        raise ValueError(
            f"Grouped sequence construction requires group_col such as Node_block_id or block_id. "
            f"Current group_col={group_col}. Available columns: {list(df.columns)}"
        )

    df = df.sort_values([group_col, time_col])
    grouped = df.groupby(group_col, sort=False)
    sequences = []

    for group_value, group in tqdm(
        grouped,
        total=grouped.ngroups,
        desc=f"{split_name}: grouping by {group_col}"
    ):
        template_sequence = group["template_id"].astype(int).tolist()
        label = int(group["label_id"].max())

        if not template_sequence:
            continue

        sequences.append(
            LogSequence(
                sequence_id=f"{dataset_name}_{group_value}",
                sequence=template_sequence,
                label=label,
                reward_label=label,
                is_labeled=True,
                reward_weight=1.0,
                dataset_name=dataset_name,
                split_name=split_name
            )
        )

    print(f"{split_name}: built {len(sequences)} grouped sequences")
    return sequences


def dataframe_to_sliding_window_sequences(
    df: pd.DataFrame,
    dataset_name: str,
    sequence_length: int,
    sliding_step: int,
    split_name: str
) -> List[LogSequence]:
    print(
        f"Building sliding-window sequences for {split_name}: "
        f"window={sequence_length}, step={sliding_step}"
    )

    template_ids = df["template_id"].astype(int).tolist()
    labels = df["label_id"].astype(int).tolist()
    sequences = []
    n = len(template_ids)

    if n == 0:
        print(f"{split_name}: no rows found.")
        return sequences

    if n < sequence_length:
        label = int(max(labels))
        sequences.append(
            LogSequence(
                sequence_id=f"{dataset_name}_{split_name}_window_0",
                sequence=template_ids,
                label=label,
                reward_label=label,
                is_labeled=True,
                reward_weight=1.0,
                dataset_name=dataset_name,
                split_name=split_name
            )
        )
        print(f"{split_name}: dataset shorter than sequence length. Built 1 sequence.")
        return sequences

    total_windows = ((n - sequence_length) // sliding_step) + 1
    window_index = 0

    for start in tqdm(
        range(0, n - sequence_length + 1, sliding_step),
        total=total_windows,
        desc=f"{split_name}: creating windows"
    ):
        end = start + sequence_length
        window_templates = template_ids[start:end]
        window_labels = labels[start:end]
        label = int(max(window_labels))

        sequences.append(
            LogSequence(
                sequence_id=f"{dataset_name}_{split_name}_window_{window_index}",
                sequence=window_templates,
                label=label,
                reward_label=label,
                is_labeled=True,
                reward_weight=1.0,
                dataset_name=dataset_name,
                split_name=split_name
            )
        )

        window_index += 1

    print(f"{split_name}: built {len(sequences)} sliding-window sequences")
    return sequences


def convert_raw_splits_to_sequences(
    cfg: dict,
    dataset_name: str,
    raw_splits: dict,
    template_to_id: Dict[str, int]
) -> Tuple[List[LogSequence], List[LogSequence], List[LogSequence]]:
    time_col = cfg["data"]["time_col"]
    template_col = cfg["data"]["template_col"]
    label_col = cfg["data"]["label_col"]
    normal_label = cfg["data"]["normal_label"]
    anomaly_label = cfg["data"]["anomaly_label"]
    sequence_length = int(cfg["sequence"]["sequence_length"])
    sliding_step = int(cfg["sequence"]["sliding_step"])
    group_col = dataset_group_col(cfg, dataset_name)

    prepared = {}
    for split_name, df in raw_splits.items():
        prepared[split_name] = prepare_dataframe(
            df=df,
            template_to_id=template_to_id,
            time_col=time_col,
            template_col=template_col,
            label_col=label_col,
            normal_label=normal_label,
            anomaly_label=anomaly_label,
            split_name=f"{dataset_name}/{split_name}"
        )

    construction_mode = str(
        cfg.get("sequence", {}).get("construction", "group_by_column")
    ).strip().lower()

    group_available = (
        group_col is not None
        and all(group_col in prepared[name].columns for name in ["train", "val", "test"])
    )

    if construction_mode in {"group", "group_by_column", "node_block_id"}:
        if not group_available:
            raise ValueError(
                f"sequence.construction is '{construction_mode}', but group_col='{group_col}' "
                f"was not found in all splits for dataset {dataset_name}. "
                f"Train columns: {list(prepared['train'].columns)}"
            )

        train_sequences = dataframe_to_grouped_sequences(
            prepared["train"], group_col, time_col, "train", dataset_name
        )
        val_sequences = dataframe_to_grouped_sequences(
            prepared["val"], group_col, time_col, "validation", dataset_name
        )
        test_sequences = dataframe_to_grouped_sequences(
            prepared["test"], group_col, time_col, "test", dataset_name
        )

    elif construction_mode in {"sliding", "sliding_window"}:
        train_sequences = dataframe_to_sliding_window_sequences(
            prepared["train"], dataset_name, sequence_length, sliding_step, "train"
        )
        val_sequences = dataframe_to_sliding_window_sequences(
            prepared["val"], dataset_name, sequence_length, sliding_step, "validation"
        )
        test_sequences = dataframe_to_sliding_window_sequences(
            prepared["test"], dataset_name, sequence_length, sliding_step, "test"
        )

    elif construction_mode == "auto":
        if group_available:
            train_sequences = dataframe_to_grouped_sequences(
                prepared["train"], group_col, time_col, "train", dataset_name
            )
            val_sequences = dataframe_to_grouped_sequences(
                prepared["val"], group_col, time_col, "validation", dataset_name
            )
            test_sequences = dataframe_to_grouped_sequences(
                prepared["test"], group_col, time_col, "test", dataset_name
            )
        else:
            train_sequences = dataframe_to_sliding_window_sequences(
                prepared["train"], dataset_name, sequence_length, sliding_step, "train"
            )
            val_sequences = dataframe_to_sliding_window_sequences(
                prepared["val"], dataset_name, sequence_length, sliding_step, "validation"
            )
            test_sequences = dataframe_to_sliding_window_sequences(
                prepared["test"], dataset_name, sequence_length, sliding_step, "test"
            )

    else:
        raise ValueError(
            f"Unknown sequence.construction='{construction_mode}'. "
            "Use group_by_column, sliding_window, or auto."
        )

    return train_sequences, val_sequences, test_sequences


def load_split_sequences_from_config(
    cfg: dict
) -> Tuple[List[LogSequence], List[LogSequence], List[LogSequence], Dict[str, int]]:
    pd.options.mode.chained_assignment = None
    tqdm.pandas()

    dataset_name = str(cfg["data"]["dataset_name"])
    raw = load_raw_splits_for_dataset(cfg, dataset_name)
    template_to_id = build_template_mapping_from_splits(
        train_df=raw["train"],
        val_df=raw["val"],
        test_df=raw["test"],
        template_col=cfg["data"]["template_col"]
    )

    train_sequences, val_sequences, test_sequences = convert_raw_splits_to_sequences(
        cfg=cfg,
        dataset_name=dataset_name,
        raw_splits=raw,
        template_to_id=template_to_id
    )

    return train_sequences, val_sequences, test_sequences, template_to_id


def load_cross_dataset_sequences_from_config(cfg: dict) -> dict:
    """
    Build one shared template vocabulary for source and target datasets.

    Source datasets train the initial policy.
    Target test evaluates direct transfer and adapted policy.
    Target adaptation uses only a normal fraction from target train.
    """
    pd.options.mode.chained_assignment = None
    tqdm.pandas()

    adaptation_cfg = cfg["adaptation"]
    source_datasets = list(adaptation_cfg["source_datasets"])
    target_dataset = str(adaptation_cfg["target_dataset"])
    all_datasets = source_datasets + [target_dataset]

    raw_by_dataset = {
        dataset: load_raw_splits_for_dataset(cfg, dataset)
        for dataset in all_datasets
    }

    all_dfs = []
    for dataset in all_datasets:
        all_dfs.extend(
            [
                raw_by_dataset[dataset]["train"],
                raw_by_dataset[dataset]["val"],
                raw_by_dataset[dataset]["test"]
            ]
        )

    template_to_id = build_template_mapping_from_dataframes(
        dataframes=all_dfs,
        template_col=cfg["data"]["template_col"]
    )

    source_train = []
    source_val = []
    source_test = []

    for dataset in source_datasets:
        train_s, val_s, test_s = convert_raw_splits_to_sequences(
            cfg=cfg,
            dataset_name=dataset,
            raw_splits=raw_by_dataset[dataset],
            template_to_id=template_to_id
        )
        source_train.extend(train_s)
        source_val.extend(val_s)
        source_test.extend(test_s)

    target_train, target_val, target_test = convert_raw_splits_to_sequences(
        cfg=cfg,
        dataset_name=target_dataset,
        raw_splits=raw_by_dataset[target_dataset],
        template_to_id=template_to_id
    )

    return {
        "source_datasets": source_datasets,
        "target_dataset": target_dataset,
        "source_train": source_train,
        "source_val": source_val,
        "source_test": source_test,
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "template_to_id": template_to_id,
    }


def select_target_normal_adaptation_sequences(
    sequences: List[LogSequence],
    fraction: float,
    seed: int
) -> List[LogSequence]:
    """
    Select only normal target training sequences for adaptation.
    This matches the requested setting: 20% normal fraction of target train data.
    """
    normal_sequences = [seq for seq in sequences if seq.label == 0]

    if not normal_sequences:
        raise ValueError("No normal target training sequences found for adaptation.")

    rng = random.Random(seed)
    rng.shuffle(normal_sequences)

    keep_count = max(1, int(round(len(normal_sequences) * fraction)))
    selected = normal_sequences[:keep_count]

    adapted = [
        clone_sequence_with_reward(
            sequence=seq,
            reward_label=0,
            is_labeled=True,
            reward_weight=1.0
        )
        for seq in selected
    ]

    print()
    print("Target adaptation data:")
    print(f"  available target normal train sequences: {len(normal_sequences)}")
    print(f"  selected fraction: {fraction:.2f}")
    print(f"  selected normal sequences: {len(adapted)}")

    return adapted


def describe_sequences(sequences: List[LogSequence], name: str) -> None:
    total = len(sequences)
    anomalies = sum(seq.label for seq in sequences)
    normals = total - anomalies
    avg_len = sum(len(seq.sequence) for seq in sequences) / max(1, total)

    print()
    print(f"{name} sequences:")
    print(f"  total     : {total}")
    print(f"  normal    : {normals}")
    print(f"  anomaly   : {anomalies}")
    print(f"  avg length: {avg_len:.2f}")
