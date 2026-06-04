import pandas as pd
from pathlib import Path

datasets = ["HDFS", "BGL", "TH_1G", "SP_150MB"]

base_dir = Path("datasets")

for dataset in datasets:
    print("\n" + "=" * 60)
    print(dataset)
    print("=" * 60)

    split_dir = base_dir / dataset / f"1_{dataset}_Splitted_Datasets"

    for split in ["train", "val", "test"]:
        path = split_dir / f"{split}_df.pkl"

        if not path.exists():
            print(f"{split}: file not found -> {path}")
            continue

        df = pd.read_pickle(path)

        print(f"\n{split}: shape={df.shape}")
        print("Columns:")
        print(list(df.columns))

        possible_label_cols = [
            "label", "Label", "labels", "Labels",
            "is_anomaly", "Anomaly", "label_id"
        ]

        found_cols = [col for col in possible_label_cols if col in df.columns]

        if not found_cols:
            print("No common label column found.")
            continue

        for col in found_cols:
            print(f"\nLabel column: {col}")
            print(df[col].value_counts(dropna=False).head(20))