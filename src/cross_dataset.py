from pathlib import Path
import copy

import torch

from src.data import (
    load_cross_dataset_sequences_from_config,
    select_target_normal_adaptation_sequences,
    describe_sequences,
)
from src.evaluate import evaluate_policy
from src.train import (
    build_model_and_embeddings,
    train_dqn_on_sequences,
    save_checkpoint,
)
from src.utils import load_config, set_seed


def run_cross_dataset_adaptation(config_path: str = "config.yaml"):
    """
    Cross-dataset experiment with target adaptation.

    Workflow:
      1. Train RL agent on source datasets.
      2. Test directly on target test set: cross-dataset direct transfer.
      3. Adapt/fine-tune on only 20% normal target training sequences.
      4. Test adapted model on target test set.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaded = load_cross_dataset_sequences_from_config(cfg)

    source_train = loaded["source_train"]
    source_val = loaded["source_val"]
    target_train = loaded["target_train"]
    target_val = loaded["target_val"]
    target_test = loaded["target_test"]
    template_to_id = loaded["template_to_id"]
    source_datasets = loaded["source_datasets"]
    target_dataset = loaded["target_dataset"]

    print()
    print("Cross-dataset setting:")
    print(f"  source datasets: {source_datasets}")
    print(f"  target dataset : {target_dataset}")
    print("  adaptation     : 20% normal target training sequences only")

    describe_sequences(source_train, "Source Train")
    describe_sequences(source_val, "Source Validation")
    describe_sequences(target_train, "Target Train")
    describe_sequences(target_val, "Target Validation")
    describe_sequences(target_test, "Target Test")

    model, slm_embedding_matrix, vocab_size = build_model_and_embeddings(
        cfg=cfg,
        template_to_id=template_to_id,
        device=device
    )

    train_dqn_on_sequences(
        model=model,
        sequences=source_train,
        cfg=cfg,
        device=device,
        phase_name="Source training"
    )

    print("\nSource validation result:")
    evaluate_policy(
        model=model,
        sequences=source_val,
        cfg=cfg,
        device=device,
        name="Source Validation"
    )

    print("\nCross-dataset direct transfer result before target adaptation:")
    evaluate_policy(
        model=model,
        sequences=target_test,
        cfg=cfg,
        device=device,
        name=f"Direct Transfer to {target_dataset}"
    )

    output_dir = Path(cfg["output"]["output_dir"])
    source_checkpoint = output_dir / cfg["adaptation"]["source_checkpoint_name"]

    save_checkpoint(
        path=source_checkpoint,
        model=model,
        template_to_id=template_to_id,
        vocab_size=vocab_size,
        slm_embedding_matrix=slm_embedding_matrix,
        cfg=cfg,
        extra={
            "experiment_type": "cross_dataset_direct_transfer",
            "source_datasets": source_datasets,
            "target_dataset": target_dataset,
        }
    )

    target_normal_sequences = select_target_normal_adaptation_sequences(
        sequences=target_train,
        fraction=float(cfg["adaptation"]["target_normal_fraction"]),
        seed=int(cfg["data"]["random_seed"])
    )

    adapted_model = copy.deepcopy(model)

    train_dqn_on_sequences(
        model=adapted_model,
        sequences=target_normal_sequences,
        cfg=cfg,
        device=device,
        phase_name=f"Target adaptation on 20% normal {target_dataset}",
        train_steps=int(cfg["adaptation"]["adapt_steps"]),
        learning_rate=float(cfg["adaptation"].get("adapt_learning_rate", cfg["model"]["learning_rate"])),
        epsilon_start=float(cfg["adaptation"].get("epsilon_start", 0.20)),
        epsilon_end=float(cfg["adaptation"].get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(cfg["adaptation"].get("epsilon_decay_steps", 3000)),
    )

    print("\nTarget validation after adaptation:")
    evaluate_policy(
        model=adapted_model,
        sequences=target_val,
        cfg=cfg,
        device=device,
        name=f"Adapted {target_dataset} Validation"
    )

    print("\nCross-dataset target result after 20% normal target adaptation:")
    evaluate_policy(
        model=adapted_model,
        sequences=target_test,
        cfg=cfg,
        device=device,
        name=f"Adapted Transfer to {target_dataset}"
    )

    adapted_checkpoint = output_dir / cfg["adaptation"]["adapted_checkpoint_name"]

    save_checkpoint(
        path=adapted_checkpoint,
        model=adapted_model,
        template_to_id=template_to_id,
        vocab_size=vocab_size,
        slm_embedding_matrix=slm_embedding_matrix,
        cfg=cfg,
        extra={
            "experiment_type": "cross_dataset_target_adaptation",
            "source_datasets": source_datasets,
            "target_dataset": target_dataset,
            "target_normal_fraction": cfg["adaptation"]["target_normal_fraction"],
        }
    )

    print()
    print("Finished cross-dataset experiment.")
    print(f"Source checkpoint : {source_checkpoint}")
    print(f"Adapted checkpoint: {adapted_checkpoint}")
