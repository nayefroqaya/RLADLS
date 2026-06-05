from pathlib import Path
import random
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data import load_split_sequences_from_config, describe_sequences
from src.slm_embedder import build_minilm_embedding_matrix
from src.env import LogAnomalyEnv
from src.model import SLM_DQN
from src.replay_buffer import ReplayBuffer
from src.evaluate import evaluate_policy, tune_decision_threshold
from src.utils import load_config, set_seed, ensure_dir, linear_epsilon


def select_action(model, state, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, 1)

    with torch.no_grad():
        state_tensor = torch.tensor(
            state,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        q_values = model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def build_model_and_embeddings(cfg, template_to_id, device):
    if not cfg["slm"]["enabled"]:
        raise ValueError("This version expects slm.enabled=true in config.yaml.")

    slm_embedding_matrix = build_minilm_embedding_matrix(
        template_to_id=template_to_id,
        model_name=cfg["slm"]["model_name"],
        cache_dir=cfg["slm"]["cache_dir"],
        batch_size=cfg["slm"]["batch_size"],
        device=device
    ).to(device)

    vocab_size = len(template_to_id) + 1

    model = SLM_DQN(
        vocab_size=vocab_size,
        id_embedding_dim=cfg["model"]["id_embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        slm_embedding_matrix=slm_embedding_matrix,
        num_actions=2
    ).to(device)

    return model, slm_embedding_matrix, vocab_size


def train_dqn_on_sequences(
    model,
    sequences,
    cfg,
    device,
    phase_name: str = "Training",
    train_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    epsilon_start: Optional[float] = None,
    epsilon_end: Optional[float] = None,
    epsilon_decay_steps: Optional[int] = None,
):
    env = LogAnomalyEnv(
        sequences=sequences,
        sequence_length=cfg["sequence"]["sequence_length"],
        continue_penalty=cfg["env"]["continue_penalty"],
        correct_alert_reward=cfg["env"]["correct_alert_reward"],
        false_alert_penalty=cfg["env"]["false_alert_penalty"],
        missed_anomaly_penalty=cfg["env"]["missed_anomaly_penalty"],
        correct_normal_reward=cfg["env"]["correct_normal_reward"],
        early_detection_bonus=cfg["env"]["early_detection_bonus"],
        shuffle=True
    )

    target_model = SLM_DQN(
        vocab_size=model.id_embedding.num_embeddings,
        id_embedding_dim=cfg["model"]["id_embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        slm_embedding_matrix=model.slm_embedding.weight.detach().clone(),
        num_actions=2
    ).to(device)

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            learning_rate
            if learning_rate is not None
            else cfg["model"]["learning_rate"]
        )
    )

    replay_buffer = ReplayBuffer(
        capacity=cfg["model"]["replay_capacity"]
    )

    gamma = cfg["model"]["gamma"]
    batch_size = cfg["model"]["batch_size"]

    total_steps = int(
        train_steps
        if train_steps is not None
        else cfg["model"]["train_steps"]
    )

    warmup_steps = min(
        int(cfg["model"]["warmup_steps"]),
        max(1, len(sequences))
    )

    target_update_steps = cfg["model"]["target_update_steps"]

    eps_start = float(
        epsilon_start
        if epsilon_start is not None
        else cfg["model"]["epsilon_start"]
    )

    eps_end = float(
        epsilon_end
        if epsilon_end is not None
        else cfg["model"]["epsilon_end"]
    )

    eps_decay = int(
        epsilon_decay_steps
        if epsilon_decay_steps is not None
        else cfg["model"]["epsilon_decay_steps"]
    )

    state = env.reset()
    episode_reward = 0.0
    recent_rewards = []

    progress_bar = tqdm(
        range(1, total_steps + 1),
        desc=phase_name
    )

    for step in progress_bar:
        epsilon = linear_epsilon(
            step=step,
            start=eps_start,
            end=eps_end,
            decay_steps=eps_decay
        )

        action = select_action(
            model=model,
            state=state,
            epsilon=epsilon,
            device=device
        )

        next_state, reward, done, _ = env.step(action)

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)

            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            state = env.reset()
            episode_reward = 0.0

        if len(replay_buffer) >= warmup_steps:
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                batch_size=min(batch_size, len(replay_buffer)),
                device=device
            )

            q_values = model(states)

            q_selected = q_values.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze(1)

            with torch.no_grad():
                next_q_values = target_model(next_states)
                max_next_q_values = next_q_values.max(dim=1).values
                target_q = rewards + gamma * (1.0 - dones) * max_next_q_values

            loss = F.smooth_l1_loss(
                q_selected,
                target_q
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

        if step % target_update_steps == 0:
            target_model.load_state_dict(model.state_dict())

        if step % 500 == 0:
            avg_reward = (
                sum(recent_rewards) / len(recent_rewards)
                if recent_rewards
                else 0.0
            )

            progress_bar.set_postfix(
                {
                    "epsilon": f"{epsilon:.3f}",
                    "avg_reward": f"{avg_reward:.3f}"
                }
            )

    return model


def save_checkpoint(
    path,
    model,
    template_to_id,
    vocab_size,
    slm_embedding_matrix,
    cfg,
    extra=None
):
    ensure_dir(str(Path(path).parent))

    payload = {
        "model_state_dict": model.state_dict(),
        "template_to_id": template_to_id,
        "vocab_size": vocab_size,
        "slm_embedding_matrix": slm_embedding_matrix.detach().cpu(),
        "config": cfg,
    }

    if extra:
        payload.update(extra)

    torch.save(
        payload,
        path
    )

    print(f"Saved checkpoint to: {path}")


def print_final_test_summary(test_result, dataset_name):
    """
    Print compact test metrics at the very end of the CMD output.
    """
    m = test_result["metrics"]

    auroc_text = "N/A" if m["auroc"] is None else f"{m['auroc']:.4f}"
    auprc_text = "N/A" if m["auprc"] is None else f"{m['auprc']:.4f}"

    avg_step_text = (
        "N/A"
        if m["average_detection_step"] is None
        else f"{m['average_detection_step']:.4f}"
    )

    avg_ratio_text = (
        "N/A"
        if m["average_detection_ratio"] is None
        else f"{m['average_detection_ratio']:.4f}"
    )

    median_ratio_text = (
        "N/A"
        if m["median_detection_ratio"] is None
        else f"{m['median_detection_ratio']:.4f}"
    )

    print()
    print("#" * 80)
    print("FINAL TEST RESULTS SUMMARY")
    print("#" * 80)

    print(f"Dataset               : {dataset_name}")
    print(f"Number of sequences   : {m['num_sequences']}")
    print(f"Decision threshold    : {m['decision_threshold']:.4f}")
    print(f"Minimum alert step    : {m['min_alert_step']}")
    print()

    print("[Classification Metrics]")
    print(f"Accuracy              : {m['accuracy']:.4f}")
    print(f"Balanced Accuracy     : {m['balanced_accuracy']:.4f}")
    print(f"Precision             : {m['precision']:.4f}")
    print(f"Recall / TPR          : {m['recall']:.4f}")
    print(f"Specificity / TNR     : {m['specificity_tnr']:.4f}")
    print(f"F1-score              : {m['f1_score']:.4f}")
    print(f"FPR                   : {m['fpr']:.4f}")
    print(f"FNR                   : {m['fnr']:.4f}")
    print(f"MCC                   : {m['mcc']:.4f}")
    print(f"AUROC                 : {auroc_text}")
    print(f"AUPRC                 : {auprc_text}")
    print()

    print("[Confusion Matrix]")
    print("Labels: 0=normal, 1=anomaly")
    print(m["confusion_matrix"])
    print()

    print("[Early Detection Metrics]")
    print(f"Total anomalies       : {m['total_anomalies']}")
    print(f"Detected anomalies    : {m['detected_anomalies']}")
    print(f"Detection coverage    : {m['anomaly_detection_coverage']:.4f}")
    print(f"Avg detection step    : {avg_step_text}")
    print(f"Avg detection ratio   : {avg_ratio_text}")
    print(f"Median detect. ratio  : {median_ratio_text}")
    print(f"EDR@25%               : {m['edr_25']:.4f}")
    print(f"EDR@50%               : {m['edr_50']:.4f}")
    print(f"EDR@75%               : {m['edr_75']:.4f}")
    print()

    print("[RL Metrics]")
    print(f"Average reward        : {m['average_reward']:.4f}")
    print(f"Alert rate            : {m['alert_rate']:.4f}")
    print()

    print("[Cost Metrics]")
    print(f"FP unit cost          : {m['false_positive_unit_cost']:.4f}")
    print(f"FN unit cost          : {m['false_negative_unit_cost']:.4f}")
    print(f"Delay unit cost       : {m['delay_unit_cost']:.4f}")
    print(f"FP total cost         : {m['false_positive_total_cost']:.4f}")
    print(f"FN total cost         : {m['false_negative_total_cost']:.4f}")
    print(f"Delay total cost      : {m['delay_total_cost']:.4f}")
    print(f"Total cost            : {m['total_cost']:.4f}")
    print(f"Avg cost / sequence   : {m['average_cost_per_sequence']:.4f}")

    print("#" * 80)
    print()


def train_in_domain(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset_name = str(cfg["data"]["dataset_name"])
    print(f"In-domain dataset: {dataset_name}")

    train_sequences, val_sequences, test_sequences, template_to_id = (
        load_split_sequences_from_config(cfg)
    )

    describe_sequences(
        train_sequences,
        "Train"
    )

    describe_sequences(
        val_sequences,
        "Validation"
    )

    describe_sequences(
        test_sequences,
        "Test"
    )

    model, slm_embedding_matrix, vocab_size = build_model_and_embeddings(
        cfg=cfg,
        template_to_id=template_to_id,
        device=device
    )

    train_dqn_on_sequences(
        model=model,
        sequences=train_sequences,
        cfg=cfg,
        device=device,
        phase_name="In-domain training"
    )

    print("\nValidation result before threshold calibration:")
    val_result = evaluate_policy(
        model=model,
        sequences=val_sequences,
        cfg=cfg,
        device=device,
        name=f"{dataset_name} Validation"
    )

    if bool(
        cfg.get("evaluation", {}).get(
            "tune_threshold_on_validation",
            False
        )
    ):
        best_threshold, threshold_val_result = tune_decision_threshold(
            model=model,
            val_sequences=val_sequences,
            cfg=cfg,
            device=device
        )

        print()
        print(f"Using calibrated threshold for test: {best_threshold:.4f}")

        val_result = threshold_val_result

    print("\nFinal in-domain test result:")
    test_result = evaluate_policy(
        model=model,
        sequences=test_sequences,
        cfg=cfg,
        device=device,
        name=f"{dataset_name} Test"
    )

    checkpoint_path = (
        Path(cfg["output"]["output_dir"])
        / cfg["output"]["checkpoint_name"]
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        template_to_id=template_to_id,
        vocab_size=vocab_size,
        slm_embedding_matrix=slm_embedding_matrix,
        cfg=cfg,
        extra={
            "experiment_type": "in_domain",
            "dataset_name": dataset_name,
            "decision_threshold": cfg.get("evaluation", {}).get(
                "decision_threshold",
                0.0
            )
        }
    )

    print_final_test_summary(
        test_result=test_result,
        dataset_name=dataset_name
    )

    return {
        "model": model,
        "validation": val_result,
        "test": test_result,
    }