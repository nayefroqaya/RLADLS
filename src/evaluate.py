import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

from src.data import load_split_sequences_from_config, describe_sequences
from src.env import LogAnomalyEnv
from src.model import SLM_DQN
from src.utils import load_config, set_seed, ensure_dir


def _safe_json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def run_one_episode(
    env,
    model,
    device,
    decision_threshold=0.0,
    min_alert_step=0
):
    """
    Run one episode.

    Post-hoc calibration:
        Alert only if Q_alert - Q_continue >= decision_threshold.

    min_alert_step:
        Prevents the agent from alerting too early.
        Example: min_alert_step=1 means no alert at step 0.
    """
    state = env.reset()
    done = False
    total_reward = 0.0

    true_label = None
    predicted_label = 0
    detection_step = None
    sequence_length = None
    sequence_id = None
    dataset_name = None

    max_q_alert = -float("inf")
    max_q_margin = -float("inf")
    final_q_alert = None
    final_q_continue = None
    final_q_margin = None

    while not done:
        state_tensor = torch.tensor(
            state,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

            q_continue = float(q_values[0, 0].item())
            q_alert = float(q_values[0, 1].item())
            q_margin = q_alert - q_continue

            # Calibrated decision rule.
            if env.t < min_alert_step:
                action = 0
            else:
                action = 1 if q_margin >= decision_threshold else 0

        max_q_alert = max(max_q_alert, q_alert)
        max_q_margin = max(max_q_margin, q_margin)

        final_q_continue = q_continue
        final_q_alert = q_alert
        final_q_margin = q_margin

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        true_label = info["label"]
        sequence_length = info["sequence_length"]
        sequence_id = info["sequence_id"]
        dataset_name = info.get("dataset_name", "")

        if info["alerted"]:
            predicted_label = 1
            detection_step = info["step"]

    return {
        "sequence_id": sequence_id,
        "dataset_name": dataset_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "detection_step": detection_step,
        "sequence_length": sequence_length,
        "reward": total_reward,
        "max_q_alert": max_q_alert,
        "max_q_margin": max_q_margin,
        "final_q_alert": final_q_alert,
        "final_q_continue": final_q_continue,
        "final_q_margin": final_q_margin,
    }


def compute_confusion_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_tnr": float(tn / max(1, tn + fp)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "fnr": float(fn / max(1, fn + tp)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm,
    }


def compute_score_metrics(y_true, anomaly_scores):
    if len(set(y_true)) < 2:
        return {
            "auroc": None,
            "auprc": None
        }

    try:
        auroc = float(roc_auc_score(y_true, anomaly_scores))
    except ValueError:
        auroc = None

    try:
        auprc = float(average_precision_score(y_true, anomaly_scores))
    except ValueError:
        auprc = None

    return {
        "auroc": auroc,
        "auprc": auprc
    }


def compute_early_detection_metrics(results):
    detection_steps = []
    detection_ratios = []

    early_25 = 0
    early_50 = 0
    early_75 = 0

    detected_anomalies = 0
    total_anomalies = 0

    for item in results:
        if item["true_label"] == 1:
            total_anomalies += 1

        if (
            item["true_label"] == 1
            and item["predicted_label"] == 1
            and item["detection_step"] is not None
        ):
            detected_anomalies += 1

            ratio = item["detection_step"] / max(
                1,
                item["sequence_length"] - 1
            )

            detection_steps.append(item["detection_step"])
            detection_ratios.append(ratio)

            if ratio <= 0.25:
                early_25 += 1
            if ratio <= 0.50:
                early_50 += 1
            if ratio <= 0.75:
                early_75 += 1

    return {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "anomaly_detection_coverage": float(
            detected_anomalies / max(1, total_anomalies)
        ),
        "average_detection_step": (
            None if not detection_steps else float(np.mean(detection_steps))
        ),
        "average_detection_ratio": (
            None if not detection_ratios else float(np.mean(detection_ratios))
        ),
        "median_detection_ratio": (
            None if not detection_ratios else float(np.median(detection_ratios))
        ),
        "edr_25": float(early_25 / max(1, detected_anomalies)),
        "edr_50": float(early_50 / max(1, detected_anomalies)),
        "edr_75": float(early_75 / max(1, detected_anomalies)),
    }


def compute_cost_metrics(results, cfg):
    eval_cfg = cfg.get("evaluation", {})

    fp_unit = float(eval_cfg.get("false_positive_cost", 10.0))
    fn_unit = float(eval_cfg.get("false_negative_cost", 20.0))
    delay_unit = float(eval_cfg.get("delay_cost", 5.0))

    fp_total = 0.0
    fn_total = 0.0
    delay_total = 0.0

    for item in results:
        y = item["true_label"]
        pred = item["predicted_label"]

        if y == 0 and pred == 1:
            fp_total += fp_unit

        elif y == 1 and pred == 0:
            fn_total += fn_unit

        elif y == 1 and pred == 1:
            ratio = item["detection_step"] / max(
                1,
                item["sequence_length"] - 1
            )
            delay_total += delay_unit * ratio

    total = fp_total + fn_total + delay_total

    return {
        "false_positive_unit_cost": fp_unit,
        "false_negative_unit_cost": fn_unit,
        "delay_unit_cost": delay_unit,
        "false_positive_total_cost": float(fp_total),
        "false_negative_total_cost": float(fn_total),
        "delay_total_cost": float(delay_total),
        "total_cost": float(total),
        "average_cost_per_sequence": float(total / max(1, len(results))),
    }


def save_evaluation_outputs(results, metrics, cfg, name):
    eval_cfg = cfg.get("evaluation", {})

    if not bool(eval_cfg.get("save_results", False)):
        return

    results_dir = Path(
        eval_cfg.get(
            "results_dir",
            "outputs/evaluation_results"
        )
    )

    ensure_dir(str(results_dir))

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    predictions_path = results_dir / f"{safe_name}_predictions.csv"
    metrics_path = results_dir / f"{safe_name}_metrics.json"

    pd.DataFrame(results).to_csv(
        predictions_path,
        index=False
    )

    json_metrics = {}

    for key, value in metrics.items():
        if key == "confusion_matrix":
            json_metrics[key] = value.tolist()
        else:
            json_metrics[key] = _safe_json_value(value)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            json_metrics,
            file,
            indent=2
        )

    print()
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics    : {metrics_path}")


def _format_optional(value):
    return "N/A" if value is None else f"{value:.4f}"


def print_metric_report(metrics, y_true, y_pred, name):
    print()
    print("=" * 70)
    print(f"{name.upper()} FINAL METRIC REPORT")
    print("=" * 70)

    print("\n[Classification Metrics]")
    print(f"Accuracy              : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy     : {metrics['balanced_accuracy']:.4f}")
    print(f"Precision             : {metrics['precision']:.4f}")
    print(f"Recall / TPR          : {metrics['recall']:.4f}")
    print(f"Specificity / TNR     : {metrics['specificity_tnr']:.4f}")
    print(f"F1-score              : {metrics['f1_score']:.4f}")
    print(f"FPR                   : {metrics['fpr']:.4f}")
    print(f"FNR                   : {metrics['fnr']:.4f}")
    print(f"MCC                   : {metrics['mcc']:.4f}")
    print(f"AUROC                 : {_format_optional(metrics['auroc'])}")
    print(f"AUPRC                 : {_format_optional(metrics['auprc'])}")

    print("\n[Confusion Matrix]")
    print("Labels: 0=normal, 1=anomaly")
    print(metrics["confusion_matrix"])

    print("\n[Classification Report]")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0
        )
    )

    print("\n[Early Detection Metrics]")
    print(f"Total anomalies       : {metrics['total_anomalies']}")
    print(f"Detected anomalies    : {metrics['detected_anomalies']}")
    print(f"Detection coverage    : {metrics['anomaly_detection_coverage']:.4f}")
    print(f"Avg detection step    : {_format_optional(metrics['average_detection_step'])}")
    print(f"Avg detection ratio   : {_format_optional(metrics['average_detection_ratio'])}")
    print(f"Median detect. ratio  : {_format_optional(metrics['median_detection_ratio'])}")
    print(f"EDR@25%               : {metrics['edr_25']:.4f}")
    print(f"EDR@50%               : {metrics['edr_50']:.4f}")
    print(f"EDR@75%               : {metrics['edr_75']:.4f}")

    print("\n[RL Metrics]")
    print(f"Average reward        : {metrics['average_reward']:.4f}")
    print(f"Alert rate            : {metrics['alert_rate']:.4f}")

    print("\n[Cost Metrics]")
    print(f"FP unit cost          : {metrics['false_positive_unit_cost']:.4f}")
    print(f"FN unit cost          : {metrics['false_negative_unit_cost']:.4f}")
    print(f"Delay unit cost       : {metrics['delay_unit_cost']:.4f}")
    print(f"FP total cost         : {metrics['false_positive_total_cost']:.4f}")
    print(f"FN total cost         : {metrics['false_negative_total_cost']:.4f}")
    print(f"Delay total cost      : {metrics['delay_total_cost']:.4f}")
    print(f"Total cost            : {metrics['total_cost']:.4f}")
    print(f"Avg cost per sequence : {metrics['average_cost_per_sequence']:.4f}")

    print("\n" + "=" * 70)


def evaluate_policy(
    model,
    sequences,
    cfg,
    device,
    name="Evaluation",
    print_report=True,
    save_outputs=True
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
        shuffle=False
    )

    eval_cfg = cfg.get("evaluation", {})

    decision_threshold = float(
        eval_cfg.get("decision_threshold", 0.0)
    )

    min_alert_step = int(
        eval_cfg.get("min_alert_step", 0)
    )

    show_progress = bool(
        eval_cfg.get("show_progress", True)
    )

    model.eval()
    results = []

    iterator = range(len(sequences))

    if show_progress:
        iterator = tqdm(
            iterator,
            desc=f"Evaluating {name}",
            unit="seq"
        )

    for _ in iterator:
        results.append(
            run_one_episode(
                env=env,
                model=model,
                device=device,
                decision_threshold=decision_threshold,
                min_alert_step=min_alert_step
            )
        )

    y_true = [
        item["true_label"]
        for item in results
    ]

    y_pred = [
        item["predicted_label"]
        for item in results
    ]

    anomaly_scores = [
        item["max_q_margin"]
        for item in results
    ]

    rewards = [
        item["reward"]
        for item in results
    ]

    metrics = {}
    metrics.update(
        compute_confusion_metrics(
            y_true,
            y_pred
        )
    )
    metrics.update(
        compute_score_metrics(
            y_true,
            anomaly_scores
        )
    )
    metrics.update(
        compute_early_detection_metrics(
            results
        )
    )
    metrics.update(
        compute_cost_metrics(
            results,
            cfg
        )
    )

    metrics["average_reward"] = float(np.mean(rewards))
    metrics["alert_rate"] = float(np.mean(y_pred)) if y_pred else 0.0
    metrics["num_sequences"] = int(len(results))
    metrics["decision_threshold"] = float(decision_threshold)
    metrics["min_alert_step"] = int(min_alert_step)

    if print_report:
        print_metric_report(
            metrics,
            y_true,
            y_pred,
            name
        )

    if save_outputs:
        save_evaluation_outputs(
            results,
            metrics,
            cfg,
            name
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "anomaly_scores": anomaly_scores,
        "results": results,
        "metrics": metrics,
    }


def tune_decision_threshold(
    model,
    val_sequences,
    cfg,
    device
):
    """
    Tune decision threshold on validation data.

    The selected threshold is stored in:
        cfg["evaluation"]["decision_threshold"]

    Supported metrics:
        f1
        cost
        precision
        recall
        balanced_accuracy
    """
    eval_cfg = cfg.setdefault("evaluation", {})

    threshold_min = float(
        eval_cfg.get("threshold_min", -2.0)
    )

    threshold_max = float(
        eval_cfg.get("threshold_max", 5.0)
    )

    threshold_steps = int(
        eval_cfg.get("threshold_steps", 50)
    )

    threshold_metric = str(
        eval_cfg.get("threshold_metric", "f1")
    ).lower()

    original_threshold = float(
        eval_cfg.get("decision_threshold", 0.0)
    )

    thresholds = np.linspace(
        threshold_min,
        threshold_max,
        threshold_steps
    )

    best_threshold = original_threshold
    best_score = -float("inf")
    best_result = None

    print()
    print("=" * 70)
    print("TUNING DECISION THRESHOLD ON VALIDATION SET")
    print("=" * 70)
    print(f"Metric          : {threshold_metric}")
    print(f"Threshold range : {threshold_min} to {threshold_max}")
    print(f"Steps           : {threshold_steps}")

    for threshold in tqdm(
        thresholds,
        desc="Threshold tuning",
        unit="thr"
    ):
        eval_cfg["decision_threshold"] = float(threshold)

        result = evaluate_policy(
            model=model,
            sequences=val_sequences,
            cfg=cfg,
            device=device,
            name=f"Validation threshold {threshold:.4f}",
            print_report=False,
            save_outputs=False
        )

        metrics = result["metrics"]

        if threshold_metric == "cost":
            score = -metrics["total_cost"]

        elif threshold_metric == "precision":
            score = metrics["precision"]

        elif threshold_metric == "recall":
            score = metrics["recall"]

        elif threshold_metric == "balanced_accuracy":
            score = metrics["balanced_accuracy"]

        else:
            score = metrics["f1_score"]

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_result = result

    eval_cfg["decision_threshold"] = best_threshold

    print()
    print("=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)
    print(f"Best threshold : {best_threshold:.4f}")
    print(f"Metric         : {threshold_metric}")
    print(f"Best score     : {best_score:.4f}")

    if best_result is not None:
        m = best_result["metrics"]
        print(f"Validation F1  : {m['f1_score']:.4f}")
        print(f"Validation P   : {m['precision']:.4f}")
        print(f"Validation R   : {m['recall']:.4f}")
        print(f"Validation FPR : {m['fpr']:.4f}")
        print(f"Validation Cost: {m['total_cost']:.4f}")

    print("=" * 70)

    return best_threshold, best_result


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml"
):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    _, _, test_sequences, _ = load_split_sequences_from_config(cfg)

    describe_sequences(
        test_sequences,
        "Test"
    )

    slm_embedding_matrix = checkpoint["slm_embedding_matrix"].to(device)

    model = SLM_DQN(
        vocab_size=checkpoint["vocab_size"],
        id_embedding_dim=cfg["model"]["id_embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        slm_embedding_matrix=slm_embedding_matrix,
        num_actions=2
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return evaluate_policy(
        model=model,
        sequences=test_sequences,
        cfg=cfg,
        device=device,
        name="Test"
    )