from typing import List, Tuple, Dict, Any
import numpy as np

from src.data import LogSequence


class LogAnomalyEnv:
    """
    RL environment for log anomaly detection.

    One sequence = one episode.

    Actions:
        0 = continue observing
        1 = alert anomaly

    Training reward uses reward_label when available.
    Evaluation metrics still use the true label stored in label.
    """

    CONTINUE = 0
    ALERT = 1

    def __init__(
        self,
        sequences: List[LogSequence],
        sequence_length: int = 120,
        continue_penalty: float = -0.05,
        correct_alert_reward: float = 10.0,
        false_alert_penalty: float = -10.0,
        missed_anomaly_penalty: float = -20.0,
        correct_normal_reward: float = 5.0,
        early_detection_bonus: float = 2.0,
        shuffle: bool = True,
    ):
        if len(sequences) == 0:
            raise ValueError("Environment received zero sequences.")

        self.sequences = sequences
        self.sequence_length = sequence_length
        self.continue_penalty = continue_penalty
        self.correct_alert_reward = correct_alert_reward
        self.false_alert_penalty = false_alert_penalty
        self.missed_anomaly_penalty = missed_anomaly_penalty
        self.correct_normal_reward = correct_normal_reward
        self.early_detection_bonus = early_detection_bonus
        self.shuffle = shuffle

        self.order = np.arange(len(sequences))
        self.pointer = 0
        self.current = None
        self.t = 0
        self.done = False

        if self.shuffle:
            np.random.shuffle(self.order)

    def reset(self) -> np.ndarray:
        if self.pointer >= len(self.order):
            self.pointer = 0
            if self.shuffle:
                np.random.shuffle(self.order)

        index = self.order[self.pointer]
        self.pointer += 1

        self.current = self.sequences[index]
        self.t = 0
        self.done = False

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() first.")

        sequence = self.current.sequence
        true_label = self.current.label
        reward_label = self.current.reward_label

        if reward_label is None:
            reward_label = true_label

        reward_weight = float(self.current.reward_weight)
        sequence_len = len(sequence)

        info = {
            "sequence_id": self.current.sequence_id,
            "dataset_name": self.current.dataset_name,
            "split_name": self.current.split_name,
            "label": true_label,
            "reward_label": reward_label,
            "is_labeled": self.current.is_labeled,
            "anomaly_score": self.current.anomaly_score,
            "reward_weight": reward_weight,
            "step": self.t,
            "sequence_length": sequence_len,
            "alerted": False,
        }

        if action == self.ALERT:
            self.done = True
            info["alerted"] = True

            if reward_label == 1:
                progress = self.t / max(1, sequence_len - 1)
                early_bonus = self.early_detection_bonus * (1.0 - progress)
                reward = self.correct_alert_reward + early_bonus
            else:
                reward = self.false_alert_penalty

            return self._get_observation(), float(reward * reward_weight), self.done, info

        if action == self.CONTINUE:
            self.t += 1

            if self.t >= sequence_len:
                self.done = True

                if reward_label == 0:
                    reward = self.correct_normal_reward
                else:
                    reward = self.missed_anomaly_penalty

                return self._get_observation(), float(reward * reward_weight), self.done, info

            return self._get_observation(), float(self.continue_penalty * reward_weight), False, info

        raise ValueError(f"Unknown action: {action}")

    def _get_observation(self) -> np.ndarray:
        prefix_sequence = self.current.sequence[: self.t + 1]
        prefix_sequence = prefix_sequence[-self.sequence_length:]
        pad_length = self.sequence_length - len(prefix_sequence)
        observation = [0] * pad_length + prefix_sequence

        return np.array(observation, dtype=np.int64)
