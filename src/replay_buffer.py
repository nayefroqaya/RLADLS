from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.int64),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.int64),
                float(done),
            )
        )

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.stack(states),
            dtype=torch.long,
            device=device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device
        )

        next_states = torch.tensor(
            np.stack(next_states),
            dtype=torch.long,
            device=device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
