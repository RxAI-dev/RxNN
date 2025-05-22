import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import TypedDict


class RlAlgorithm(ABC):
    def __init__(self):
        super(RlAlgorithm, self).__init__()
        self.critic_loss = nn.MSELoss()

    @abstractmethod
    def policy_loss(self, input_ids: torch.Tensor, logits: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pass

    def critic_loss(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return self.critic_loss(rewards, values)

class PPOConfig(TypedDict):
    gae_gamma: float
    gae_lambda: float
    clip_eps: float

class PPOAlgorithm(RlAlgorithm):
    def __init__(self, config: PPOConfig):
        super(PPOAlgorithm, self).__init__()

        # PPO Config
        self.gae_gamma = config.get('gae_gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)

    def policy_loss(self, input_ids: torch.Tensor, logits: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        # a) Get new log probs
        new_probs = F.log_softmax(logits, dim=-1)
        new_log_probs = new_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        # b) Calculate ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # c) Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # d) Entropy bonus
        entropy = -torch.sum(new_probs * new_probs.exp(), dim=-1).mean()
        policy_loss -= 0.01 * entropy

        return policy_loss

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards, device=values.device)
        last_advantage = 0
        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + self.gae_gamma * next_value - values[t]
            advantages[t] = delta + self.gae_gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
        return advantages

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = self._compute_gae(rewards, values[:-1], values[-1])
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return normalized_advantages
