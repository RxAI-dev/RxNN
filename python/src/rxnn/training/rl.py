import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import TypedDict, Optional
from .utils import TokenizedDict
from .ddp import distributed_mean


class RlAlgorithm(ABC):
    def __init__(self):
        super(RlAlgorithm, self).__init__()
        self.critic_loss_fn = nn.MSELoss()

    @abstractmethod
    def policy_loss(self, query: TokenizedDict, answer: TokenizedDict, logits: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def critic_loss(self, values: torch.Tensor, ref_values: torch.Tensor) -> torch.Tensor:
        return self.critic_loss_fn(values, ref_values)


class PPOConfig(TypedDict):
    clip_eps: Optional[float]
    gae_lambda: Optional[float]
    gae_gamma: Optional[float]
    entropy_coef: Optional[float]
    use_distributed_advantage_norm: Optional[bool]
    clip_critic_values: Optional[bool]
    critic_value_clip: Optional[float]
    debug_mode: Optional[bool]
    debug_interval: Optional[int]


class PPOAlgorithm(RlAlgorithm):
    def __init__(self, config: Optional[PPOConfig] = None):
        super(PPOAlgorithm, self).__init__()

        if config is None:
            config = {}

        # PPO Config
        self.clip_eps = config.get('clip_eps', 0.2)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.gae_gamma = config.get('gae_gamma', 0.99)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.use_distributed_advantage_norm = config.get('use_distributed_advantage_norm', False)
        self.clip_critic_values = config.get('clip_critic_values', True)
        self.critic_value_clip = config.get('critic_value_clip', 20.0)
        self.debug_mode = config.get('debug_mode', False)
        self.debug_interval = config.get('debug_interval', 10)
        self.debug_step = 0

    def critic_loss(self, values: torch.Tensor, ref_values: torch.Tensor) -> torch.Tensor:
        # Critic loss with clipped values
        if self.clip_critic_values:
            values = torch.clamp(values, -self.critic_value_clip, self.critic_value_clip)
            ref_values = torch.clamp(ref_values, -self.critic_value_clip, self.critic_value_clip)
        return self.critic_loss_fn(values, ref_values)

    def policy_loss(self, query: TokenizedDict, answer: TokenizedDict, logits: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        # 1. Get query, answer, max and combined lengths in batch
        query_lens = query['attention_mask'].sum(dim=1).long()  # Query lengths per sample
        answer_mask = answer['attention_mask']
        answer_lens = answer_mask.sum(dim=1).long()  # Answer lengths per sample (before padding)
        max_length = query['input_ids'].size(1)

        combined_lens = torch.minimum(
            query_lens + answer_lens,
            torch.full_like(query_lens, max_length)
        )
        # 2. Extract only answer logits
        batch_size, _, vocab_size = logits.size()
        new_logits = torch.zeros((batch_size, max_length, vocab_size), dtype=logits.dtype, device=logits.device)

        for i in range(batch_size):
            start = query_lens[i].item()
            end = combined_lens[i].item()
            valid_len = end - start
            if valid_len > 0:
                new_logits[i, :valid_len] = logits[i, start:end]

        # 3. Shift sequences for correct probabilities alignment
        shifted_logits = new_logits[:, :-1, :] # Remove last sequence element logits - most likely padding or [EOS]
        shifted_targets = answer['input_ids'][:, 1:] # Remove first answer token - deterministic [A] token
        shifted_mask = answer_mask[:, 1:] # Remove also first position from attention mask
        shifted_old_log_probs = old_log_probs[:, 1:] # And from old log probs - it's for [A] deterministic token

        # 4. Calculate and mask new shifted log probs
        new_log_probs = F.log_softmax(shifted_logits, dim=-1)
        shifted_log_probs = new_log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        shifted_log_probs *= shifted_mask
        shifted_old_log_probs *= shifted_mask

        # 5. Calculate ratio
        ratio = (shifted_log_probs - shifted_old_log_probs).exp()

        advantages = advantages.unsqueeze(-1)

        # 6. Log most important stats in debug mode
        if self.debug_mode:
            if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                self.debug_step = 0
                print(
                    f"Logits stats: min={new_logits.min().item():.4f}, max={new_logits.max().item():.4f}, mean={new_logits.mean().item():.4f}")
                print(
                    f"Ratio stats: min={ratio.min().item():.4f}, max={ratio.max().item():.4f}, mean={ratio.mean().item():.4f}")
                print(
                    f"Advantage stats: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, mean={advantages.mean().item():.4f}")
            else:
                self.debug_step += 1

        # 7. Calculate base policy loss
        surr1 = (ratio * shifted_mask) * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * shifted_mask) * advantages
        policy_loss = -(torch.min(surr1, surr2).sum(dim=-1) / shifted_mask.sum(dim=-1)).mean()

        # 8. Add Entropy bonus
        entropy = -(
            (new_log_probs * new_log_probs.exp() * shifted_mask).sum(dim=-1) / shifted_mask.sum(dim=-1)
        ).mean()
        policy_loss -= self.entropy_coef * entropy

        return policy_loss

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     last_value: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_len, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards, device=rewards.device)
        last_advantage = 0
        next_value = last_value
        dones = dones.float()

        for t in reversed(range(trajectory_len)):
            # Calculate delta from rewards, stored next_value, masked by stored next_done, and values
            delta = rewards[t] + self.gae_gamma * next_value * (1 - dones[t]) - values[t]
            # Calculate advantages based on delta, gamma/lambda factors and last advantage, masked by current done flags
            advantages[t] = delta + self.gae_gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            # Store current step data as last_advantage, next_done and next_value, for the next iteration step
            last_advantage = advantages[t]
            next_value = values[t]

        # Calculate reference returns, based on advantages and values, and return them with advantages for critic update
        returns = advantages + values
        return advantages, returns

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        advantages, ref_values = self._compute_gae(rewards[:-1], values[:-1], values[-1], dones[:-1])
        if self.use_distributed_advantage_norm:
            mean_advantage = distributed_mean(advantages.mean())
            std_advantage = distributed_mean(advantages.std())
            normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)
        else:
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return normalized_advantages, ref_values
