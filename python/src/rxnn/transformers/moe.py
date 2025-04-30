import torch
import torch.nn as nn
import torch.nn.functional as F

from .ff import FeedForward, GatedFeedForward

class MoeRouter(nn.Module):
    """Mixture-of-Experts Router layer - computes routing weights for each expert."""

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1, *args, **kwargs):
        super(MoeRouter, self).__init__(*args, **kwargs)
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # For expert load balancing
        self.register_buffer('aux_loss', torch.tensor(0.0), persistent=False)

    # def calculate_aux_loss(self, top_k_indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    #     expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
    #     expert_usage = expert_mask.sum(dim=0).mean(dim=0)
    #     mean_probs = probs.mean(dim=0)
    #     return (expert_usage * mean_probs).sum() * self.num_experts

    def calculate_aux_loss(self, top_k_indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # Get shapes
        T, K = top_k_indices.shape  # Batch, Sequence length, Top-K

        # 1. Compute expert selection mask (one-hot encoded)
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()  # (B, S, K, E)

        # 2. Total number of times each expert is selected
        expert_usage = expert_mask.sum(dim=(0, 1))  # (E,)

        # 3. Fraction of tokens assigned to each expert
        total_selections = T * K
        fraction_expert = expert_usage / (total_selections + 1e-6)  # (E,)

        # 4. Sum of probabilities for each expert's selected tokens
        probs_expanded = probs.unsqueeze(1).expand(-1, K, -1)  # (B_K, K, E)
        sum_probs = (probs_expanded * expert_mask).sum(dim=(0, 1))

        # 5. Average probability per expert (avoid division by zero)
        avg_probs = sum_probs / expert_usage.clamp(min=1e-6)  # (E,)

        # 6. Compute load balancing loss
        loss = (fraction_expert * avg_probs).sum() * self.num_experts

        return loss

    def forward(self, x: torch.Tensor):
        # Input shape: [batch*seq_len, embed_dim]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        # Get top-k experts for each token
        top_k_weights, top_k_indices = probs.topk(self.top_k, dim=-1)

        # Normalize weights (sum to 1 for each token)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        # Load Balance Loss
        self.aux_loss = self.calculate_aux_loss(top_k_indices, probs)

        return top_k_weights, top_k_indices


class MoeFeedForward(nn.Module):
    """Mixture-of-Experts Feed-Forward layer - combines multiple experts into a single model."""

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_experts: int,
            activation: nn.Module,
            top_k: int = 1,
            dropout: float = 0.0,
            *args,
            **kwargs
    ):
        super(MoeFeedForward, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoeRouter(embed_dim, num_experts, top_k)

        # Batch all expert parameters together
        self._init_experts(num_experts, embed_dim, hidden_dim, activation, dropout)

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        self.experts = nn.ModuleList([
            FeedForward(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])

    def router_loss(self):
        return self.router.aux_loss

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        x = x.view(-1, self.embed_dim)  # [batch*seq_len, embed_dim]

        # Get routing weights and indices
        weights, indices = self.router(x)  # [B*T, top_k], [B*T, top_k]

        # Create mask for expert contributions (B*T, num_experts)
        expert_mask = F.one_hot(indices, self.num_experts).float()  # [B*T, top_k, num_experts]
        expert_weights = (weights.unsqueeze(-1) * expert_mask).sum(dim=1)  # [B*T, num_experts]

        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Mask for tokens where this expert is in top_k
            mask = expert_weights[:, expert_idx] > 0
            if not mask.any():
                continue

            # Compute expert output for selected tokens
            expert_input = x[mask]
            expert_output = self.experts[expert_idx](expert_input)

            # Apply combined weights for this expert
            output[mask] += expert_output * expert_weights[mask, expert_idx].unsqueeze(-1)

        return output.view(*orig_shape)


class GatedMoeFeedForward(MoeFeedForward):
    """Gated Mixture-of-Experts Feed-Forward layer - enable GLU-based activations for MoE"""

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_experts: int,
            activation: nn.Module = nn.SiLU(),
            top_k: int = 1,
            dropout: float = 0.1,
            *args,
            **kwargs
    ):
        super(GatedMoeFeedForward, self).__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=activation,
            top_k=top_k,
            dropout=dropout,
            *args,
            **kwargs
        )

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        self.experts = nn.ModuleList([
            GatedFeedForward(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])
