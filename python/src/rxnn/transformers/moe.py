import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeRouter(nn.Module):
    """Mixture-of-Experts Router layer - computes routing weights for each expert."""

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1, *args, **kwargs):
        super(MoeRouter, self).__init__(*args, **kwargs)
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # For expert load balancing
        self.register_buffer('aux_loss', torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size*seq_len, embed_dim]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        # Expert load balancing loss
        mean_probs = probs.mean(dim=0)  # Mean probability per expert across batch
        self.aux_loss = (mean_probs * torch.log(mean_probs + 1e-9)).sum()  # Entropy-based loss

        top_k_weights, top_k_indices = probs.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

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
        self.w1 = nn.Parameter(torch.empty(num_experts, embed_dim, self._w1_dim_factor(hidden_dim)))
        self.b1 = nn.Parameter(torch.zeros(num_experts, self._w1_dim_factor(hidden_dim)))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, embed_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, embed_dim))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_linear_parameters()
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def _init_linear_parameters(self):
        nn.init.kaiming_normal_(self.w1, nonlinearity='relu')
        nn.init.kaiming_normal_(self.w2, nonlinearity='relu')

    def _w1_dim_factor(self, hidden_dim: int) -> int:
        return hidden_dim

    def _activate(self, h: torch.Tensor):
        return self.activation(h)

    def router_loss(self):
        return self.router.aux_loss

    def forward(self, x: torch.Tensor):
        # orig_shape = x.shape
        # x = x.view(-1, self.embed_dim)  # [batch*seq_len, embed_dim]
        #
        # # Get routing weights and indices
        # weights, indices = self.router(x)  # [batch*seq_len, top_k]
        #
        # # Create expert masks and combine it with masks
        # mask = F.one_hot(indices, self.num_experts).float()  # [batch*seq_len, top_k, num_experts]
        # weights = (weights.unsqueeze(-1) * mask).sum(dim=1)  # [batch*seq_len, num_experts]
        #
        # # Expert computation
        # x = x.unsqueeze(1).expand(-1, self.num_experts, -1)  # [batch*seq_len, num_experts, embed_dim]
        #
        # # First linear layer
        # h = torch.einsum('bie,ieh->bih', x, self.w1) + self.b1  # [batch*seq_len, num_experts, hidden_dim]
        # h = self._activate(h)
        # h = self.dropout(h)
        #
        # # Second linear layer (projection back to embed_dim)
        # out = torch.einsum('bih,ihe->bie', h, self.w2) + self.b2  # [batch*seq_len, num_experts, embed_dim]
        #
        # # Weighted sum of expert outputs
        # out = (out * weights.unsqueeze(-1)).sum(dim=1)  # [batch*seq_len, embed_dim]
        #
        # return out.view(*orig_shape)
        orig_shape = x.shape
        x = x.view(-1, self.embed_dim)  # [batch*seq_len, embed_dim]

        # Get routing weights and indices
        weights, indices = self.router(x)  # [batch*seq_len, top_k], [batch*seq_len, top_k]

        # Flatten indices and weights
        batch_size = x.size(0)
        top_k = indices.size(1)
        indices = indices.view(-1)  # [batch*seq_len * top_k]
        weights = weights.view(-1, 1)  # [batch*seq_len * top_k, 1]

        # Select only the relevant experts for each token
        selected_w1 = self.w1[indices]  # [batch*seq_len * top_k, embed_dim, hidden_dim]
        selected_b1 = self.b1[indices]  # [batch*seq_len * top_k, hidden_dim]
        selected_w2 = self.w2[indices]  # [batch*seq_len * top_k, hidden_dim, embed_dim]
        selected_b2 = self.b2[indices]  # [batch*seq_len * top_k, embed_dim]

        # Reshape x for batched computation
        x_expanded = x.unsqueeze(1).repeat(1, top_k, 1).view(-1, self.embed_dim)  # [batch*seq_len * top_k, embed_dim]

        # Compute only the selected experts
        h = torch.einsum('be, beh -> bh', x_expanded, selected_w1) + selected_b1
        h = self._activate(h)
        h = self.dropout(h)

        out = torch.einsum('bh, bhe -> be', h, selected_w2) + selected_b2

        # Reshape back and apply weights
        out = out.view(batch_size, top_k, -1)  # [batch*seq_len, top_k, embed_dim]
        weights = weights.view(batch_size, top_k, 1)  # [batch*seq_len, top_k, 1]
        out = (out * weights).sum(dim=1)  # Weighted sum over top_k experts

        return out.view(*orig_shape)


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

    def _init_linear_parameters(self):
        nn.init.kaiming_normal_(self.w1, nonlinearity='relu')
        nn.init.kaiming_normal_(self.w2, nonlinearity='linear')

    def _w1_dim_factor(self, hidden_dim: int) -> int:
        return 2 * hidden_dim

    def _activate(self, h: torch.Tensor):
        a, b = h.chunk(2, dim=-1)
        return a * self.activation(b)
