import torch
from torch import nn
from ..transformers.attention import MultiHeadAttention, GroupedQueryAttention
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.moe import MoeRouter

class MoeGroupedAttention(GroupedQueryAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_groups: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            num_experts: int = None,
            *args,
            **kwargs,
    ):
        self.num_experts = num_experts if num_experts is not None else num_heads
        super(MoeGroupedAttention, self).__init__(
            embed_dim,
            num_heads,
            num_groups=num_groups,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs,
        )

    def _init_kv(self, embed_dim: int):
        self.router = MoeRouter(embed_dim, self.num_experts, top_k=self.num_groups)
        hidden_dim = embed_dim // (self.num_heads // self.num_groups)
        self.wk = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        self.bk = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        self.wv = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        self.bv = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        self._init_experts()

    def _init_experts(self):
        torch.nn.init.xavier_uniform_(self.wk)
        torch.nn.init.xavier_uniform_(self.wv)
        if self.use_bias:
            torch.nn.init.zeros_(self.bk)
            torch.nn.init.zeros_(self.bv)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int):
        head_dim = d // self.num_heads
        group_heads = self.num_heads // self.num_groups

        # Process Query as in GQA
        q = self.q_proj(query).view(b, t, self.num_heads, head_dim).transpose(1, 2)

        # Process Key and Value with MoE routing
        key_flat = key.view(-1, d)
        weights, indices = self.router(key_flat)
        weights = weights.view(b, key.size(1), self.num_groups, 1)
        indices = indices.view(b, key.size(1), self.num_groups)

        # Compute all experts' K and V projections
        # Shape: (batch_size, seq_len, num_experts, head_dim * num_groups)
        k_all = torch.einsum(
            'be, ehd -> bedh',
            key_flat,
            self.wk.view(self.num_experts, d, -1)
        ).view(b, key.size(1), self.num_experts, -1)

        v_all = torch.einsum(
            'be, ehd -> bedh',
            value.view(-1, d),
            self.wv.view(self.num_experts, d, -1)
        ).view(b, value.size(1), self.num_experts, -1)

        # Select top_k experts and compute weighted sum
        selected_k = torch.gather(
            k_all,
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, k_all.size(-1))
        )
        selected_v = torch.gather(
            v_all,
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, v_all.size(-1))
        )

        selected_k = (selected_k * weights).sum(dim=2)
        selected_v = (selected_v * weights).sum(dim=2)
        # Reshape to GQA format: (B, G, S, head_dim)
        k = selected_k.view(b, key.size(1), self.num_groups, head_dim).transpose(1, 2)
        v = selected_v.view(b, value.size(1), self.num_groups, head_dim).transpose(1, 2)

        if not self.use_flash_attention:
            group_heads = self.num_heads // self.num_groups

            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)

        return q, k, v


class MoeSparseAttention(GroupedQueryAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_groups: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            num_experts: int = None,
            num_query_experts: int = None,
            num_active_query_heads: int = None,
            *args,
            **kwargs,
    ):
        self.num_experts = num_experts if num_experts is not None else num_heads
        self.num_query_experts = num_query_experts if num_query_experts is not None else num_heads
        self.num_active_query_heads = num_active_query_heads if num_active_query_heads is not None else num_groups
        super(MoeSparseAttention, self).__init__(
            embed_dim,
            num_heads,
            num_groups=num_groups,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs,
        )

    def _init_q(self, embed_dim: int):
        self.query_router = MoeRouter(embed_dim, self.num_query_experts, top_k=self.num_active_query_heads)
        hidden_dim = embed_dim // (self.num_heads // self.num_groups)
        self.wq = nn.Parameter(torch.empty(self.num_query_experts, embed_dim, hidden_dim))
        self.bq = nn.Parameter(torch.zeros(self.num_query_experts, hidden_dim)) if self.use_bias else None
        self._init_query_experts()

    def _init_kv(self, embed_dim: int):
        self.router = MoeRouter(embed_dim, self.num_experts, top_k=self.num_groups)
        hidden_dim = embed_dim // (self.num_heads // self.num_groups)
        self.wk = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        self.bk = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        self.wv = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        self.bv = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        self._init_experts()

    def _init_query_experts(self):
        torch.nn.init.xavier_uniform_(self.wq)
        if self.use_bias:
            torch.nn.init.zeros_(self.bq)

    def _init_experts(self):
        torch.nn.init.xavier_uniform_(self.wk)
        torch.nn.init.xavier_uniform_(self.wv)
        if self.use_bias:
            torch.nn.init.zeros_(self.bk)
            torch.nn.init.zeros_(self.bv)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int):
        head_dim = d // self.num_heads

        # Process Query with MoE routing
        query_flat = query.view(-1, d)
        weights_q, indices_q = self.router_q(query_flat)
        weights_q = weights_q.view(b, t, self.num_active_query_heads, 1)
        indices_q = indices_q.view(b, t, self.num_active_query_heads)

        # Compute all experts' Q projections
        q_all = torch.einsum(
            'be, ehd -> bedh',
            query_flat,
            self.q_experts.view(self.num_query_experts, d, -1)
        ).view(b, t, self.num_query_experts, -1)

        selected_q = torch.gather(
            q_all,
            2,
            indices_q.unsqueeze(-1).expand(-1, -1, -1, q_all.size(-1))
        )
        selected_q = (selected_q * weights_q).sum(dim=2)

        q = selected_q.view(b, t, self.num_heads, head_dim).transpose(1, 2)

        # Process Key and Value with MoE routing
        key_flat = key.view(-1, d)
        weights, indices = self.router(key_flat)
        weights = weights.view(b, key.size(1), self.num_groups, 1)
        indices = indices.view(b, key.size(1), self.num_groups)

        # Compute all experts' K and V projections
        # Shape: (batch_size, seq_len, num_experts, head_dim * num_groups)
        k_all = torch.einsum(
            'be, ehd -> bedh',
            key_flat,
            self.wk.view(self.num_experts, d, -1)
        ).view(b, key.size(1), self.num_experts, -1)

        v_all = torch.einsum(
            'be, ehd -> bedh',
            value.view(-1, d),
            self.wv.view(self.num_experts, d, -1)
        ).view(b, value.size(1), self.num_experts, -1)

        # Select top_k experts and compute weighted sum
        selected_k = torch.gather(
            k_all,
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, k_all.size(-1))
        )
        selected_v = torch.gather(
            v_all,
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, v_all.size(-1))
        )

        selected_k = (selected_k * weights).sum(dim=2)
        selected_v = (selected_v * weights).sum(dim=2)

        # Reshape to GQA format: (B, G, S, head_dim)
        k = selected_k.view(b, key.size(1), self.num_groups, head_dim).transpose(1, 2)
        v = selected_v.view(b, value.size(1), self.num_groups, head_dim).transpose(1, 2)

        if not self.use_flash_attention:
            group_heads = self.num_heads // self.num_groups

            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)

        return q, k, v

class FlexAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_global_tokens: int = 16,
        window_size: int = 128,
        **kwargs
    ):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, embed_dim))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        b, t, d = query.size()
        head_dim = d // self.num_heads

        # Split into global and local
        x = torch.cat([self.global_tokens.expand(b, -1, -1), query], dim=1)
        seq_len = x.size(1)
        num_windows = (seq_len - self.num_global_tokens + self.window_size - 1) // self.window_size

        # Project Q, K, V
        q, k, v = self._forward_qkv(x, key, value, b, seq_len, d)

        # Process Global-to-Global Attention
        global_q = q[:, :, :self.num_global_tokens]  # [B, H, G, head_dim]
        global_k = k[:, :, :self.num_global_tokens]
        global_v = v[:, :, :self.num_global_tokens]
        global_attn = self._calculate_attn_weights(global_q, global_k, d) @ global_v

        # Process Global-to-Local Attention
        local_k = k[:, :, self.num_global_tokens:]  # [B, H, (num_windows * window_size), head_dim]
        local_v = v[:, :, self.num_global_tokens:]
        # Apply RoPE to local_k if needed
        if self.rope:
            # Compute frequencies for entire local sequence
            local_k = self.rope.forward_one(local_k)

        global_local_attn = self._calculate_attn_weights(global_q, local_k, d) @ local_v

        # Process Local-to-Local Attention (per window)
        local_q = q[:, :, self.num_global_tokens:]  # [B, H, (num_windows * window_size), head_dim]
        local_q = local_q.view(b, self.num_heads, num_windows, self.window_size, head_dim)
        local_k = local_k.view(b, self.num_heads, num_windows, self.window_size, head_dim)
        local_v = local_v.view(b, self.num_heads, num_windows, self.window_size, head_dim)

        local_attn = []
        for i in range(num_windows):
            window_q = local_q[:, :, i]  # [B, H, window_size, head_dim]
            window_k = local_k[:, :, i]
            window_v = local_v[:, :, i]

            # Apply RoPE to window_q and window_k
            if self.rope:
                # Compute frequencies for this window
                window_q, window_k = self.rope(window_q, window_k)

            # Calculate attention for this window
            attn = self._calculate_attn_weights(window_q, window_k, d)
            attn_i = torch.einsum('bhij, bhjd -> bhid', attn, window_v)
            local_attn.append(attn_i)
        local_attn = torch.cat(local_attn, dim=2).view(b, self.num_heads, -1, head_dim)

        # Combine all attention outputs
        combined_attn = torch.cat([global_attn, global_local_attn, local_attn], dim=2)
        output = self._calculate_output(combined_attn, v, b, t, d)
        return self.out_proj(output)

class InfiniteAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int = 128,
        use_rotary: bool = True,
        **kwargs
    ):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.kernel_size = kernel_size
        self.use_rotary = use_rotary
        self.register_buffer("fourier_basis", self._init_fourier_basis(embed_dim))

    def _init_fourier_basis(self, embed_dim):
        # Initialize Fourier features for positional encoding
        freqs = torch.randn(embed_dim // 2)
        return freqs

    def _positional_encodings(self, x: torch.Tensor, device: torch.device):
        """Generate positional encodings for arbitrary sequence length."""
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device).float()
        fourier_features = torch.einsum("d, s -> sd", self.fourier_basis, pos)
        pe = torch.cat([torch.sin(fourier_features), torch.cos(fourier_features)], dim=1)
        return pe.unsqueeze(0).expand(x.size(0), -1, -1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        b, t, d = query.size()
        # Add positional encodings
        pe = self._positional_encodings(query, query.device)
        query = query + pe
        key = key + pe

        # Split into chunks for kernel-based attention
        chunks = []
        for i in range(0, t, self.kernel_size):
            chunk = query[:, i:i + self.kernel_size]
            chunks.append(chunk)

        # Compute attention for each chunk
        attn_output = []
        for chunk in chunks:
            q, k, v = self._forward_qkv(chunk, key, value, b, chunk.size(1), d)
            # Use kernel approximation (e.g., Performer)
            attn = self._performer_attention(q, k, v)
            attn_output.append(attn)

        # Concatenate and apply output projection
        output = torch.cat(attn_output, dim=1)
        return self.out_proj(output)

    def _performer_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Performer kernel approximation (simplified)
        # TODO: Replace with preferred kernel method
        q = q / (q.shape[-1] ** 0.5)
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = torch.softmax(attn, dim=-1)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)