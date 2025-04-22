import torch
from torch import nn
from rxnn.transformers.attention import MultiHeadAttention

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