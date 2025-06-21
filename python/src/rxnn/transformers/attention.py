import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import math
from .positional import RotaryPositionalEmbedding, RelativePositionalEmbedding


class MultiHeadAttention(nn.Module):
    """Custom, extendable Multi-head attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            rope_only_for_keys: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = True,
            is_causal: bool = False,
            use_bias: bool = False,
            *args,
            **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.use_flash_attention = use_flash_attention
        self.is_causal = is_causal
        self.use_bias = use_bias
        if use_relative_embeddings:
            self.use_flash_attention = False
            self.rel_embed = RelativePositionalEmbedding(max_seq_len, embed_dim // num_heads)
            self.rope = None
            self.rope_only_for_query = False
            self.rope_only_for_keys = False
        else:
            self.rel_embed = None
            self.rope = rope
            self.rope_only_for_query = rope_only_for_query
            self.rope_only_for_keys = rope_only_for_keys
        self.dropout = nn.Dropout(dropout)
        self._init_q(embed_dim)
        self._init_kv(embed_dim)
        self._init_out(embed_dim)

    def _init_q(self, embed_dim: int):
        """Initialize query projection"""
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)

    def _init_kv(self, embed_dim: int):
        """Initialize key and value projections"""
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)

    def _init_out(self, embed_dim: int):
        """Initialize output projection"""
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int):
        """Forward pass through query, key, and value projections, and split the results into heads"""
        q = self.q_proj(query).view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
        k = self.k_proj(key).view(b, -1, self.num_heads, d // self.num_heads).transpose(1, 2)
        v = self.v_proj(value).view(b, -1, self.num_heads, d // self.num_heads).transpose(1, 2)
        return q, k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, separate: bool = False):
        if self.rope is not None:
            if self.rope_only_for_query:
                q = self.rope.forward_one(q)
            elif self.rope_only_for_keys:
                k = self.rope.forward_one(k)
            elif separate:
                q, k = self.rope.forward_one(q), self.rope.forward_one(k)
            else:
                q, k = self.rope(q, k)
        return q, k

    def _calculate_attn_weights(self, q: torch.Tensor, k: torch.Tensor, d: int, mask: torch.Tensor = None):
        """Calculate attention weights using scaled dot-product attention"""
        q, k = self._apply_rope(q, k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d // self.num_heads) ** 0.5
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_logits, dim=-1)

    def _calculate_attn_weight_with_relative_embeddings(self, q: torch.Tensor, k: torch.Tensor,
                                                        mask: torch.Tensor = None):
        """Calculate attention weights using scaled dot-product attention and apply relative embedding"""
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        rel_pos_bias = self.rel_embed(q, k)
        attn_logits += rel_pos_bias
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_logits, dim=-1)

    def _transpose_output(self, attn_output: torch.Tensor, b: int, t: int, d: int):
        """Transpose attention output back to (B, T, D) shape"""
        return attn_output.transpose(1, 2).contiguous().view(b, t, d)

    def _calculate_output(self, attn_weights: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int):
        """Calculate the output by multiplying attention weights with values and concatenating heads"""
        return self._transpose_output(torch.matmul(attn_weights, v), b, t, d)

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int,
                         mask: torch.Tensor = None, enable_gqa: bool = False):
        # After ~6h of fighthing, PyTorch based is still now working so I decided to use FlashAttention directly
        # with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #     return self._torch_attention(q, k, v, b, t, d, mask=mask, enable_gqa=enable_gqa)
        from flash_attn import flash_attn_func
        attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, causal=self.is_causal)
        return self._transpose_output(attn_output, b, t, d)

    def _torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int,
                         mask: torch.Tensor = None, enable_gqa: bool = False):
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask if not self.is_causal else None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
            enable_gqa=enable_gqa,
        )
        return self._transpose_output(attn_output, b, t, d)

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None):
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask)

    def _calculate_attention_with_relative_embedding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None):
        attn_weights = self._calculate_attn_weight_with_relative_embeddings(q, k, mask=mask)
        attn_weights = self.dropout(attn_weights)
        return self._calculate_output(attn_weights, v, b, t, d)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        b, t, d = query.size()
        q, k, v = self._forward_qkv(query, key, value, b, t, d)
        if not self.rel_embed:
            q, k = self._apply_rope(q, k)
            attn_output = self._calculate_attention(q, k, v, b, t, d, mask=mask)
        else:
            attn_output = self._calculate_attention_with_relative_embedding(q, k, v, b, t, d, mask=mask)
        return self.out_proj(attn_output)


class GroupedQueryAttention(MultiHeadAttention):
    """Custom Grouped Query attention layer, with RoPE support"""

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
            *args,
            **kwargs,
    ):
        self.num_groups = num_groups
        super(GroupedQueryAttention, self).__init__(
            embed_dim,
            num_heads,
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
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

    def _init_kv(self, embed_dim: int):
        self.k_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int):
        """Override query, key, and value projections for GQA case - split data into heads and groups"""
        head_dim = d // self.num_heads
        if not self.rel_embed:
            q = self.q_proj(query).view(b, t, self.num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(key).view(b, -1, self.num_groups, head_dim).transpose(1, 2)
            v = self.v_proj(value).view(b, -1, self.num_groups, head_dim).transpose(1, 2)
        else:
            # Relative embedding version is not working without this strange mapping - it will be removed in next versions
            group_heads = self.num_heads // self.num_groups

            # Process Q
            q = self.q_proj(query).view(b, t, self.num_groups, group_heads, head_dim).permute(0, 2, 3, 1,
                                                                                              4)  # (B, G, group_heads, T, head_dim)

            # Process K and V
            k = self.k_proj(key).view(b, -1, self.num_groups, head_dim).transpose(1, 2)  # (B, G, S, head_dim)
            v = self.v_proj(value).view(b, -1, self.num_groups, head_dim).transpose(1, 2)  # (B, G, S, head_dim)

            # Expand and flatten to 4D tensors
            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            q = q.flatten(start_dim=1, end_dim=2)  # (B, H, T, head_dim)
            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
        return q, k, v

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None):
        is_gqa = self.num_heads != self.num_groups
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa)


class MultiQueryAttention(MultiHeadAttention):
    """Custom Multi Query attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            *args,
            **kwargs,
    ):
        super(MultiQueryAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs
        )

    def _init_kv(self, embed_dim: int):
        """Override key/value initialization for MQA case"""
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.num_heads, bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.num_heads, bias=self.use_bias)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int):
        """Override query, key, and value projections for GQA case - use multiple heads
        for query and single for key/values"""
        if not self.rel_embed:
            q = self.q_proj(query).view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
            k = self.k_proj(key).view(b, -1, 1, d // self.num_heads).transpose(1, 2)
            v = self.v_proj(value).view(b, -1, 1, d // self.num_heads).transpose(1, 2)
        else:
            q = self.q_proj(query).view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
            k = self.k_proj(key).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            v = self.v_proj(value).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        return q, k, v

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None):
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=True)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=True)


def init_attention(
        embed_dim: int,
        num_heads: int,
        attention_type: str,
        gqa_groups: int = 1,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding = None,
        rope_only_for_query: bool = False,
        rope_only_for_keys: bool = False,
        use_relative_embeddings: bool = False,
        max_seq_len: int = 1024,
        use_flash_attention: bool = False,
        is_causal: bool = False,
        use_bias: bool = False,
) -> MultiHeadAttention:
    assert attention_type == 'mha' or attention_type == 'gqa' or attention_type == 'mqa', \
        "Error, attention type should be one of: 'mha', 'gqa', 'mqa'"

    if attention_type == "gqa":
        return GroupedQueryAttention(
            embed_dim,
            num_heads,
            gqa_groups,
            dropout=dropout,
            rope=rope,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
        )
    elif attention_type == "mqa":
        return MultiQueryAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
        )
    else:
        return MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
        )
