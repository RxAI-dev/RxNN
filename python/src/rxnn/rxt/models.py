import torch
from torch import nn
from typing import TypedDict, Union
from huggingface_hub import PyTorchModelHubMixin
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.attention import init_attention
from ..transformers.layers import ReactiveTransformerLayer
from ..transformers.models import ReactiveTransformerBase, ReactiveTransformerEncoder, ReactiveTransformerDecoder, ReactiveTransformerEncoderDetachStm
from ..transformers.ff import get_activation_layer
from ..memory.stm import ShortTermMemory
from ..memory.norm import init_memory_norm
from ..memory.attention import StmMemoryAttention, InterlayerStmMemoryAttention, SelfStmMemoryAttention
from ..memory.gate import ResidualGate, ResidualGateType, SlotStatusType
from ..utils import get_model_size
from ..experimental.attention import init_experimental_attention


class RxTAlphaComponentConfig(TypedDict):
    num_layers: int
    vocab_size: int
    embed_dim: int
    ff_dim: int
    att_heads: int
    seq_len: int
    stm_size: int
    use_flash_attention: bool
    use_gated: bool
    ff_activation: str
    ff_dropout: float
    att_dropout: float
    use_rms_norm: bool
    att_groups: int
    use_moe: bool
    num_experts: int
    moe_top_k: int
    self_att_type: str
    cross_att_type: str
    att_experts: int
    att_query_experts: int
    att_query_groups: int
    cross_att_groups: int
    cross_att_query_groups: int
    use_head_norm: bool
    init_identity_norm: bool


class RxTAlphaComponentBase(nn.Module, PyTorchModelHubMixin):
    """Base class for RxT-Alpha (Reactive Transformer) components (encoder and decoder)"""

    def __init__(
            self,
            is_causal: bool,
            num_layers: int = 12,
            vocab_size: int = 20000,
            embed_dim: int = 512,
            ff_dim: int = 1536,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            use_gated: bool = True,
            ff_activation: str = "swish",
            ff_dropout: float = 0.0,
            att_dropout: float = 0.0,
            use_rms_norm: bool = True,
            att_groups: int = 1,
            use_moe: bool = False,
            num_experts: int = 1,
            moe_top_k: int = 1,
            self_att_type: str = 'gqa',
            cross_att_type: str = 'mqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            cross_att_groups: int = None,
            cross_att_query_groups: int = None,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            **kwargs
    ):
        super(RxTAlphaComponentBase, self).__init__(**kwargs)
        assert ff_activation in ['relu', 'gelu',
                                 'swish', 'silu', 'linear',
                                 'sigmoid'], 'Feed-forward activation could be "relu", "gelu", "swish", "silu", "linear", "sigmoid".'
        assert self_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                                 'sqa'], 'Self-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'
        assert cross_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                                  'sqa'], 'Memory cross-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        embedding = nn.Embedding(vocab_size, embed_dim)
        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        ff_activation = get_activation_layer(ff_activation)

        if self_att_type in ['mha', 'gqa', 'mqa']:
            att_init = lambda: init_attention(embed_dim, att_heads, self_att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=is_causal)
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, self_att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=is_causal,
                                                           num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=att_query_groups)

        if cross_att_type in ['mha', 'gqa', 'mqa']:
            cross_att_init = lambda: init_attention(embed_dim, att_heads, cross_att_type, att_groups, rope=rope,
                                                    use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                    max_seq_len=seq_len, is_causal=False, rope_only_for_query=True)
        else:
            cross_att_init = lambda: init_experimental_attention(embed_dim, att_heads, cross_att_type,
                                                                 cross_att_groups or att_groups, rope=rope,
                                                                 use_flash_attention=use_flash_attention,
                                                                 dropout=att_dropout,
                                                                 max_seq_len=seq_len, is_causal=False,
                                                                 num_experts=att_experts,
                                                                 num_query_experts=att_query_experts,
                                                                 num_query_groups=cross_att_query_groups or att_query_groups,
                                                                 rope_only_for_query=True)

        layers = nn.ModuleList([
            ReactiveTransformerLayer(
                embed_dim,
                ff_dim,
                use_gated=use_gated,
                use_moe=use_moe,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                ff_activation=ff_activation,
                ff_dropout=ff_dropout,
                use_rms_norm=use_rms_norm,
                self_attention=att_init(),
                memory_cross_attention=cross_att_init(),
            ) for _ in range(num_layers)
        ])
        self.model = self._init_model(
            stm, layers, embedding, use_flash_attention, embed_dim, vocab_size, use_moe,
            use_head_norm=use_head_norm, init_identity_norm=init_identity_norm,
        )

    def _init_model(self, stm: ShortTermMemory, layers: nn.ModuleList, embedding: nn.Embedding,
                    use_flash_attention: bool, embed_dim: int, vocab_size: int, use_moe: bool,
                    use_head_norm: bool = False, init_identity_norm: bool = False) -> ReactiveTransformerBase:
        pass

    def params_count(self):
        return get_model_size(self.model)

    def load_shared_embedding(self, embedding: nn.Embedding):
        self.model.embedding = embedding

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def memory_parameters(self) -> list[nn.Parameter]:
        return self.model.memory_parameters()

    def not_memory_parameters(self) -> list[nn.Parameter]:
        return self.model.not_memory_parameters()

    def freeze_without_memory(self, unfreeze_norms: bool = True):
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.trainable_cross_attention_(True, with_norms=unfreeze_norms)

    def freeze_memory(self, with_norms: bool = True):
        self.model.trainable_cross_attention_(False, with_norms=with_norms)

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad_(True)

    def update_max_len(self, max_seq_len: int):
        for layer in self.model.layers:
            layer.update_max_len(max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.model(x, attention_mask=attention_mask)


class RxTAlphaEncoder(RxTAlphaComponentBase, pipeline_tag="fill-mask", license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) encoder model"""

    def __init__(self, **kwargs: RxTAlphaComponentConfig):
        super(RxTAlphaEncoder, self).__init__(False, **kwargs)

    def _init_model(
            self,
            stm: ShortTermMemory,
            layers: nn.ModuleList,
            embedding: nn.Embedding,
            use_flash_attention: bool,
            embed_dim: int,
            vocab_size: int,
            use_moe: bool,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
    ) -> ReactiveTransformerEncoder:
        return ReactiveTransformerEncoder(
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
            use_moe=use_moe,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask)


class RxTAlphaDecoder(RxTAlphaComponentBase, pipeline_tag="text-generation", license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) decoder model"""

    def __init__(self, **kwargs):
        super(RxTAlphaDecoder, self).__init__(True, **kwargs)

    def _init_model(
            self, stm: ShortTermMemory,
            layers: nn.ModuleList,
            embedding: nn.Embedding,
            use_flash_attention: bool,
            embed_dim: int,
            vocab_size: int,
            use_moe: bool,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
    ) -> ReactiveTransformerDecoder:
        return ReactiveTransformerDecoder(
            embed_dim,
            vocab_size,
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
            use_moe=use_moe,
            use_head_norm=use_head_norm,
            init_identity_norm=init_identity_norm,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None, use_self_attn_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache)


class RxTAlphaMemoryAttention(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) memory attention model"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            disable_residual: bool = False,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTAlphaMemoryAttention, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa']:
            att_init = lambda: init_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True)
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=False,
                                                           num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=att_query_groups, rope_only_for_keys=True)

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
                disable_residual=disable_residual,
            ) for _ in range(num_layers)
        ])

        self.model = StmMemoryAttention(
            stm, attention_layers, memory_norm_layers,
            memory_input_norm_layers, residual_gates,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)


class RxTAlphaInterlayerMemoryAttention(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) memory attention model with interlayer STM attention"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            interlayer_att_dropout: float = 0.0,
            interlayer_att_groups: int = 1,
            interlayer_att_type: str = 'sqa',
            interlayer_att_experts: int = None,
            interlayer_att_query_experts: int = None,
            interlayer_att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTAlphaInterlayerMemoryAttention, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa']:
            att_init = lambda: init_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True
            )
        else:
            att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, num_experts=att_experts,
                num_query_experts=att_query_experts, num_query_groups=att_query_groups,
                rope_only_for_keys=True
            )

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        # Interlayer attention
        if interlayer_att_type in ['mha', 'gqa', 'mqa']:
            interlayer_att_init = lambda: init_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False
            )
        else:
            interlayer_att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False,
                num_experts=interlayer_att_experts, num_query_experts=interlayer_att_query_experts, num_query_groups=interlayer_att_query_groups
            )

        mean_attention_layers = nn.ModuleList([interlayer_att_init() for _ in range(num_layers)])

        mean_stm_norm = init_memory_norm(
            norm_type, embed_dim, stm_size, decay=norm_decay,
            init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale
        )

        mean_memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])

        mean_residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        self.model = InterlayerStmMemoryAttention(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gates,
            mean_attention_layers, mean_memory_norm_layers, mean_residual_gates, mean_stm_norm,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)

class RxTAlphaSelfMemoryAttention(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) memory attention model with STM layer self-attention"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            self_att_dropout: float = 0.0,
            self_att_groups: int = 1,
            self_att_type: str = 'sqa',
            self_att_experts: int = None,
            self_att_query_experts: int = None,
            self_att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            use_gate_for_self_attention: bool = False,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTAlphaSelfMemoryAttention, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa']:
            att_init = lambda: init_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True
            )
        else:
            att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, num_experts=att_experts,
                num_query_experts=att_query_experts, num_query_groups=att_query_groups,
                rope_only_for_keys=True
            )

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        # Self attention
        if self_att_type in ['mha', 'gqa', 'mqa']:
            self_att_init = lambda: init_attention(
                embed_dim, att_heads, self_att_type, self_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=self_att_dropout, is_causal=False
            )
        else:
            self_att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, self_att_type, self_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=self_att_dropout, is_causal=False,
                num_experts=self_att_experts, num_query_experts=self_att_query_experts, num_query_groups=self_att_query_groups
            )

        self_attention_layers = nn.ModuleList([self_att_init() for _ in range(num_layers)])

        self_memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])

        self_residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gate_for_self_attention, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        self.model = SelfStmMemoryAttention(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gates,
            self_attention_layers, self_memory_norm_layers, self_residual_gates,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)


class RxTAlphaCriticEncoder(RxTAlphaComponentBase, pipeline_tag="text-classification", license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) encoder model"""

    def __init__(self, **kwargs: RxTAlphaComponentConfig):
        super(RxTAlphaCriticEncoder, self).__init__(False, **kwargs)

    def _init_model(
            self,
            stm: ShortTermMemory,
            layers: nn.ModuleList,
            embedding: nn.Embedding,
            use_flash_attention: bool,
            embed_dim: int,
            vocab_size: int,
            use_moe: bool = False,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
    ) -> ReactiveTransformerEncoderDetachStm:
        return ReactiveTransformerEncoderDetachStm(
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
            use_moe=use_moe,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask)

