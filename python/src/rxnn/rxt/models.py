import torch
from torch import nn
from typing import TypedDict, Union
from huggingface_hub import PyTorchModelHubMixin
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.attention import init_attention
from ..transformers.layers import ReactiveTransformerLayer
from ..transformers.models import ReactiveTransformerBase, ReactiveTransformerEncoder, ReactiveTransformerDecoder
from ..transformers.ff import get_activation_layer
from ..memory.stm import ShortTermMemory
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
            use_flash_attention: bool = True,
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
            **kwargs
    ):
        super(RxTAlphaComponentBase, self).__init__(**kwargs)
        assert ff_activation in ['relu', 'gelu',
                                 'swish', 'silu', 'linear',
                                 'sigmoid'], 'Feed-forward activation could be "relu", "gelu", "swish", "silu", "linear", "sigmoid".'
        assert self_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma', 'sqa'], 'Self-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'
        assert cross_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma', 'sqa'], 'Memory cross-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

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
                                                           max_seq_len=seq_len, is_causal=is_causal, num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=att_query_groups)

        if cross_att_type in ['mha', 'gqa', 'mqa']:
            cross_att_init = lambda: init_attention(embed_dim, att_heads, cross_att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=is_causal)
        else:
            cross_att_init = lambda: init_experimental_attention(embed_dim, att_heads, cross_att_type, cross_att_groups or att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=is_causal, num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=cross_att_query_groups or att_query_groups)

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
        self.model = self._init_model(stm, layers, embedding, use_flash_attention, embed_dim, vocab_size)

    def _init_model(self, stm: ShortTermMemory, layers: nn.ModuleList, embedding: nn.Embedding,
                    use_flash_attention: bool, embed_dim: int, vocab_size: int) -> ReactiveTransformerBase:
        pass

    def params_count(self):
        return get_model_size(self.model)

    def load_shared_embedding(self, embedding: nn.Embedding):
        self.model.embedding = embedding

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

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
            vocab_size: int
    ) -> ReactiveTransformerEncoder:
        return ReactiveTransformerEncoder(
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
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
            vocab_size: int
    ) -> ReactiveTransformerDecoder:
        return ReactiveTransformerDecoder(
            embed_dim,
            vocab_size,
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask)


def build_rxt_alpha_for_pretraining(
        encoder_config: RxTAlphaComponentConfig,
        decoder_config: RxTAlphaComponentConfig,
) -> tuple[RxTAlphaEncoder, RxTAlphaDecoder]:
    encoder = RxTAlphaEncoder(**encoder_config)
    decoder = RxTAlphaDecoder(**decoder_config)

    encoder.load_shared_memory(decoder.model.stm)
    encoder.load_shared_embedding(decoder.model.embedding)

    return encoder, decoder

