import torch
import torch.nn as nn
from .positional import AbsolutePositionalEmbedding
from .mask import create_causal_mask
from ..memory.stm import ShortTermMemory


class ReactiveTransformerBase(nn.Module):
    """Base class for Reactive Transformer models - common logic for both decoders and encoders."""

    def __init__(
            self,
            stm: ShortTermMemory,
            embedding: nn.Embedding,
            own_layers: nn.ModuleList,
            shared_layers: nn.ModuleList = None,
            absolute_embedding: AbsolutePositionalEmbedding = None,
            use_flash_attention: bool = False,
            use_relative_embedding: bool = False,
            use_moe: bool = False,
            *args,
            **kwargs,
    ):
        super(ReactiveTransformerBase, self).__init__(*args, **kwargs)

        self.embedding = embedding
        self.stm = stm
        self.pos_embedding = absolute_embedding
        self.use_flash_attention = use_flash_attention
        self.use_relative_embedding = use_relative_embedding

        self.shared_layers = shared_layers
        self.layers = own_layers
        self.num_shared_layers = len(shared_layers) if shared_layers else 0
        self.num_own_layers = len(own_layers) if own_layers else 0
        self.use_moe = use_moe

    def trainable_cross_attention_(self, is_trainable: bool, with_norms: bool = True):
        for i in range(self.num_shared_layers):
            self.shared_layers[i].trainable_cross_attention_(is_trainable, with_norms)
        for i in range(self.num_own_layers):
            self.layers[i].trainable_cross_attention_(is_trainable, with_norms)

    def memory_parameters(self) -> list[nn.Parameter]:
        own = [param for layer in self.layers for param in layer.memory_parameters()]
        shared = [param for layer in self.shared_layers for param in layer.memory_parameters()] if self.shared_layers else []
        return own + shared

    def not_memory_parameters(self) -> list[nn.Parameter]:
        own = [param for layer in self.layers for param in layer.not_memory_parameters()]
        shared = [param for layer in self.shared_layers for param in layer.not_memory_parameters()] if self.shared_layers else []
        return own + shared

    def moe_router_loss(self):
        if self.use_moe:
            return torch.stack([self.layers[i].moe_router_loss() for i in range(self.num_own_layers) if self.layers[i].use_moe or self.layers[i].use_moe_att] + [
                self.shared_layers[i].moe_router_loss() for i in range(self.num_shared_layers) if self.shared_layers[i].use_moe or self.shared_layers[i].use_moe_att]).mean()
        else:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared logic for encoders and decoders - apply embeddings and positional encoding
        x = self.embedding(x)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        return x


class ReactiveTransformerDecoder(ReactiveTransformerBase):
    """Reactive Transformer decoder - extending the classic Transformer decoder with Memory Cross-Attention"""

    def __init__(self, embed_dim: int, vocab_size: int, *args, **kwargs):
        super(ReactiveTransformerDecoder, self).__init__(*args, **kwargs)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = super().forward(x)  # apply embeddings
        seq_len = x.size(1)
        if not self.use_flash_attention and self.use_relative_embedding:
            mask = create_causal_mask(seq_len, device=x.device)
            if attention_mask is not None:
                mask &= attention_mask.unsqueeze(1).unsqueeze(1).bool()
        elif attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        else:
            mask = None
        # Process shared layers
        if self.shared_layers is not None:
            for i in range(self.num_shared_layers):
                layer_stm = self.stm(i)
                # expand layer STM to batch size, if it's not in batch mode
                if layer_stm.size(0) == 1:
                    layer_stm = layer_stm.expand(x.size(0), -1, -1)
                x = self.shared_layers[i](x, layer_stm, mask=mask)
        # Process own layers
        for i in range(self.num_own_layers):
            layer_stm = self.stm(i)
            # expand layer STM to batch size, if it's not in batch mode
            if layer_stm.size(0) == 1:
                layer_stm = layer_stm.expand(x.size(0), -1, -1)
            x = self.layers[i](x, layer_stm, mask=mask)
        return self.head(x)


class ReactiveTransformerEncoder(ReactiveTransformerBase):
    """Reactive Transformer encoder - extending the classic Transformer encoder with Memory Cross-Attention"""

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)  # apply embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        hidden_states = []
        # Process shared layers
        if self.shared_layers is not None:
            for i in range(self.num_shared_layers):
                layer_stm = self.stm(i)
                # expand layer STM to batch size, if it's not in batch mode
                if layer_stm.size(0) == 1:
                    layer_stm = layer_stm.expand(x.size(0), -1, -1)
                x = self.shared_layers[i](x, layer_stm, mask=attention_mask)
                hidden_states.append(x)
        # Process own layers
        for i in range(self.num_own_layers):
            layer_stm = self.stm(i)
            # expand layer STM to batch size, if it's not in batch mode
            if layer_stm.size(0) == 1:
                layer_stm = layer_stm.expand(x.size(0), -1, -1)
            x = self.layers[i](x, layer_stm, mask=attention_mask)
            hidden_states.append(x)
        return x, torch.stack(hidden_states)


class ReactiveTransformerEncoderDetachStm(ReactiveTransformerBase):
    """
    Reactive Transformer encoder DetachStm version - reactive transformer encoder that's detaching Short-Term Memory tensors,
    before processing them in layers (memory cross-attention). Made for Memory-Aware Critic models, to not include memory
    update gradients in Critic optimization.
    """

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)  # apply embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        hidden_states = []
        # Process shared layers
        if self.shared_layers is not None:
            for i in range(self.num_shared_layers):
                layer_stm = self.stm(i).detach() # <- Detach STM layer
                # expand layer STM to batch size, if it's not in batch mode
                if layer_stm.size(0) == 1:
                    layer_stm = layer_stm.expand(x.size(0), -1, -1)
                x = self.shared_layers[i](x, layer_stm, mask=attention_mask)
                hidden_states.append(x)
        # Process own layers
        for i in range(self.num_own_layers):
            layer_stm = self.stm(i).detach() # <- Detach STM layer
            # expand layer STM to batch size, if it's not in batch mode
            if layer_stm.size(0) == 1:
                layer_stm = layer_stm.expand(x.size(0), -1, -1)
            x = self.layers[i](x, layer_stm, mask=attention_mask)
            hidden_states.append(x)
        return x, torch.stack(hidden_states)


class ClassicTransformerBase(nn.Module):
    """Base class for Classic Transformer models - common logic for both decoders and encoders."""

    def __init__(
            self,
            embedding: nn.Embedding,
            layers: nn.ModuleList,
            absolute_embedding: AbsolutePositionalEmbedding = None,
            use_flash_attention: bool = False,
            use_relative_embedding: bool = False,
            *args,
            **kwargs,
    ):
        super(ClassicTransformerBase, self).__init__(*args, **kwargs)

        self.embedding = embedding
        self.pos_embedding = absolute_embedding
        self.use_flash_attention = use_flash_attention
        self.use_relative_embedding = use_relative_embedding

        self.layers = layers
        self.num_layers = len(layers) if layers else 0

    def moe_router_loss(self):
        return torch.stack([self.layers[i].moe_router_loss() for i in range(self.num_layers) if self.layers[i].use_moe or self.layers[i].use_moe_att]).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared logic for encoders and decoders - apply embeddings and positional encoding
        x = self.embedding(x)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        return x


class ClassicTransformerDecoder(ClassicTransformerBase):
    """Classic Transformer decoder - for decoder-only Transformer models"""

    def __init__(self, embed_dim: int, vocab_size: int, *args, **kwargs):
        super(ClassicTransformerDecoder, self).__init__(*args, **kwargs)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = super().forward(x)  # apply embeddings
        seq_len = x.size(1)
        if not self.use_flash_attention and self.use_relative_embedding:
            mask = create_causal_mask(seq_len, device=x.device)
            if attention_mask is not None:
                mask &= attention_mask.unsqueeze(1).unsqueeze(1).bool()
        elif attention_mask is not None:
            print(attention_mask.size())
            mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        else:
            mask = None

        # Process layers
        for i in range(self.num_layers):
            x = self.layers[i](x, mask=mask)
        return self.head(x)


class ClassicTransformerEncoder(ClassicTransformerBase):
    """Classic Transformer encoder - for encoder-only Transformer models"""

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)  # apply embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        hidden_states = []
        # Process own layers
        for i in range(self.num_own_layers):
            x = self.layers[i](x, mask=attention_mask)
            hidden_states.append(x)
        return x, torch.stack(hidden_states)
