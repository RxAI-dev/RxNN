import torch
import torch.nn as nn
from enum import Enum
from typing import Literal
from huggingface_hub import PyTorchModelHubMixin
from ..transformers.models import ReactiveTransformerEncoder, ReactiveTransformerDecoder

class MLMHead(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    def __init__(self, embed_dim: int, vocab_size: int, *args, **kwargs):
        super(MLMHead, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class MLMTrainingModel(nn.Module):
    def __init__(
            self,
            encoder: ReactiveTransformerEncoder,
            mlm_head: MLMHead,
            *args,
            **kwargs
    ):
        super(MLMTrainingModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.mlm_head = mlm_head

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        h, _ = self.encoder(x, attention_mask=attention_mask)
        y = self.mlm_head(h)
        return y

class JointTrainingModel(nn.Module):
    def __init__(
            self,
            encoder: ReactiveTransformerEncoder,
            decoder: ReactiveTransformerDecoder,
            mlm_head: MLMHead,
            *args,
            **kwargs
    ):
        super(JointTrainingModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.mlm_head = mlm_head
        self.decoder = decoder

    def forward(self, x_e: torch.Tensor, x_d: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor]:
        encoder_result, _ = self.encoder(x_e, attention_mask=attention_mask)
        y_e = self.mlm_head(encoder_result)
        y_d = self.decoder(x_d, attention_mask=attention_mask)
        return y_e, y_d

class MrlActorAction(Enum):
    DECODE = 1
    UPDATE = 2

class MrlActorModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            memory_attention: nn.Module,
            **kwargs
    ):
        super(MrlActorModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.memory_attention = memory_attention

    def freeze_components(self, stage: Literal['update', 'fetch', 'both'] = 'both'):
        """Freeze encoder/decoder except memory-related layers."""
        if self.encoder.freeze_without_memory is not None:
            self.encoder.freeze_without_memory()
            if stage == 'update':
                self.encoder.freeze_memory()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.model.trainable_cross_attention_(True if stage != 'update' else False)
        if self.decoder.freeze_without_memory is not None:
            self.decoder.freeze_without_memory()
            if stage == 'update':
                self.decoder.freeze_memory()
        else:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.model.trainable_cross_attention_(True if stage != 'update' else False)
        # Unfreeze memory attention
        for param in self.memory_attention.parameters():
            param.requires_grad = True if stage != 'fetch' else False

    def unfreeze_components(self):
        """Unfreeze all components after initial training."""
        if self.encoder.unfreeze_all is not None:
            self.encoder.unfreeze_all()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
        if self.decoder.unfreeze_all is not None:
            self.decoder.unfreeze_all()
        else:
            for param in self.decoder.parameters():
                param.requires_grad = True
        for param in self.memory_attention.parameters():
            param.requires_grad = True

    def reset_memory(self):
        self.memory_attention.reset_memory()

    def unique_parameters(self):
        return list(set(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.memory_attention.parameters())
        ))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, action: MrlActorAction = MrlActorAction.DECODE) -> torch.Tensor:
        if action == MrlActorAction.DECODE:
            return self.decoder(x, attention_mask=attention_mask)
        else:
            _, ed = self.encoder(x, attention_mask=attention_mask)
            return self.memory_attention(ed, attention_mask=attention_mask)

class MrlCriticModel(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, **kwargs):
        super(MrlCriticModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x, _ = self.encoder(x, attention_mask=attention_mask)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        return self.value_head(x)