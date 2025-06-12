import torch
import torch.nn as nn
from .stm import ShortTermMemory

class StmMemoryAttention(nn.Module):
    def __init__(
            self,
            stm: ShortTermMemory,
            attention_layers: nn.ModuleList,
            memory_norm_layers: nn.ModuleList,
            use_gated_residual: bool = False,
            per_slot_gate: bool = False,
            init_gate: float = 0.0,
            use_dynamic_gate: bool = False,
            *args,
            **kwargs
    ):
        super(StmMemoryAttention, self).__init__(*args, **kwargs)
        self.stm = stm
        self.attention_layers = attention_layers
        self.memory_norm_layers = memory_norm_layers
        assert len(self.attention_layers) == len(self.memory_norm_layers) == self.stm.memory.size(0)
        self.num_layers = len(attention_layers)
        self.use_gated_residual = use_gated_residual
        self.per_slot_gate = per_slot_gate
        self.use_dynamic_gate = use_dynamic_gate
        if self.use_gated_residual:
            gate_shape = (self.num_layers, self.stm.stm_size, 1) if self.per_slot_gate else (self.num_layers,)
            self.gate = nn.Parameter(torch.full(gate_shape, init_gate))

    def update_max_len(self, max_seq_len: int):
        for i in range(self.num_layers):
            if self.attention_layers[i].rope is not None:
                self.attention_layers[i].rope.update_max_len(max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        mask = attention_mask.unsqueeze(1).unsqueeze(1).bool() if attention_mask is not None else None

        new_stm = torch.zeros_like(self.stm.memory)
        for i in range(self.num_layers):
            layer_stm = self.stm(i)
            # expand layer STM to batch size, if it's not in batch mode
            if layer_stm.size(0) == 1:
                layer_stm = layer_stm.expand(x.size(0), -1, -1)
            encoded_layer_data = x[i]
            normalized_layer_stm = self.memory_norm_layers[i](layer_stm)
            new_layer_stm = self.attention_layers[i](normalized_layer_stm, encoded_layer_data, encoded_layer_data, mask=mask)
            if self.use_gated_residual:
                # gated residual
                gate_input = self.gate[i] * (new_layer_stm + layer_stm) if self.use_dynamic_gate else self.gate[i]
                layer_gate = torch.sigmoid(gate_input)
                new_stm[i] = layer_gate * new_layer_stm + (1 - layer_gate) * layer_stm
            else:
                new_stm[i] = new_layer_stm + layer_stm # residual
        self.stm.update_all(new_stm)
        return self.stm.memory

