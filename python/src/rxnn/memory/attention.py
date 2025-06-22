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
            use_tanh_gate: bool = False,
            debug_mode: bool = False,
            debug_interval: int = 10,
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
        self.use_tanh_gate = use_tanh_gate
        if self.use_gated_residual:
            gate_shape = (self.num_layers, self.stm.stm_size, 1) if self.per_slot_gate else (self.num_layers,)
            self.gate = nn.Parameter(torch.full(gate_shape, init_gate))

        self.debug_mode = debug_mode
        self.debug_interval = debug_interval
        self.debug_step = 0

    def update_max_len(self, max_seq_len: int):
        for i in range(self.num_layers):
            if self.attention_layers[i].rope is not None:
                self.attention_layers[i].rope.update_max_len(max_seq_len)

    def _residual_gate(self, gate: torch.Tensor, layer_stm: torch.Tensor, new_layer_stm: torch.Tensor) -> torch.Tensor:
        if self.use_dynamic_gate:
            mean_dim = -1 if self.per_slot_gate else [1, 2]
            gate_input = gate * (new_layer_stm + layer_stm).mean(dim=mean_dim, keepdim=True)
            layer_gate = torch.tanh(gate_input) if self.use_tanh_gate else torch.sigmoid(gate_input)
        else:
            layer_gate = torch.tanh(gate) if self.use_tanh_gate else torch.sigmoid(gate)
        if self.use_tanh_gate:
            return (1 + layer_gate) * new_layer_stm + (1 - layer_gate) * layer_stm
        else:
            return layer_gate * new_layer_stm + (1 - layer_gate) * layer_stm

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        new_stm = torch.zeros_like(self.stm.memory)
        for i in range(self.num_layers):
            layer_stm = self.stm(i)
            # expand layer STM to batch size, if it's not in batch mode
            if layer_stm.size(0) == 1:
                layer_stm = layer_stm.expand(x.size(0), -1, -1)
            encoded_layer_data = x[i]
            normalized_layer_stm = self.memory_norm_layers[i](layer_stm)

            if self.debug_mode and self.training:
                if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                    self.debug_step = 0
                    print(f"Normalized STM stats - mean: {normalized_layer_stm.mean().item():.4f}, std: {normalized_layer_stm.std().item():.4f}")
                else:
                    self.debug_step += 1

            new_layer_stm = self.attention_layers[i](normalized_layer_stm, encoded_layer_data, encoded_layer_data, mask=attention_mask)
            if self.use_gated_residual:
                new_stm[i] = self._residual_gate(self.gate[i], layer_stm, new_layer_stm) # gated residual
            else:
                new_stm[i] = new_layer_stm + layer_stm # residual
        self.stm.update_all(new_stm)
        return self.stm.memory
