import torch
import torch.nn as nn
from typing import TypeAlias, Literal

ResidualGateType: TypeAlias = Literal['static', 'elementwise', 'linear']


class ResidualGate(nn.Module):
    def __init__(
            self,
            stm_size: int,
            use_gate: bool = False,
            gate_type: ResidualGateType = 'static',
            per_slot_gate: bool = True,
            init_gate: float = 0.0,
            use_tanh_gate: bool = True,
            **kwargs,
    ):
        super(ResidualGate, self).__init__(**kwargs)
        self.use_gate = use_gate
        self.per_slot_gate = per_slot_gate
        self.gate_type = gate_type
        self.use_tanh_gate = use_tanh_gate

        if self.use_gate:
            if self.gate_type == 'linear':
                self.gate = nn.Linear(stm_size, stm_size if self.per_slot_gate else 1)
            else:
                gate_shape = (stm_size, 1) if self.per_slot_gate else (1,)
                self.gate = nn.Parameter(torch.full(gate_shape, init_gate))
        else:
            self.gate = None

        self.gate_activation = nn.Tanh() if self.use_tanh_gate else nn.Sigmoid()

    def _dynamic_gate(self, old_value: torch.Tensor, new_value: torch.Tensor):
        if self.gate_type == 'linear':
            mean_residual = (new_value + old_value).mean(dim=-1)
            gate_input = self.gate(mean_residual).unsqueeze(-1)
        else:
            mean_dim = -1 if self.per_slot_gate else [1, 2]
            gate_input = self.gate * (new_value + old_value).mean(dim=mean_dim, keepdim=True)
        return self.gate_activation(gate_input)

    def _calculate_output(self, layer_gate: torch.Tensor, old_value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        if self.use_tanh_gate:
            return (1 + layer_gate) * new_value + (1 - layer_gate) * old_value
        else:
            return layer_gate * new_value + (1 - layer_gate) * old_value

    def forward(self, old_value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        if not self.use_gate:
            return new_value + old_value

        if self.gate_type == 'static':
            layer_gate = self.gate_activation(self.gate)
        else:
            layer_gate = self._dynamic_gate(old_value, new_value)

        return self._calculate_output(layer_gate, old_value, new_value)
