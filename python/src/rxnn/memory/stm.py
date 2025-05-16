import torch
import torch.nn as nn

class ShortTermMemory(nn.Module):
    """Short-term memory module for the Attention-based Memory System"""

    def __init__(self, num_layers: int, embed_dim: int, stm_size: int, init_type: str = 'normal',
                 is_trainable: bool = False, *args, **kwargs):
        super(ShortTermMemory, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.stm_size = stm_size
        self.is_trainable = is_trainable
        assert init_type in ['normal', 'standard', 'uniform', 'ones', 'zeros'], \
            'STM init type must be one of "normal", "standard", "uniform", "ones", "zeros"'
        self.init_type = init_type
        stm = self._init_tensor()
        if self.is_trainable:
            self.memory = nn.Parameter(stm)
        else:
            self.register_buffer('memory', stm)

    def _init_tensor(self, init_type: str = None):
        init_type = init_type or self.init_type
        if init_type == 'normal':
            return torch.normal(0, 0.02, (self.num_layers, self.stm_size, self.embed_dim))
        elif init_type == 'standard':
            return torch.normal(0, 1, (self.num_layers, self.stm_size, self.embed_dim))
        elif init_type == 'uniform':
            return torch.rand(self.num_layers, self.stm_size, self.embed_dim) * 0.02
        elif init_type == 'ones':
            return torch.ones(self.num_layers, self.stm_size, self.embed_dim)
        else:
            return torch.zeros(self.num_layers, self.stm_size, self.embed_dim)

    def forward(self, layer: int) -> torch.Tensor:
        return self.memory[layer].unsqueeze(0)

    def update_layer(self, layer: int, new_stm: torch.Tensor):
        self.memory[layer] = new_stm

    def update_all(self, new_stm: torch.Tensor):
        self.memory.copy_(new_stm)

    def make_trainable(self):
        if not self.is_trainable:
            self.is_trainable = True
            initial_stm = self.memory.clone()
            del self.memory
            self.memory = nn.Parameter(initial_stm)

    def freeze(self):
        if self.is_trainable:
            self.requires_grad_(False)
            trained_stm = self.memory.clone()
            del self.memory
            self.register_buffer('memory', trained_stm)

    def reset(self, init_type: str = None):
        self.memory = self._init_tensor(init_type)

    def resize(self, new_stm_size: int, init_type: str = None):
        self.stm_size = new_stm_size
        self.memory = self._init_tensor(init_type)
