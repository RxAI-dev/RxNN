import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator

def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Apply top-p (nucleus) sampling
    if top_p is not None and 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices back to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample from distribution
    return torch.multinomial(probs, num_samples=1)

class Sampler:
    def __init__(self, model: nn.Module, device: torch.device, end_token_id: int):
        self.model = model.to(device)
        self.device = device
        self.end_token_id = end_token_id

    def _generate_token(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float ,
        attention_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        # Forward pass to get next token logits
        outputs = self.model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs[:, -1, :]  # Get logits for next token
        # Apply sampling
        next_token = sample(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        next_token = next_token.item()  # Extract scalar token
        next_token_ten = torch.tensor([[next_token]], device=self.device)
        next_input_ids = torch.cat([input_ids, next_token_ten], dim=1)
        new_one = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        next_mask = torch.cat([attention_mask, new_one], dim=1) if attention_mask is not None else None
        # Yield the generated token
        return (
            next_token,
            next_input_ids,
            next_mask
        )

    def __call__(
        self,
        initial_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        max_seq_len: int = 50,
        attention_mask: torch.Tensor = None,
        no_grad: bool = True,
    ) -> Iterator[int]:
        # Convert initial tokens to tensor and move to device
        input_ids = initial_tokens

        if no_grad:
            with torch.no_grad():
                for _ in range(max_seq_len):
                    next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p, attention_mask)
                    yield next_token
                    if next_token == self.end_token_id:
                        break
        else:
            for _ in range(max_seq_len):
                next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p, attention_mask)
                yield next_token
                if next_token == self.end_token_id:
                    break