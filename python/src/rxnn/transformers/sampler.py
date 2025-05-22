import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, Union, Optional
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer


def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
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
            top_p: float,
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
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 50,
            attention_mask: torch.Tensor = None,
            no_grad: bool = True,
    ) -> Iterator[int]:
        # Convert initial tokens to tensor and move to device
        input_ids = initial_tokens

        if no_grad:
            with torch.no_grad():
                for _ in range(max_seq_len):
                    next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p,
                                                                                 attention_mask)
                    yield next_token
                    if next_token == self.end_token_id:
                        break
        else:
            for _ in range(max_seq_len):
                next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p,
                                                                             attention_mask)
                yield next_token
                if next_token == self.end_token_id:
                    break


class SampleDecoder:
    def __init__(self, sampler: Sampler, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = self.sampler.device

    def tokenize_input(self, text: str, max_seq_len: int = 256):
        tokenized = self.tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_attention_mask=True
        )
        tokenized['input_ids'] = tokenized['input_ids'][:, :-1].to(self.device)
        tokenized['attention_mask'] = tokenized['attention_mask'][:, :-1].to(self.device)
        del tokenized['token_type_ids']
        return tokenized

    def ids_iter(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        tokenized = self.tokenize_input(text, max_seq_len=max_seq_len)
        return self.sampler(
            tokenized['input_ids'],
            temperature=temperature,
            top_p=top_p,
            max_seq_len=max_seq_len,
            attention_mask=tokenized['attention_mask']
        )

    def txt_iter(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        return map(
            lambda x: self.tokenizer.decode([x]).replace('Ċ', '\n').replace('Ġ', ' '),
            self.ids_iter(text, temperature, top_p, max_seq_len)
        )

    def txt(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        return text + ''.join(self.txt_iter(text, temperature, top_p, max_seq_len))

    def print_stream(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        print(text, end='')
        resp = text
        for txt_token in self.txt_iter(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len):
            print(txt_token, end='')
            resp += txt_token
        return resp

    def __call__(self, text: str, print_stream: bool = False, temperature: float = 0.1, top_p: float = 0.9,
                 max_seq_len=256):
        if print_stream:
            return self.print_stream(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len)
        else:
            return self.txt(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len)

class InteractionSampler(SampleDecoder):
    def __init__(self, sampler: Sampler, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        super(InteractionSampler, self).__init__(sampler, tokenizer)

    def txt(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len: int = 256, special_token_spaces: bool = True):
        txt = '[Q]' + text + '[A]'
        start_txt = '[Q] ' + text + ' [A] ' if special_token_spaces else txt
        return start_txt + ''.join(self.txt_iter(txt, temperature, top_p, max_seq_len))

    def print_stream(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len: int = 256, special_token_spaces: bool = True):
        txt = '[Q]' + text + '[A]'
        start_txt = '[Q] ' + text + ' [A] ' if special_token_spaces else txt
        print(start_txt, end='')
        resp = start_txt
        for txt_token in self.txt_iter(txt, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len):
            print(txt_token, end='')
            resp += txt_token
        return resp

    def __call__(self, text: str, print_stream: bool = False, temperature: float = 0.1, top_p: float = 0.9,
                 max_seq_len: int = 256, special_token_spaces: bool = True):
        if print_stream:
            return self.print_stream(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, special_token_spaces=special_token_spaces)
        else:
            return self.txt(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, special_token_spaces=special_token_spaces)


def sample_batch(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sampled_tokens, log_probs)"""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    # Apply temperature scaling first
    logits = logits / temperature

    # Store original dtype for precision
    # original_dtype = logits.dtype

    # Work in float32 for numerical stability
    # logits = logits.float()

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_logits[..., -1, None]
        logits = torch.where(logits < min_top_k, torch.tensor(float('-inf'), device=logits.device), logits)

    # Apply top-p (nucleus) sampling
    if top_p is not None and 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create mask to remove tokens above cumulative probability
        sorted_mask = cumulative_probs <= top_p
        sorted_mask[..., 0] = 1  # Ensure at least one token

        # Scatter sorted mask back to original indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=logits.device))

    # Compute log probabilities once
    log_probs = F.log_softmax(logits, dim=-1)

    # Convert back to original dtype for sampling
    # log_probs = log_probs.to(original_dtype)

    # Calculate probabilities using stable exponentiation
    probs = torch.exp(log_probs)

    # Sample from distribution
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Gather log probabilities for the chosen tokens
    selected_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    return next_tokens, selected_log_probs


class BatchSampler:
    def __init__(self, model: nn.Module, device: torch.device, end_token_id: int):
        self.model = model.to(device)
        self.device = device
        self.end_token_id = end_token_id

    def __call__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_gen_len: int = 256,
            no_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_seq_len = input_ids.shape

        # Calculate actual initial lengths from attention mask
        initial_lens = attention_mask.sum(dim=1)

        # Create buffers for generation tracking
        current_lens = initial_lens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_probs = torch.zeros((batch_size, max_gen_len), device=self.device)

        # Create working copies that we'll modify
        working_ids = input_ids.clone()
        working_mask = attention_mask.clone()

        for step in range(max_gen_len):
            # Find active sequences that haven't reached max length and aren't finished
            active = (~finished) & (current_lens < max_seq_len)
            if not active.any():
                break

            # Forward pass for active sequences only
            with torch.set_grad_enabled(not no_grad):
                logits = self.model(
                    working_ids[active, :current_lens[active].max()],
                    attention_mask=working_mask[active, :current_lens[active].max()]
                )

            # Get last token logits
            last_logits = logits[:, -1, :]

            # Sample next tokens and log probs
            next_tokens, step_log_probs = sample_batch(
                last_logits, temperature, top_k, top_p
            )

            # Update working tensors for active sequences
            for i, idx in enumerate(active.nonzero(as_tuple=True)[0]):
                if current_lens[idx] >= max_seq_len:
                    continue

                # Store generated token
                working_ids[idx, current_lens[idx]] = next_tokens[i]
                working_mask[idx, current_lens[idx]] = 1
                log_probs[idx, step] = step_log_probs[i]

                # Update tracking
                current_lens[idx] += 1
                if next_tokens[i] == self.end_token_id:
                    finished[idx] = True

        # Extract only the generated portion (from initial_lens to current_lens)
        generated_ids = torch.zeros((batch_size, max_gen_len), dtype=torch.long, device=self.device)
        final_mask = torch.zeros((batch_size, max_gen_len), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            start = initial_lens[i].item()
            end = current_lens[i].item()
            gen_len = min(end - start, max_gen_len)
            generated_ids[i, :gen_len] = working_ids[i, start:start + gen_len]
            final_mask[i, :gen_len] = working_mask[i, start:start + gen_len]

        return generated_ids, final_mask, log_probs


class BatchSampleDecoder:
    def __init__(self, sampler: BatchSampler, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = self.sampler.device

    def tokenize_batch(self, texts: list[str], max_seq_len: int = 256):
        tokenized = self.tokenizer(
            texts,
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
        )
        return {
            'input_ids': tokenized['input_ids'].to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device)
        }

    def generate(
            self,
            texts: list[str],
            temperature: float = 0.1,
            top_p: float = 0.9,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> list[str]:
        tokenized = self.tokenize_batch(texts, max_seq_len)
        generated_ids, _, _ = self.sampler(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
            no_grad=no_grad,
        )

        decoded = []
        for seq in generated_ids:
            # Trim after end token
            end_pos = (seq == self.sampler.end_token_id).nonzero()
            if end_pos.size(0) > 0:
                seq = seq[:end_pos[0] + 1]
            decoded.append(self.tokenizer.decode(seq).replace('Ċ', '\n').replace('Ġ', ' '))

        return decoded

    def generate_with_log_probs(
            self,
            texts: list[str],
            temperature: float = 0.1,
            top_p: float = 0.9,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> tuple[list[str], torch.Tensor]:
        tokenized = self.tokenize_batch(texts, max_seq_len)
        generated_ids, _, log_probs = self.sampler(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
            no_grad=no_grad,
        )

        decoded = []
        for i, seq in enumerate(generated_ids):
            # Trim after end token
            end_pos = (seq == self.sampler.end_token_id).nonzero()
            if end_pos.size(0) > 0:
                seq = seq[:end_pos[0] + 1]
            decoded.append(self.tokenizer.decode(seq).replace('Ċ', '\n').replace('Ġ', ' '))

        return decoded, log_probs

    def __call__(
            self,
            texts: list[str],
            temperature: float = 0.1,
            top_p: float = 0.9,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> list[str]:
        return self.generate(texts, temperature, top_p, max_seq_len, no_grad)