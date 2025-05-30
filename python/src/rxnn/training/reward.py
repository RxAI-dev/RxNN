import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional
from .utils import TokenizedDict


class MrlRewardMode(Enum):
    STANDARD = 1
    NEGATIVE = 2
    LONG_RANGE = 3

class MrlRewardModel:
    def __init__(
            self,
            shared_embedding: nn.Embedding,
            device: torch.device,
            bleu_with_saved_data: bool = False,
            bleu_factor: float = 0.5,
            cos_factor: float = 0.5,
            cos_ref_factor: float = 0.5,
            cos_saved_factor: float = 0.5,
            neg_bleu_factor: Optional[float] = None,
            neg_cos_factor: Optional[float] = None,
            neg_cos_ref_factor: Optional[float] = None,
            neg_cos_saved_factor: Optional[float] = None,
            neg_bleu_ref_factor: float = 0.5,
            neg_bleu_saved_factor: float = 0.5,
            allow_not_summing_factors: bool = False,
    ):
        self.shared_embedding = shared_embedding.to(device)
        self.device = device
        self.bleu_with_saved_data = bleu_with_saved_data

        self.bleu_factor = bleu_factor
        self.cos_factor = cos_factor
        self.cos_ref_factor = cos_ref_factor
        self.cos_saved_factor = cos_saved_factor
        self.neg_bleu_factor = neg_bleu_factor if neg_bleu_factor is not None else bleu_factor
        self.neg_cos_factor = neg_cos_factor if neg_cos_factor is not None else cos_factor
        self.neg_cos_ref_factor = neg_cos_ref_factor if neg_cos_ref_factor is not None else cos_ref_factor
        self.neg_cos_saved_factor = neg_cos_saved_factor if neg_cos_saved_factor is not None else cos_saved_factor
        self.neg_bleu_ref_factor = neg_bleu_ref_factor
        self.neg_bleu_saved_factor = neg_bleu_saved_factor

        if not allow_not_summing_factors:
            assert self.bleu_factor + self.cos_factor == 1.0
            assert self.cos_ref_factor + self.cos_saved_factor == 1.0
            assert self.neg_bleu_factor + self.neg_cos_factor == 1.0
            assert self.neg_cos_ref_factor + self.neg_cos_saved_factor == 1.0
            assert self.neg_bleu_ref_factor + self.neg_bleu_saved_factor == 1.0

    def _sentence_bleu(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> float:
        from nltk.translate.bleu_score import sentence_bleu
        refs = [reference, saved_data] if self.bleu_with_saved_data else [reference]
        return sentence_bleu(refs, generated, weights=(0.25, 0.25, 0.25, 0.25))

    def _negative_sentence_bleu(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> float:
        from nltk.translate.bleu_score import sentence_bleu

        if self.bleu_with_saved_data:
            ref_bleu = sentence_bleu([reference], generated, weights=(0.25, 0.25, 0.25, 0.25))
            saved_bleu = sentence_bleu([saved_data], generated, weights=(0.25, 0.25, 0.25))
            saved_bleu = 1 - saved_bleu

            return (self.neg_bleu_ref_factor * ref_bleu + self.neg_bleu_saved_factor * saved_bleu) / 2
        else:
            return sentence_bleu([reference], generated, weights=(0.25, 0.25, 0.25, 0.25))

    def batch_bleu(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> list[float]:
        batch_size = generated.size(0)
        return [self._sentence_bleu(generated[i], reference[i], saved_data[i]) for i in range(batch_size)]

    def _sequence_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        embedding = self.shared_embedding(sequence.to(self.device))
        return embedding.mean(dim=1)

    def _cosine_sim(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor):
        generated_emb = self._sequence_embedding(generated)

        gen_and_saved = F.cosine_similarity(generated_emb, self._sequence_embedding(saved_data))
        gen_and_ref = F.cosine_similarity(generated_emb, self._sequence_embedding(reference))
        return gen_and_saved, gen_and_ref

    def batch_cosine(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> torch.Tensor:
        gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)

        return self.cos_saved_factor * gen_and_saved + self.cos_ref_factor * gen_and_ref

    def negative_cosine(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> torch.Tensor:
        gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)

        return self.neg_cos_saved_factor * (1 - gen_and_saved) + self.neg_cos_ref_factor * gen_and_ref

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            mode: MrlRewardMode = MrlRewardMode.STANDARD
    ) -> list[float]:
        if mode == MrlRewardMode.STANDARD or mode == MrlRewardMode.LONG_RANGE:
            bleu = self.batch_bleu(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            cosine = self.batch_cosine(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            return (self.bleu_factor * torch.tensor(bleu, device=self.device) + self.cos_factor * cosine).tolist()
        else:
            bleu = self.batch_bleu(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            cosine = self.negative_cosine(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            return (self.neg_bleu_factor * torch.tensor(bleu, device=self.device) + self.neg_cos_factor * cosine).tolist()

