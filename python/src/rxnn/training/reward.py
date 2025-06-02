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
            bleu_ref_factor: float = 0.5,
            bleu_saved_factor: float = 0.5,
            cos_factor: float = 0.5,
            cos_ref_factor: float = 0.5,
            cos_saved_factor: float = 0.5,
            multi_cos_ref_factor: float = 0.3,
            multi_cos_saved_factor: float = 0.5,
            multi_cos_running_mean_factor: float = 0.2,
            neg_bleu_factor: Optional[float] = None,
            neg_cos_factor: Optional[float] = None,
            neg_cos_ref_factor: Optional[float] = None,
            neg_cos_saved_factor: Optional[float] = None,
            neg_bleu_ref_factor: float = 0.5,
            neg_bleu_saved_factor: float = 0.5,
            allow_not_summing_factors: bool = False,
            reward_len: bool = False,
            neg_reward_len: bool = False,
            max_rewarded_len: int = None,
            len_factor: int = None,
            use_running_mean: bool = True,
            running_mean_decay: float = 0.2,
            bleu_saved_weights: tuple = (0.5, 0.5),
            bleu_ref_weights: tuple = (0.5, 0.5),
            tanh_reward_scale: bool = False,
            rewards_scale: float = 1.0,
    ):
        self.shared_embedding = shared_embedding.to(device)
        self.device = device
        self.bleu_with_saved_data = bleu_with_saved_data

        self.bleu_factor = bleu_factor
        self.bleu_ref_factor = bleu_ref_factor
        self.bleu_saved_factor = bleu_saved_factor
        self.cos_factor = cos_factor
        self.cos_ref_factor = cos_ref_factor
        self.cos_saved_factor = cos_saved_factor
        self.multi_cos_ref_factor = multi_cos_ref_factor
        self.multi_cos_saved_factor = multi_cos_saved_factor
        self.multi_cos_running_mean_factor = multi_cos_running_mean_factor
        self.neg_bleu_factor = neg_bleu_factor if neg_bleu_factor is not None else bleu_factor
        self.neg_cos_factor = neg_cos_factor if neg_cos_factor is not None else cos_factor
        self.neg_cos_ref_factor = neg_cos_ref_factor if neg_cos_ref_factor is not None else cos_ref_factor
        self.neg_cos_saved_factor = neg_cos_saved_factor if neg_cos_saved_factor is not None else cos_saved_factor
        self.neg_bleu_ref_factor = neg_bleu_ref_factor
        self.neg_bleu_saved_factor = neg_bleu_saved_factor
        self.reward_len = reward_len
        self.neg_reward_len = neg_reward_len
        self.max_rewarded_len = max_rewarded_len
        self.len_factor = len_factor
        self.use_running_mean = use_running_mean
        self.running_mean_decay = running_mean_decay
        self.bleu_ref_weights = bleu_ref_weights
        self.bleu_saved_weights = bleu_saved_weights
        self.tanh_reward_scale = tanh_reward_scale
        self.rewards_scale = rewards_scale

        self.prev_data_running_mean = None

        if not allow_not_summing_factors:
            if reward_len:
                assert self.bleu_factor + self.cos_factor + self.len_factor == 1.0
                assert self.neg_bleu_factor + self.neg_cos_factor + self.len_factor == 1.0
                assert self.multi_cos_ref_factor + self.multi_cos_saved_factor + self.multi_cos_running_mean_factor == 1.0
                assert self.bleu_ref_factor + self.bleu_saved_factor == 1.0
                assert self.cos_ref_factor + self.cos_saved_factor == 1.0
                assert self.neg_cos_ref_factor + self.neg_cos_saved_factor == 1.0
                assert self.neg_bleu_ref_factor + self.neg_bleu_saved_factor == 1.0
            else:
                assert self.bleu_factor + self.cos_factor == 1.0
                assert self.bleu_ref_factor + self.bleu_saved_factor == 1.0
                assert self.cos_ref_factor + self.cos_saved_factor == 1.0
                assert self.multi_cos_ref_factor + self.multi_cos_saved_factor + self.multi_cos_running_mean_factor == 1.0
                assert self.neg_bleu_factor + self.neg_cos_factor == 1.0
                assert self.neg_cos_ref_factor + self.neg_cos_saved_factor == 1.0
                assert self.neg_bleu_ref_factor + self.neg_bleu_saved_factor == 1.0

    def _sentence_bleu(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> float:
        from nltk.translate.bleu_score import sentence_bleu

        if self.bleu_with_saved_data:
            ref_bleu = sentence_bleu([reference], generated, weights=self.bleu_ref_weights)
            saved_bleu = sentence_bleu([saved_data], generated, weights=self.bleu_saved_weights)
            return self.bleu_ref_factor * ref_bleu + self.bleu_saved_factor * saved_bleu
        else:
            return sentence_bleu([reference], generated, weights=self.bleu_ref_weights)


    def _negative_sentence_bleu(self, generated: torch.Tensor, reference: torch.Tensor,
                                saved_data: torch.Tensor) -> float:
        from nltk.translate.bleu_score import sentence_bleu

        if self.bleu_with_saved_data:
            ref_bleu = sentence_bleu([reference], generated, weights=self.bleu_ref_weights)
            saved_bleu = sentence_bleu([saved_data], generated, weights=self.bleu_saved_weights)
            saved_bleu = 1 - saved_bleu

            return self.neg_bleu_ref_factor * ref_bleu + self.neg_bleu_saved_factor * saved_bleu
        else:
            return sentence_bleu([reference], generated, weights=self.bleu_ref_weights)

    def batch_bleu(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor) -> list[float]:
        batch_size = generated.size(0)
        return [self._sentence_bleu(generated[i], reference[i], saved_data[i]) for i in range(batch_size)]

    def _sequence_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        embedding = self.shared_embedding(sequence.to(self.device))
        return embedding.mean(dim=1)

    def _cosine_sim(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor):
        generated_emb = self._sequence_embedding(generated)

        gen_and_saved = (F.cosine_similarity(generated_emb, self._sequence_embedding(saved_data)) + 1) / 2
        gen_and_ref = (F.cosine_similarity(generated_emb, self._sequence_embedding(reference)) + 1) / 2
        return gen_and_saved, gen_and_ref

    def _cosine_sim_running_mean(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor):
        generated_emb = self._sequence_embedding(generated)

        gen_and_saved = (F.cosine_similarity(generated_emb, self._sequence_embedding(saved_data)) + 1) / 2
        gen_and_ref = (F.cosine_similarity(generated_emb, self._sequence_embedding(reference)) + 1) / 2
        gen_and_mean = (F.cosine_similarity(generated_emb, self.prev_data_running_mean) + 1) / 2
        return gen_and_saved, gen_and_ref, gen_and_mean

    def batch_cosine(self, generated: torch.Tensor, reference: torch.Tensor, saved_data: torch.Tensor,
                     include_running_mean: bool = False, negative_running_mean: bool = False) -> torch.Tensor:
        if self.use_running_mean and negative_running_mean:
            gen_and_saved, gen_and_ref, gen_and_mean = self._cosine_sim_running_mean(generated, reference, saved_data)
            return self.multi_cos_saved_factor * gen_and_saved + self.multi_cos_ref_factor * gen_and_ref + self.multi_cos_saved_factor * (
                        1 - gen_and_mean)
        elif self.use_running_mean and include_running_mean:
            gen_and_saved, gen_and_ref, gen_and_mean = self._cosine_sim_running_mean(generated, reference, saved_data)
            return self.multi_cos_saved_factor * gen_and_saved + self.multi_cos_ref_factor * gen_and_ref + self.multi_cos_saved_factor * gen_and_mean
        else:
            gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)
            return self.cos_saved_factor * gen_and_saved + self.cos_ref_factor * gen_and_ref

    def negative_cosine(self, generated: torch.Tensor, reference: torch.Tensor,
                        saved_data: torch.Tensor) -> torch.Tensor:
        gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)

        return self.neg_cos_saved_factor * (1 - gen_and_saved) + self.neg_cos_ref_factor * gen_and_ref

    def len_reward(self, generated: TokenizedDict):
        lens = generated['attention_mask'].sum(dim=1)
        neg_lens = self.max_rewarded_len / lens if self.neg_reward_len else 1.0
        len_reward = torch.where(lens >= self.max_rewarded_len, neg_lens, lens / self.max_rewarded_len)
        return len_reward

    def reset_running_mean(self):
        self.prev_data_running_mean = None

    def init_running_mean(self, prev_data: torch.Tensor):
        self.prev_data_running_mean = self._sequence_embedding(prev_data)

    def update_running_mean(self, prev_data: torch.Tensor):
        self.prev_data_running_mean = (1 - self.running_mean_decay) * self._sequence_embedding(
            prev_data) + self.running_mean_decay * self.prev_data_running_mean

    def _pre_scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.tanh_reward_scale:
            return (rewards * 2) - 1  # Convert [0,1] to [-1,1]
        else:
            return rewards

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            prev_data: TokenizedDict = None,
            mode: MrlRewardMode = MrlRewardMode.STANDARD
    ) -> list[float]:
        if prev_data is not None:
            if self.prev_data_running_mean is None:
                self.init_running_mean(prev_data['input_ids'])
            else:
                self.update_running_mean(prev_data['input_ids'])

        if mode == MrlRewardMode.STANDARD:
            bleu = self.batch_bleu(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            cosine = self.batch_cosine(generated['input_ids'], reference['input_ids'], saved_data['input_ids'],
                                       include_running_mean=prev_data is not None)
            sim_rewards = self.bleu_factor * torch.tensor(bleu, device=self.device) + self.cos_factor * cosine
        elif mode == MrlRewardMode.LONG_RANGE:
            bleu = self.batch_bleu(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            cosine = self.batch_cosine(generated['input_ids'], reference['input_ids'], saved_data['input_ids'],
                                       negative_running_mean=prev_data is not None)
            sim_rewards = self.bleu_factor * torch.tensor(bleu, device=self.device) + self.cos_factor * cosine
        else:
            bleu = self.batch_bleu(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            cosine = self.negative_cosine(generated['input_ids'], reference['input_ids'], saved_data['input_ids'])
            sim_rewards = self.neg_bleu_factor * torch.tensor(bleu, device=self.device) + self.neg_cos_factor * cosine

        rewards = self._pre_scale_rewards(sim_rewards + self.len_factor * self.len_reward(generated) if self.reward_len else sim_rewards) * self.rewards_scale
        return rewards.tolist()
