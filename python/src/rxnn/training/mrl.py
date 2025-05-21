import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.opt_einsum import strategy
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, Union, TypedDict
from enum import Enum
import random, os
from ..transformers.sampler import Sampler, InteractionSampler, BatchSampler
from .callbacks import MrlTrainerCallback
from .dataset import MrlCurriculumDataset
from .utils import smart_concat, smart_concat_critic_states, SpecialTokenIds, TokenizedDict

class MrlConfig(TypedDict):
    lr: float
    critic_lr: float
    max_seq_len: int
    critic_max_len: int
    weight_decay: float
    critic_weight_decay: float

class MrlStrategy(Enum):
    SINGLE_STEP_STRATEGY = 1
    MULTI_STEP_STRATEGY = 2
    LONG_RANGE_STRATEGY = 3

class CurriculumConfig(TypedDict):
    steps: int
    epochs: int
    dataset: MrlCurriculumDataset
    eval_dataset: Optional[MrlCurriculumDataset]
    callbacks: list[MrlTrainerCallback]
    strategy: MrlStrategy
    unfreeze_epoch: int
    random_resets: bool
    random_resets_from: Optional[int]
    random_resets_ratio: Optional[float]

class SamplerConfig(TypedDict):
    temperature: float
    top_k: int
    top_p: float

class MrlActorAction(Enum):
    DECODE = 1
    UPDATE = 2

class MrlRewardMode(Enum):
    STANDARD = 1
    NEGATIVE = 2
    LONG_RANGE = 3

class MrlTrajectoryStep(TypedDict):
    state: tuple[TokenizedDict, TokenizedDict, TokenizedDict]
    action: TokenizedDict
    log_probs: torch.Tensor
    reward: list[float]
    reference: TokenizedDict

class MrlTrajectoryEpisode(TypedDict):
    reset_stm: bool
    steps: list[MrlTrajectoryStep]

class MrlRewardModel:
    def __init__(self):
        pass

    def __call__(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict, mode: MrlRewardMode = MrlRewardMode.STANDARD) -> list[float]:
        from nltk.translate.bleu_score import sentence_bleu
        return sentence_bleu(generated['input_ids'], reference['input_ids'], weights=(0.25, 0.25, 0.25, 0.25))

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

    def freeze_components(self):
        """Freeze encoder/decoder except memory-related layers."""
        if self.encoder.freeze_without_memory is not None:
            self.encoder.freeze_without_memory()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.model.trainable_cross_attention_(False)
        if self.decoder.freeze_without_memory is not None:
            self.decoder.freeze_without_memory()
        else:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.model.trainable_cross_attention_(False)
        # Unfreeze memory attention
        for param in self.memory_attention.parameters():
            param.requires_grad = True

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
        return self.value_head(x)

class MRLTrainer:
    def __init__(
            self,
            actor: MrlActorModel,
            critic: MrlCriticModel,
            reward: MrlRewardModel,
            device: torch.device,
            config: MrlConfig,
            sampler_config: SamplerConfig,
            log_dir: str = None,
            pad_token_id: int = 0,
            start_token_id: int = 2,
            end_token_id: int = 3,
            use_ddp: bool = False,
            use_amp: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        """
        Trainer for Memory Reinforcement Learning (MRL) in Reactive Transformer.

        Args:
            actor: MRL Actor model with encoder, decoder and memory attention.
            critic: Critic network for advantage estimation.
            config: Configuration dictionary with hyperparameters.
        """
        self.actor = actor
        self.critic = critic
        self.reward = reward
        self.device = device
        self.max_seq_len = config.get('max_seq_len', 1024)
        self.critic_max_len = config.get('critic_max_len', 2048)

        # Move models to device
        if use_amp:
            self.actor.to(self.device)
            self.critic.to(self.device)
        else:
            self.actor.to(self.device, dtype=dtype)
            self.critic.to(self.device, dtype=dtype)

        # Batch Sampler for answer generation
        self.generator = BatchSampler(self.actor, self.device, end_token_id=end_token_id)
        self.sampler_config = sampler_config

        self.special_token_ids: SpecialTokenIds = {
            'pad': pad_token_id,
            'bos': start_token_id,
            'eos': end_token_id,
        }

        self.use_ddp = use_ddp
        self.use_amp = use_amp
        self.dtype = dtype

        # Optimizers
        self.optimizer = torch.optim.AdamW(
            self.actor.unique_parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.get("critic_lr", 1e-4),
            weight_decay=config.get("critic_weight_decay", 0.01),
        )

        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        self.critic_scaler = torch.amp.GradScaler() if self.use_amp else None

        # TensorBoard Writer
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Dynamic fields, updated for each curriculum step
        self.curriculum_steps = 0
        self.train_dataset = None
        self.eval_dataset = None
        self.random_resets_ratio = 0.0
        self.strategy = None
        self.callbacks = []


    def reset_stm(self) -> bool:
        """Reset Short-Term Memory state with random reset ratio."""
        if self.random_resets_ratio == 1.0:
            self.actor.reset_memory()
            return True
        else:
            rng = random.random()
            if rng <= self.random_resets_ratio:
                self.actor.reset_memory()
                return True
            else:
                return False

    def encode_and_update_stm(self, query: TokenizedDict, answer: TokenizedDict):
        """Encode interaction and update STM."""
        # 1. Encode data and update memory - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 2. Concatenate batch of queries and answers (they are already on training device)
                inputs = smart_concat(query, answer, self.max_seq_len, self.special_token_ids['pad'])
                # 3. Encode data and update STM
                self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)
        else:
            # 2. Concatenate batch of queries and answers (they are already on training device)
            inputs = smart_concat(query, answer, self.max_seq_len, self.special_token_ids['pad'])
            # 3. Encode data and update STM
            self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)

    def generate_answer(self, query: TokenizedDict) -> tuple[TokenizedDict, torch.Tensor]:
        """Generate response using batch sampler with decoder."""
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                input_ids, attention_mask, log_probs = self.generator(
                    query['input_ids'],
                    query['attention_mask'],
                    max_gen_len=self.max_seq_len,
                    **self.sampler_config,
                )
        else:
            input_ids, attention_mask, log_probs = self.generator(
                query['input_ids'],
                query['attention_mask'],
                max_gen_len=self.max_seq_len,
                **self.sampler_config,
            )

        generated_answer: TokenizedDict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        return generated_answer, log_probs

    def compute_reward(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: tuple[TokenizedDict, TokenizedDict], mode: MrlRewardMode = MrlRewardMode.STANDARD) -> list[float]:
        """Compute reward based on memory retention (e.g., BLEU-4)."""
        saved_query, saved_answer = saved_data
        saved_interaction = smart_concat(saved_query, saved_answer, max_length=self.max_seq_len, pad_token_id=self.special_token_ids['pad'])
        reward = self.reward(generated, reference, saved_interaction, mode=mode)
        for cb in self.callbacks:
            cb.on_reward(self.actor, reward, generated, reference, saved_interaction)
        return reward

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        if self.use_amp:
            return {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.device, dtype=self.dtype),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=self.dtype),
            }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def _cpu_detach(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].detach().cpu(),
            'attention_mask': batch['attention_mask'].detach().cpu(),
        }

    def _cpu_detach_multiple(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._cpu_detach(batch) for batch in batches]

    def collect_trajectories(self, dataloader: DataLoader) -> list[MrlTrajectoryEpisode]:
        """Collect trajectories for PPO for current curriculum step."""
        # 1. Init PyTorch DataLoader and trajectories list
        trajectories = []

        # Trajectories should be collected in no grad mode (?)
        with torch.no_grad():
            # 2. Collect episode trajectories for all batches in dataset
            for batch_idx, batch in enumerate(dataloader):
                # 3. Reset Short-Term Memory state (with random reset ratio - sometimes it will be good to build memory
                # state from existing one, instead of new random one)
                reset_done = self.reset_stm()

                # 4. Get first batch of interactions (data to save) and follow-up interactions for current episode, based on curriculum step
                first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                interactions = interactions[:self.curriculum_steps]
                interactions_len = len(interactions)
                # 5. Encode and update STM with data to save from first interaction
                self.encode_and_update_stm(*self._move_multiple_batches(first_query, first_answer))

                # 6. Save first interaction as data to save (for trajectory state)
                query, answer = first_query, first_answer

                # 7. Run training strategy for follow-up interactions
                episode_steps = []
                for i, interaction in enumerate(interactions):
                    # 8. Generate batch of answers based on batch of follow-up queries
                    next_query = self._move_batch(interaction['query'])
                    generated_answer, log_probs = self.generate_answer(next_query)

                    is_last_interaction = (i + 1) == interactions_len

                    detached_answer = self._cpu_detach(generated_answer) # detach and keep states on CPU

                    # 9. Depending on strategy compute reward
                    if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                        # a) long-range - first interaction - change topic - negative reward (it shouldn't include saved data)
                        reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer), mode=MrlRewardMode.NEGATIVE)
                    elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                        # b) long-range - last interaction - first interaction topic - long-range reward (it should include content from first interaction)
                        reward = self.compute_reward(detached_answer, interaction['answer'], (first_query, first_answer), mode=MrlRewardMode.LONG_RANGE)
                    else:
                        # c) standard reward - generated answer should include some content from previous interaction (saved data), like reference answer
                        reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer), mode=MrlRewardMode.STANDARD)

                    # 10. Update STM with generated response (except last interaction, it's not needed)
                    if not is_last_interaction:
                        self.encode_and_update_stm(next_query, generated_answer) # update with generated_answer on GPU

                    # 11. Store trajectory step
                    trajectory: MrlTrajectoryStep = {
                        'state': (query, answer, interaction['query']),
                        'action': detached_answer, # or we need action with gradients here ?
                        'log_probs': log_probs.detach().cpu(),
                        'reward': reward,
                        'reference': interaction['answer'],
                    }
                    episode_steps.append(trajectory)

                    # 12. Set current interaction query and generated answer (batches), as saved data for next interaction
                    query, answer = interaction['query'], detached_answer

                # 13. Append full batched episode (number of steps depends on curriculum stage) to trajectories
                episode_trajectory: MrlTrajectoryEpisode = {
                    'reset_stm': reset_done,
                    'steps': episode_steps,
                }
                trajectories.append(episode_trajectory)

                # 14. Run "on episode collected" callbacks
                for cb in self.callbacks:
                    cb.on_episode_collected(self.actor, batch_idx, episode_trajectory)

        return trajectories

    def _critic_loss(self, inputs: TokenizedDict, rewards: torch.Tensor) -> torch.Tensor:
        # 3. Calculate values with critic encoder
        values = self.critic(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).logits.squeeze()
        # 4. Calculate critic loss, run backpropagation and optimizer step
        loss = nn.MSELoss()(values, rewards)
        return loss

    def update_critic(self, states: list[tuple[TokenizedDict, TokenizedDict, TokenizedDict]], rewards: list[torch.Tensor], epoch: int):
        """Update critic network using MSE loss."""
        # 1. Run critic updates for all collected batches
        critic_losses = []
        for state, reward in zip(states, rewards):
            # 2. Move state batches to training device (GPU)
            prev_query, prev_answer, next_query = self._move_multiple_batches(*state)

            # 3. Reset critic gradients
            self.critic_optimizer.zero_grad()

            # 4. Run critic and calculate loss - in autocast on/off mode
            if self.use_amp:
                # Move tensors to training device and calculate loss in autocast mode
                batch_rewards = reward.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                    # Concatenate state into single critic input sequence
                    inputs = smart_concat_critic_states(
                        prev_query, prev_answer, next_query,
                        max_length=self.critic_max_len,
                        special_token_ids=self.special_token_ids,
                    )
                    loss = self._critic_loss(inputs, batch_rewards)
                # Run backpropagation with scaler
                self.critic_scaler.scale(loss).backward()
                # Unscale and clip gradients
                self.critic_scaler.unscale_(self.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=False)
                # Run scaled optimization step
                self.critic_scaler.step(self.critic_optimizer)
                self.critic_scaler.update()
            else:
                # Concatenate state into single critic input sequence
                inputs = smart_concat_critic_states(
                    prev_query, prev_answer, next_query,
                    max_length=self.critic_max_len,
                    special_token_ids=self.special_token_ids,
                )
                # Calculate loss
                loss = self._critic_loss(inputs, reward.to(self.device, dtype=self.dtype))
                # Run backpropagation
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=False)
                # Run optimizer step
                self.critic_optimizer.step()
            # Accumulate loss for callbacks
            critic_losses.append(loss.item())

        # 5. Calculate mean loss for callbacks (logging, etc.)
        critic_mean_loss = torch.stack(critic_losses).mean().item()
        # 6. Run "on critic updated" callbacks
        for cb in self.callbacks:
            cb.on_critic_updated(self.actor, self.critic, epoch, critic_mean_loss)

        return critic_mean_loss

    def _critic_advantages(self, critic_state: TokenizedDict, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            values = self.critic(critic_state['input_ids'],
                                 attention_mask=critic_state['attention_mask']).logits.squeeze()
            return rewards - values

    def ppo_step(self, trajectories: list[MrlTrajectoryEpisode]):
        """Perform PPO update step using trajectories."""
        # 1. Run update separately for episodes in trajectory - we have to reset memory before each episode, and update
        # memory, based on collected episode data
        all_losses = []
        trajectories_len = len(trajectories)
        for episode_idx, episode in enumerate(trajectories):
            episode_steps = episode['steps']
            should_reset_stm = episode['reset_stm']
            # 2. Reset memory for current batch episode
            if should_reset_stm:
                self.reset_stm()

            # 3. Accumulate logits and advantages for the batched episode
            action_logits = []
            action_advantages = []

            # 4. Run episode steps - each episode has number of steps depending on curriculum stage. Each step is run for all batch
            for step in episode_steps:
                state, action, reward, log_probs = step['state'], step['action'], step['reward'], step['log_probs']
                query, answer, next_query = self._move_multiple_batches(*state)
                action = self._move_batch(action)
                rewards = torch.tensor(reward).to(self.device)

                # 5. Compute advantages using critic
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        critic_state = smart_concat_critic_states(query, answer, next_query,
                                                                  max_length=self.critic_max_len,
                                                                  special_token_ids=self.special_token_ids)
                        advantages = self._critic_advantages(critic_state, rewards)
                else:
                    critic_state = smart_concat_critic_states(query, answer, next_query, max_length=self.critic_max_len,
                                                              special_token_ids=self.special_token_ids)
                    advantages = self._critic_advantages(critic_state, rewards)

                action_advantages.append(advantages)

                # TODO: Use accumulated log_probs in PPO Calculation

                # 6. Encode and update STM on each step, to include encoder and memory attention gradients in loss
                self.encode_and_update_stm(query, answer)
                # 7. Concatenate next query and action and get action logits from decoder
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        inputs = smart_concat(next_query, action, max_length=self.max_seq_len, pad_token_id=self.special_token_ids['pad'])
                        logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.DECODE)
                else:
                    inputs = smart_concat(next_query, action, max_length=self.max_seq_len, pad_token_id=self.special_token_ids['pad'])
                    logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.DECODE)
                action_logits.append(logits)

                # 8. Calculate PPO loss (per step)
                # TODO: Implement correct PPO handling - the question is if we should run backpropagation and update after each episode step (batch)
                # TODO: or after all steps in episode (multiple batches). I think, that rather first option, each step is a full run through encoder,
                # TODO: memory attention and decoder, so accumulating gradients from multiple runs may not be a good idea
                # Placeholder for actual PPO loss calculation
                # This would involve computing log_probs, ratios, and clipped surrogate loss
                policy_loss = -torch.mean(logits * advantages)

                # TODO: All correct PPO steps

                # 9. Reset gradients
                self.optimizer.zero_grad()

                # 10. Update the model in AMP or regular mode
                if self.use_amp:
                    self.scaler.scale(policy_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0, error_if_nonfinite=False)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0, error_if_nonfinite=False)
                    self.optimizer.step()

                policy_loss_item = policy_loss.item()
                all_losses.append(policy_loss_item)

            # 8. Calculate PPO loss (per episode)
            # TODO: Implement correct PPO handling - the question is if we should run backpropagation and update after each episode step (batch)
            # TODO: or after all steps in episode (multiple batches). I think, that rather first option, each step is a full run through encoder,
            # TODO: memory attention and decoder, so accumulating gradients from multiple runs may not be a good idea
            # Placeholder for actual PPO loss calculation
            # This would involve computing log_probs, ratios, and clipped surrogate loss
            # policy_loss = -torch.mean(torch.stack(action_logits).squeeze() * torch.stack(action_advantages).squeeze())
            #
            # self.optimizer.zero_grad()
            #
            # if self.use_amp:
            #     self.scaler.scale(policy_loss).backward()
            #     self.scaler.unscale_(self.optimizer)
            #     torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0, error_if_nonfinite=False)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            #     policy_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0, error_if_nonfinite=False)
            #     self.optimizer.step()
            #
            # policy_loss_item = policy_loss.item()
            # all_losses.append(policy_loss_item)

            # 10. Run "on episode updated" callback
            for cb in self.callbacks:
                cb.on_episode_updated(self.actor, episode_idx, trajectories_len, policy_loss_item)

        return torch.mean(torch.tensor(all_losses)).item()

    def _critic_states_and_rewards(self, trajectories: list[MrlTrajectoryEpisode]):
        flat_trajectories = [t for episode in trajectories for t in episode['steps']]
        states = [t['state'] for t in flat_trajectories]
        rewards = [torch.tensor(t["reward"]) for t in flat_trajectories]
        return states, rewards

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        # 1. Collect trajectories for current epoch
        trajectories = self.collect_trajectories(dataloader)

        # 2. Flatten trajectories and collect state and rewards for critic update
        states, rewards = self._critic_states_and_rewards(trajectories)
        # 3. Update critic model, based on states and rewards
        critic_loss = self.update_critic(states, rewards, epoch)

        # 4. Run PPO algorithm step
        policy_loss = self.ppo_step(trajectories)
        return policy_loss, critic_loss


    def evaluate(self, batch_size: int, epoch: int):
        """Evaluate model on validation dataset."""
        # 1. Init evaluation DataLoader
        dataloader = self._eval_loader(batch_size)
        total_reward = torch.tensor(0.0).to(self.device)
        count = torch.tensor(0).to(self.device)

        # 2. Run evaluation on all batch episodes
        for batch in dataloader:
            with torch.no_grad():
                # 3. Reset STM with random resets ratio
                self.reset_stm()

                # 4. Get batches for first queries, answers and all follow-up interactions
                first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                # 5. Encode and update STM with initial interactions (batch)
                self.encode_and_update_stm(*self._move_multiple_batches(first_query, first_answer))

                # 6. Save follow-up interactions len and first query and answer as previous one for iteration
                interactions_len = len(interactions)
                query, answer = first_query, first_answer
                # 7. Run all follow-up interactions
                for i, interaction in enumerate(interactions):
                    # 8. Generate batch of answers
                    next_query = self._move_batch(interaction['query'])
                    generated_answer, _ = self.generate_answer(next_query)

                    is_last_interaction = (i + 1) == interactions_len

                    detached_answer = self._cpu_detach(generated_answer)

                    # 9. Depending on current strategy and step, compute reward
                    if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                        reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer), mode=MrlRewardMode.NEGATIVE)
                    elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                        reward = self.compute_reward(detached_answer, interaction['answer'],(first_query, first_answer), mode=MrlRewardMode.LONG_RANGE)
                    else:
                        reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer), mode=MrlRewardMode.STANDARD)

                    # 10. Encode and update memory for the next interaction
                    if not is_last_interaction:
                        self.encode_and_update_stm(next_query, generated_answer)

                    # 11. Accumulate rewards
                    total_reward += torch.tensor(reward).mean()
                    count += 1
                    # 12. Save previous interaction
                    query, answer = interaction['query'], detached_answer

        # 13. Calculate average reward
        if self.use_ddp:
            total_sum = dist.all_reduce(total_reward, dist.ReduceOp.SUM)
            count_sum = dist.all_reduce(count, dist.ReduceOp.SUM)
            avg_reward = (total_sum / count_sum).item() if count_sum > 0 else 0
        else:
            avg_reward = (total_reward / count).item() if count > 0 else 0

        # 14. Run "on eval end" callbacks
        for cb in self.callbacks:
            cb.on_eval_end(self.actor, epoch, avg_reward)

    def _setup_curriculum_step(self, config: CurriculumConfig) -> tuple[tuple[int, int], tuple[bool, int, float]]:
        # 1. Set common fields based on config
        self.curriculum_steps = config.get('steps', 1) # number of steps to run in episode
        self.train_dataset = config.get('dataset', None) # training dataset for current curriculum stage
        self.eval_dataset = config.get('eval_dataset', None) # evaluation dataset for current curriculum stage
        self.callbacks = config.get('callbacks', []) # trainer callbacks for current curriculum stage
        self.strategy = config.get('strategy', MrlStrategy.MULTI_STEP_STRATEGY) # MRL strategy for given curriculum stage

        epochs = config.get('epochs', 5) # number of epochs for current stage
        unfreeze_epoch = config.get('unfreeze_epoch', 0) # epoch when components (other than memory) are unfrozen (before epoch starts)
        random_resets = config.get('random_resets', False) # flag for using random STM resets (recommended, as model should learn transitions between different states)
        random_resets_from = config.get('random_resets_from', None) # epoch from which random STM resets are started
        random_resets_ratio = config.get('random_resets_ratio', None) # ratio of random STM resets - 1.0 is "always reset", 0.0 is "no resets"

        return (epochs, unfreeze_epoch), (random_resets, random_resets_from, random_resets_ratio)

    def _eval_loader(self, batch_size: int):
        eval_dataset = self.eval_dataset
        if self.use_ddp:
            return DataLoader(
                eval_dataset,
                batch_size=batch_size,
                pin_memory=True,
                sampler=DistributedSampler(eval_dataset, shuffle=False),
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )
        else:
            return DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )

    def __call__(self, curriculum_config: list[CurriculumConfig], batch_size: int):
        """Start Memory Reinforcement Learning Curriculum."""

        # 1. Init DDP for distributed training mode
        if self.use_ddp:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            self.actor = DistributedDataParallel(self.actor, device_ids=[self.device.index])
            self.critic = DistributedDataParallel(self.critic, device_ids=[self.device.index])

        # 2. Run each curriculum step based on config
        for current_curriculum_step in curriculum_config:
            # 3. Setup training config for curriculum step
            epochs_config, random_resets_config = self._setup_curriculum_step(current_curriculum_step)
            epochs, unfreeze_epoch = epochs_config
            random_resets, random_resets_from, random_resets_ratio = random_resets_config
            assert self.train_dataset is not None
            print(f'Curriculum Steps Increased to {self.curriculum_steps}')

            # 4. Freeze all components except memory attention and memory cross-attention layers in decoder/encoder
            if unfreeze_epoch != 0:
                self.actor.freeze_components()

            # 5. Setup train DataLoader
            if self.use_ddp:
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    pin_memory=True,
                    collate_fn=MrlCurriculumDataset.collate_mrl_batch,
                )
            else:
                train_sampler = None
                dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    collate_fn=MrlCurriculumDataset.collate_mrl_batch,
                )


            # 6. Run selected number of epochs for given curriculum stage
            for epoch in range(epochs):
                # 7. Run "on epoch start" callbacks (log info, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_start(self.actor, epoch, epochs)

                # 8. Set random STM resets ratio from selected epoch
                if random_resets and random_resets_from <= epoch:
                    self.random_resets_ratio = random_resets_ratio
                else:
                    self.random_resets_ratio = 1.0

                # 9. Unfreeze all components before selected epoch
                if epoch == unfreeze_epoch:
                    self.actor.unfreeze_components()

                # 10. Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                # 11. Run reinforcement learning algorithms for current epoch
                policy_loss, critic_loss = self.train_epoch(dataloader, epoch)

                # 12. If evaluation dataset is provided, run evaluation steps
                if self.eval_dataset:
                    self.evaluate(batch_size, epoch)

                # 13. Finally, run "on epoch end" callbacks (save models, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_end(self.actor, epoch, epochs, policy_loss, critic_loss)

                # 14. Synchronize devices in DDP mode
                if self.use_ddp:
                    dist.barrier()

            # 15. Run "on_training_end" callbacks after each curriculum stage (they have own callbacks)
            for cb in self.callbacks:
                cb.on_training_end(self.actor)


        # 16. Training end - finish processes after all curriculum stages
        if self.use_ddp:
            dist.destroy_process_group()

        if self.writer:
            self.writer.close()