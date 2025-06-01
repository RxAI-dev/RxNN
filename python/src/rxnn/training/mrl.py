import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, TypedDict, Union, TypeAlias, Literal
from enum import Enum
import random, os
from ..transformers.sampler import BatchSampler
from .callbacks import MrlTrainerCallback
from .dataset import MrlCurriculumDataset
from .utils import smart_concat, smart_concat_critic_states, TokenizedDict
from .rl import RlAlgorithm
from .reward import MrlRewardMode, MrlRewardModel
from .models import MrlActorAction, MrlActorModel, MrlCriticModel


class MrlConfig(TypedDict):
    lr: float
    separate_memory_lr: Optional[bool]
    memory_lr: Optional[float]
    critic_lr: float
    max_seq_len: int
    critic_max_len: int
    weight_decay: float
    critic_weight_decay: float


class MrlStrategy(Enum):
    SINGLE_STEP_STRATEGY = 1
    MULTI_STEP_STRATEGY = 2
    LONG_RANGE_STRATEGY = 3

UnfreezeItem = Union[int, tuple[int, float]]
UnfreezeEpochsStrategy: TypeAlias = Union[int, tuple[UnfreezeItem, UnfreezeItem, UnfreezeItem, int]]

class CurriculumConfig(TypedDict):
    steps: int
    epochs: int
    dataset: MrlCurriculumDataset
    eval_dataset: Optional[MrlCurriculumDataset]
    callbacks: Optional[list[MrlTrainerCallback]]
    strategy: MrlStrategy
    unfreeze_epoch: Optional[UnfreezeEpochsStrategy]
    random_resets: Optional[bool]
    random_resets_from: Optional[int]
    random_resets_ratio: Optional[float]
    reward_model: Optional[MrlRewardModel]
    separate_memory_lr: Optional[bool]
    lr: Optional[float]
    memory_lr: Optional[float]
    critic_lr: Optional[float]
    weight_decay: Optional[float]
    critic_weight_decay: Optional[float]


class SamplerConfig(TypedDict):
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]


class MrlTrajectoryStep(TypedDict):
    state: tuple[TokenizedDict, TokenizedDict, TokenizedDict]
    action: TokenizedDict
    log_probs: torch.Tensor
    reward: list[float]
    reference: TokenizedDict


class MrlTrajectoryEpisode(TypedDict):
    reset_stm: bool
    steps: list[MrlTrajectoryStep]


class MRLTrainer:
    def __init__(
            self,
            actor: MrlActorModel,
            critic: MrlCriticModel,
            reward: MrlRewardModel,
            device: torch.device,
            config: MrlConfig,
            rl_algorithm: RlAlgorithm,
            sampler_config: Optional[SamplerConfig] = None,
            log_dir: str = None,
            pad_token_id: int = 0,
            end_token_id: int = 3,
            use_ddp: bool = False,
            use_amp: bool = False,
            dtype: torch.dtype = torch.float32,
            callbacks: list[MrlTrainerCallback] = None,

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
        self.shared_reward_model = reward
        self.reward = reward
        self.device = device
        self.max_seq_len = config.get('max_seq_len', 256)
        self.critic_max_len = config.get('critic_max_len', 512)

        # Move models to device
        if use_amp:
            self.actor.to(self.device)
            self.critic.to(self.device)
        else:
            self.actor.to(self.device, dtype=dtype)
            self.critic.to(self.device, dtype=dtype)

        # Batch Sampler for answer generation
        self.generator = BatchSampler(self.actor, self.device, end_token_id=end_token_id)
        self.sampler_config = SamplerConfig(
            temperature=1.0,
            top_k=None,
            top_p=None,
        ) if sampler_config is None else sampler_config

        self.pad_token_id = pad_token_id

        self.use_ddp = use_ddp
        self.use_amp = use_amp
        self.dtype = dtype

        self.separate_memory_lr = config.get('separate_memory_lr', False)

        if self.separate_memory_lr:
            self.base_optim_config = {
                'lr': config.get('lr', 3e-4),
                'memory_lr': config.get('memory_lr', 5e-4),
                'critic_lr': config.get('critic_lr', 1e-4),
                'weight_decay': config.get('weight_decay', 0.01),
                'critic_weight_decay': config.get('critic_weight_decay', 0.01),
            }
        else:
            self.base_optim_config = {
                'lr': config.get('lr', 3e-4),
                'critic_lr': config.get('critic_lr', 1e-4),
                'weight_decay': config.get('weight_decay', 0.01),
                'critic_weight_decay': config.get('critic_weight_decay', 0.01),
            }

        self.optim_config = self.base_optim_config

        self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)

        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        self.critic_scaler = torch.amp.GradScaler() if self.use_amp else None

        # TensorBoard Writer
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.global_step = self._init_steps()
        self.epoch_step = self._init_steps()
        self.stage_step = self._init_steps()

        self.rl_algorithm = rl_algorithm

        # Dynamic fields, updated for each curriculum step
        self.curriculum_steps = 0
        self.train_dataset = None
        self.eval_dataset = None
        self.random_resets_ratio = 0.0
        self.strategy = None
        self.shared_callbacks = callbacks if callbacks else []
        self.callbacks = []
        self.global_epoch = 0
        self.global_epochs_count = 0

    def _init_optimizers(
            self,
            lr: float,
            critic_lr: float,
            weight_decay: float,
            critic_weight_decay: float,
            memory_lr: Optional[float] = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        if memory_lr is not None:
            optimizer = torch.optim.AdamW([
                { 'params': self.actor.not_memory_parameters(), 'lr': lr },
                { 'params': self.actor.memory_parameters(), 'lr': memory_lr },
            ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.actor.unique_parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )

        return optimizer, critic_optimizer


    def _init_steps(self):
        return {
            'collect': 0,
            'critic': 0,
            'rl': 0,
            'eval': 0,
        }

    def _increment_steps(self, step_type: str):
        self.global_step[step_type] += 1
        self.epoch_step[step_type] += 1
        self.stage_step[step_type] += 1

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
                inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
                # 3. Encode data and update STM
                self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)
        else:
            # 2. Concatenate batch of queries and answers (they are already on training device)
            inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
            # 3. Encode data and update STM
            self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)

    def generate_answer(self, query: TokenizedDict) -> tuple[TokenizedDict, torch.Tensor]:
        """Generate response using batch sampler with decoder."""
        # 1. Generate answer with BatchSampler - with autocast on/off
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
        # 2. Convert generated answer to TokenizedDict
        generated_answer: TokenizedDict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        return generated_answer, log_probs

    def _calculate_reward(self, generated: TokenizedDict, reference: TokenizedDict,
                          saved_query: TokenizedDict, saved_answer: TokenizedDict,
                          mode: MrlRewardMode = MrlRewardMode.STANDARD,
                          prev_data: tuple[TokenizedDict, TokenizedDict] = None):
        saved_interaction = smart_concat(saved_query, saved_answer, max_length=self.max_seq_len,
                                         pad_token_id=self.pad_token_id)
        prev_data = smart_concat(prev_data[0], prev_data[1], self.max_seq_len,
                                 self.pad_token_id) if prev_data is not None else None
        return self.reward(generated, reference, saved_interaction, mode=mode, prev_data=prev_data), saved_interaction

    def compute_reward(self, generated: TokenizedDict, reference: TokenizedDict,
                       saved_data: tuple[TokenizedDict, TokenizedDict], mode: MrlRewardMode = MrlRewardMode.STANDARD,
                       eval_mode: bool = False, prev_data: tuple[TokenizedDict, TokenizedDict] = None) -> list[float]:
        """Compute reward based on memory retention (e.g., BLEU-4)."""
        saved_query, saved_answer = saved_data
        # 1. Concat saved (previous) interaction and calculate reward using generated sequence, reference and saved data - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                reward, saved_interaction = self._calculate_reward(generated, reference, saved_query, saved_answer,
                                                                   mode=mode, prev_data=prev_data)
        else:
            reward, saved_interaction = self._calculate_reward(generated, reference, saved_query, saved_answer,
                                                               mode=mode, prev_data=prev_data)

        # 2. Run 'on reward' callbacks
        for cb in self.callbacks:
            cb.on_reward(self.actor, reward, generated, reference, saved_interaction, eval_mode)
        # 3. Return rewards for batch
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

    def _collect_writer(self, avg_reward: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Collect/episode reward (global)', avg_reward, self.global_step['collect'])
            self.writer.add_scalar(f'Collect/episode reward (steps: {self.curriculum_steps}, epoch: {epoch})',
                                   avg_reward, self.epoch_step['collect'])
            self.writer.add_scalar(f'Collect/episode reward (steps: {self.curriculum_steps})', avg_reward,
                                   self.stage_step['collect'])

    def collect_trajectories(self, dataloader: DataLoader, epoch: int, batch_size: int) -> list[MrlTrajectoryEpisode]:
        """Collect trajectories for PPO for current curriculum step."""
        # 1. Init trajectories list
        trajectories = []

        with torch.no_grad():
            # 2. Collect episode trajectories for all batches in dataset
            for batch_idx, batch in enumerate(dataloader):
                if batch['query']['input_ids'].size(0) == batch_size:
                    self._increment_steps('collect')
                    # 3. Reset Short-Term Memory state (with random reset ratio - sometimes it will be good to build memory
                    # state from existing one, instead of new random one)
                    reset_done = self.reset_stm()

                    # 4. Reset reward prev data running mean - it's calculated for multi-step retention, we have to reset it before episode
                    self.reward.reset_running_mean()

                    # 5. Get first batch of interactions (data to save) and follow-up interactions for current episode, based on curriculum step
                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                    interactions = interactions[:self.curriculum_steps]
                    interactions_len = len(interactions)
                    # 6. Encode and update STM with data to save from first interaction
                    self.encode_and_update_stm(*self._move_multiple_batches(first_query, first_answer))

                    # 7. Save first interaction as data to save (for trajectory state)
                    query, answer = first_query, first_answer

                    # 8. Run training strategy for follow-up interactions
                    episode_steps = []
                    episode_rewards = []

                    prev_interaction = None

                    for i, interaction in enumerate(interactions):
                        # 9. Generate batch of answers based on batch of follow-up queries
                        next_query = self._move_batch(interaction['query'])
                        generated_answer, log_probs = self.generate_answer(next_query)

                        is_last_interaction = (i + 1) == interactions_len

                        detached_answer = self._cpu_detach(generated_answer)  # detach and keep states on CPU

                        # 10. Depending on strategy compute reward
                        if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                            # a) long-range - first interaction - change topic - negative reward (it shouldn't include saved data)
                            reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer),
                                                         mode=MrlRewardMode.NEGATIVE)
                        elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                            # b) long-range - last interaction - first interaction topic - long-range reward (it should include content from first interaction)
                            reward = self.compute_reward(detached_answer, interaction['answer'],
                                                         (first_query, first_answer), mode=MrlRewardMode.LONG_RANGE,
                                                         prev_data=prev_interaction)
                        else:
                            # c) standard reward - generated answer should include some content from previous interaction (saved data), like reference answer
                            reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer),
                                                         mode=MrlRewardMode.STANDARD, prev_data=prev_interaction)

                        # 11. Update STM with generated response (except last interaction, it's not needed)
                        if not is_last_interaction:
                            self.encode_and_update_stm(next_query,
                                                       generated_answer)  # update with generated_answer on GPU

                        # 12. Store trajectory step
                        trajectory: MrlTrajectoryStep = {
                            'state': (query, answer, interaction['query']),
                            'action': detached_answer,
                            'log_probs': log_probs.detach().cpu(),
                            'reward': reward,
                            'reference': interaction['answer'],
                        }
                        episode_steps.append(trajectory)
                        episode_rewards.append(reward)

                        # 13. Set previous and current interaction query and generated answer (batches), as saved data for next interaction
                        if not (self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0):
                            prev_interaction = (query, answer)
                        query, answer = interaction['query'], detached_answer

                    # 14. Append full batched episode (number of steps depends on curriculum stage) to trajectories
                    episode_trajectory: MrlTrajectoryEpisode = {
                        'reset_stm': reset_done,
                        'steps': episode_steps,
                    }
                    trajectories.append(episode_trajectory)

                    mean_episode_reward = torch.tensor(episode_rewards).mean().item()

                    self._collect_writer(mean_episode_reward, epoch)

                    # 15. Run "on episode collected" callbacks
                    for cb in self.callbacks:
                        cb.on_episode_collected(self.actor, batch_idx, episode_trajectory, mean_episode_reward)

        return trajectories

    def _critic_loss(self, inputs: TokenizedDict, rewards: torch.Tensor) -> torch.Tensor:
        # 1. Calculate values with critic encoder
        values = self.critic(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).squeeze()
        # 2. Calculate critic loss
        loss = self.rl_algorithm.critic_loss(values, rewards)
        return loss

    def _critic_writer(self, critic_loss: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Loss/critic (global)', critic_loss, self.global_step['critic'])
            self.writer.add_scalar(f'Loss/critic (steps: {self.curriculum_steps}, epoch: {epoch})', critic_loss,
                                   self.epoch_step['critic'])
            self.writer.add_scalar(f'Loss/critic (steps: {self.curriculum_steps})', critic_loss,
                                   self.stage_step['critic'])

    def update_critic(self, states: list[tuple[TokenizedDict, TokenizedDict, TokenizedDict]],
                      rewards: list[torch.Tensor], epoch: int):
        """Update critic network using MSE loss."""
        # 1. Run critic updates for all collected batches
        critic_losses = []
        for step_idx, (state, reward) in enumerate(zip(states, rewards)):
            self._increment_steps('critic')
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
                        pad_token_id=self.pad_token_id,
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
                    pad_token_id=self.pad_token_id,
                )
                # Calculate loss
                loss = self._critic_loss(inputs, reward.to(self.device, dtype=self.dtype))
                # Run backpropagation
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=False)
                # Run optimizer step
                self.critic_optimizer.step()
            critic_loss = loss.item()
            self._critic_writer(critic_loss, epoch)

            # 5. Run "on critic updated" callbacks
            for cb in self.callbacks:
                cb.on_critic_updated(self.actor, self.critic, epoch, step_idx, critic_loss)

            # 6. Accumulate loss for epoch callbacks
            critic_losses.append(critic_loss)

        # 7. Calculate mean loss for epoch callbacks
        critic_mean_loss = torch.tensor(critic_losses).mean().item()

        return critic_mean_loss

    def _critic_advantages(self, critic_state: TokenizedDict, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            values = self.critic(critic_state['input_ids'],
                                 attention_mask=critic_state['attention_mask']).squeeze()
        return self.rl_algorithm.calculate_advantages(rewards, values)

    def _rl_writer(self, policy_loss: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Loss/policy (global)', policy_loss, self.global_step['rl'])
            self.writer.add_scalar(f'Loss/policy (steps: {self.curriculum_steps}, epoch: {epoch})', policy_loss,
                                   self.epoch_step['rl'])
            self.writer.add_scalar(f'Loss/policy (steps: {self.curriculum_steps})', policy_loss, self.stage_step['rl'])

    def rl_step(self, trajectories: list[MrlTrajectoryEpisode], epoch: int):
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

            # 3. Run episode steps - each episode has number of steps depending on curriculum stage. Each step is run for all batch
            for step in episode_steps:
                self._increment_steps('rl')
                state, action, reward, log_probs = step['state'], step['action'], step['reward'], step['log_probs']
                query, answer, next_query = self._move_multiple_batches(*state)
                action = self._move_batch(action)
                log_probs = log_probs.to(self.device)
                rewards = torch.tensor(reward).to(self.device)

                # 4. Compute advantages using critic
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        critic_state = smart_concat_critic_states(query, answer, next_query,
                                                                  max_length=self.critic_max_len,
                                                                  pad_token_id=self.pad_token_id)
                        advantages = self._critic_advantages(critic_state, rewards)
                else:
                    critic_state = smart_concat_critic_states(query, answer, next_query, max_length=self.critic_max_len,
                                                              pad_token_id=self.pad_token_id)
                    advantages = self._critic_advantages(critic_state, rewards)

                # 5. Encode and update STM on each step, to include encoder and memory attention gradients in loss
                self.encode_and_update_stm(query, answer)
                # 6. Concatenate next query and action and get action logits from decoder
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                              pad_token_id=self.pad_token_id)
                        logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                            action=MrlActorAction.DECODE)
                else:
                    inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                          pad_token_id=self.pad_token_id)
                    logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                        action=MrlActorAction.DECODE)

                # 7. Calculate RL Algorithm (PPO etc.) loss
                policy_loss = self.rl_algorithm.policy_loss(next_query, action, logits, log_probs, advantages)

                # 8. Reset gradients
                self.optimizer.zero_grad()

                # 9. Update the model in AMP or regular mode
                if self.use_amp:
                    self.scaler.scale(policy_loss).backward(retain_graph=True)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0,
                                                   error_if_nonfinite=False)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    policy_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.unique_parameters(), max_norm=1.0,
                                                   error_if_nonfinite=False)
                    self.optimizer.step()

                policy_loss_item = policy_loss.item()
                self._rl_writer(policy_loss_item, epoch)
                all_losses.append(policy_loss_item)

                # 10. Run "on batch updated" callback
                for cb in self.callbacks:
                    cb.on_batch_updated(self.actor, epoch, self.epoch_step['rl'], policy_loss_item)

        return torch.mean(torch.tensor(all_losses)).item()

    def _critic_states_and_rewards(self, trajectories: list[MrlTrajectoryEpisode]):
        flat_trajectories = [t for episode in trajectories for t in episode['steps']]
        states = [t['state'] for t in flat_trajectories]
        rewards = [torch.tensor(t['reward']) for t in flat_trajectories]
        return states, rewards

    def train_epoch(self, dataloader: DataLoader, epoch: int, batch_size: int):
        """Train for one epoch."""
        # 1. Collect trajectories for current epoch
        trajectories = self.collect_trajectories(dataloader, epoch, batch_size)

        # 2. Flatten trajectories and collect state and rewards for critic update
        states, rewards = self._critic_states_and_rewards(trajectories)
        # 3. Update critic model, based on states and rewards
        critic_loss = self.update_critic(states, rewards, epoch)

        # 4. Run PPO algorithm step
        policy_loss = self.rl_step(trajectories, epoch)

        # 5. Return policy and critic mean losses for epoch callbacks
        return policy_loss, critic_loss

    def _eval_loader(self, batch_size: int):
        if self.use_ddp:
            return DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                pin_memory=True,
                sampler=DistributedSampler(self.eval_dataset, shuffle=False),
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )
        else:
            return DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )

    def _eval_writer(self, avg_reward: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Eval/episode reward (global)', avg_reward, self.global_step['eval'])
            self.writer.add_scalar(f'Eval/episode reward (steps: {self.curriculum_steps}, epoch: {epoch})', avg_reward,
                                   self.epoch_step['eval'])
            self.writer.add_scalar(f'Eval/episode reward (steps: {self.curriculum_steps})', avg_reward,
                                   self.stage_step['eval'])

    def evaluate(self, batch_size: int, epoch: int):
        """Evaluate model on validation dataset."""
        # 1. Init evaluation DataLoader
        dataloader = self._eval_loader(batch_size)
        total_reward = torch.tensor(0.0).to(self.device)
        count = torch.tensor(0).to(self.device)

        # 2. Run evaluation on all batch episodes
        for batch in dataloader:
            with torch.no_grad():
                if batch['query']['input_ids'].size(0) == batch_size:
                    self._increment_steps('eval')
                    # 3. Reset STM with random resets ratio and reward model running mean
                    self.reset_stm()
                    self.reward.reset_running_mean()

                    # 4. Get batches for first queries, answers and all follow-up interactions
                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                    # 5. Encode and update STM with initial interactions (batch)
                    self.encode_and_update_stm(*self._move_multiple_batches(first_query, first_answer))

                    # 6. Save follow-up interactions len and first query and answer as previous one for iteration
                    interactions_len = len(interactions)
                    query, answer = first_query, first_answer
                    episode_reward = torch.tensor(0.0).to(self.device)
                    episode_interactions = torch.tensor(0).to(self.device)

                    prev_interaction = None

                    # 7. Run all follow-up interactions
                    for i, interaction in enumerate(interactions):
                        # 8. Generate batch of answers
                        next_query = self._move_batch(interaction['query'])
                        generated_answer, _ = self.generate_answer(next_query)

                        is_last_interaction = (i + 1) == interactions_len

                        detached_answer = self._cpu_detach(generated_answer)

                        # 9. Depending on current strategy and step, compute reward
                        if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                            reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer),
                                                         mode=MrlRewardMode.NEGATIVE, eval_mode=True)
                        elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                            reward = self.compute_reward(detached_answer, interaction['answer'],
                                                         (first_query, first_answer), mode=MrlRewardMode.LONG_RANGE,
                                                         eval_mode=True, prev_data=prev_interaction)
                        else:
                            reward = self.compute_reward(detached_answer, interaction['answer'], (query, answer),
                                                         mode=MrlRewardMode.STANDARD, eval_mode=True,
                                                         prev_data=prev_interaction)

                        # 10. Encode and update memory for the next interaction
                        if not is_last_interaction:
                            self.encode_and_update_stm(next_query, generated_answer)

                        # 11. Accumulate rewards
                        step_reward = torch.tensor(reward).mean().to(self.device)
                        # total
                        total_reward += step_reward
                        count += 1
                        # episode
                        episode_reward += step_reward
                        episode_interactions += 1
                        # 12. Save previous interaction
                        if not (self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0):
                            prev_interaction = (query, answer)
                        query, answer = interaction['query'], detached_answer
                    avg_episode_reward = (episode_reward / episode_interactions).item()
                    # 13. Run eval TensorBoard writer with average episode reward
                    self._eval_writer(avg_episode_reward, epoch)

                    # 14. Run "on eval episode end" callbacks
                    for cb in self.callbacks:
                        cb.on_eval_episode_end(self.actor, epoch, self.epoch_step['eval'], avg_episode_reward)

        # 15. Calculate average reward
        if self.use_ddp:
            total_sum = dist.all_reduce(total_reward, dist.ReduceOp.SUM)
            count_sum = dist.all_reduce(count, dist.ReduceOp.SUM)
            avg_reward = (total_sum / count_sum).item() if count_sum > 0 else 0
        else:
            avg_reward = (total_reward / count).item() if count > 0 else 0

        should_stop_stage = False
        # 16. Run "on eval end" callbacks
        for cb in self.callbacks:
            should_stop = cb.on_eval_end(self.actor, self.critic, epoch, avg_reward)
            if should_stop:
                should_stop_stage = True

        return should_stop_stage

    def _setup_curriculum_step(self, config: CurriculumConfig) -> tuple[tuple[int, UnfreezeEpochsStrategy], tuple[bool, int, float]]:
        # 1. Set common fields based on config
        self.curriculum_steps = config.get('steps', 1)  # number of steps to run in episode
        self.train_dataset = config.get('dataset', None)  # training dataset for current curriculum stage
        self.eval_dataset = config.get('eval_dataset', None)  # evaluation dataset for current curriculum stage
        self.callbacks = config.get('callbacks',
                                    self.shared_callbacks)  # trainer callbacks for current curriculum stage
        self.strategy = config.get('strategy',
                                   MrlStrategy.MULTI_STEP_STRATEGY)  # MRL strategy for given curriculum stage
        self.reward = config.get('reward_model', self.shared_reward_model)  # MRL Reward Model for curriculum stage
        if config['lr'] is not None or config['critic_lr'] is not None or config['weight_decay'] is not None or config['critic_weight_decay'] is not None or (config['separate_memory_lr'] and config['memory_lr'] is not None):
            if config.get('separate_memory_lr', False):
                self.optim_config = {
                    'lr': config.get('lr', self.base_optim_config['lr']),
                    'critic_lr': config.get('critic_lr', self.base_optim_config['critic_lr']),
                    'weight_decay': config.get('weight_decay', self.base_optim_config['weight_decay']),
                    'critic_weight_decay': config.get('critic_weight_decay', self.base_optim_config['critic_weight_decay']),
                    'memory_lr': config.get('memory_lr', self.base_optim_config['memory_lr']),
                }
            else:
                self.optim_config = {
                    'lr': config.get('lr', self.base_optim_config['lr']),
                    'critic_lr': config.get('critic_lr', self.base_optim_config['critic_lr']),
                    'weight_decay': config.get('weight_decay', self.base_optim_config['weight_decay']),
                    'critic_weight_decay': config.get('critic_weight_decay', self.base_optim_config['critic_weight_decay']),
                }
            self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)
        elif self.optim_config != self.base_optim_config:
            self.optim_config = self.base_optim_config
            self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)




        # 2. Get epochs and random resets configs
        epochs = config.get('epochs', 5)  # number of epochs for current stage
        unfreeze_epoch = config.get('unfreeze_epoch',
                                    0)  # epoch when components (other than memory) are unfrozen (before epoch starts)
        random_resets = config.get('random_resets',
                                   False)  # flag for using random STM resets (recommended, as model should learn transitions between different states)
        random_resets_from = config.get('random_resets_from', None)  # epoch from which random STM resets are started
        random_resets_ratio = config.get('random_resets_ratio',
                                         None)  # ratio of random STM resets - 1.0 is "always reset", 0.0 is "no resets"

        # 3. Reset stage step counter
        self.stage_step = self._init_steps()

        return (epochs, unfreeze_epoch), (random_resets, random_resets_from, random_resets_ratio)

    def _apply_unfreeze_strategy(self, epoch: int, unfreeze_epoch: UnfreezeEpochsStrategy):
        is_staged_unfreeze = isinstance(unfreeze_epoch, tuple)
        if is_staged_unfreeze:
            update_epoch, fetch_epoch, joint_epoch, all_epoch = unfreeze_epoch

            if isinstance(update_epoch, tuple):
                switch_epoch, cross_att_lr = update_epoch
                if epoch == switch_epoch:
                    self.actor.freeze_components('joint')
                    self.optimizer = self._init_unfreeze_optimizer('update', cross_att_lr)
                    print(f"Activating 'update' unfreeze strategy with custom cross_att_lr: {cross_att_lr}")
            elif epoch == update_epoch:
                 self.actor.freeze_components('update')
                 print(f"Activating 'update' unfreeze strategy - mem-att trainable / cross-att frozen / rest model frozen")

            if isinstance(fetch_epoch, tuple):
                switch_epoch, mem_att_lr = fetch_epoch
                if epoch == fetch_epoch:
                    self.actor.freeze_components('joint')
                    self.optimizer = self._init_unfreeze_optimizer('fetch', mem_att_lr)
                    print(f"Activating 'fetch' unfreeze strategy with custom mem_att_lr: {mem_att_lr}")
            elif epoch == fetch_epoch:
                self.actor.freeze_components('fetch')
                print(f"Activating 'fetch' unfreeze strategy - mem-att frozen / cross-att trainable / rest model frozen")

            if isinstance(joint_epoch, tuple):
                switch_epoch, model_lr = joint_epoch
                if epoch == joint_epoch:
                    self.actor.unfreeze_components()
                    self.optimizer = self._init_unfreeze_optimizer('joint', model_lr)
                    print(f"Activating 'joint' unfreeze strategy with custom model_lr: {model_lr}")
            elif epoch == joint_epoch:
                    self.actor.freeze_components('joint')
                    print(f"Activating 'joint' unfreeze strategy - mem-att/cross-att trainable / rest model frozen")
            if epoch == all_epoch:
                self.actor.unfreeze_components()
                self.optimizer = self._init_unfreeze_optimizer('all', 0.)
                print(f"Switching to train 'all' strategy - unfreeze all components")
        elif epoch == unfreeze_epoch:
            self.actor.unfreeze_components()
            print(f"Switching to train 'all' strategy - unfreeze all components")

    def _init_unfreeze_optimizer(
            self,
            mode: Literal['update', 'fetch', 'joint', 'all'],
            unfreeze_lr: float,
    ) -> torch.optim.Optimizer:
        memory_lr = self.optim_config['memory_lr'] if 'memory_lr' in self.optim_config else self.optim_config['lr']
        model_lr = self.optim_config['lr']

        if mode == 'update':
            params = [
                {'params': self.actor.not_memory_parameters(), 'lr': model_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': memory_lr},
                {'params': self.actor.memory_cross_attention_parameters(), 'lr': unfreeze_lr},
            ]
        elif mode == 'fetch':
            params = [
                {'params': self.actor.not_memory_parameters(), 'lr': model_lr},
                {'params': self.actor.memory_cross_attention_parameters(), 'lr': memory_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': unfreeze_lr},
            ]
        elif mode == 'joint':
            params = [
                {'params': self.actor.not_memory_parameters(), 'lr': unfreeze_lr},
                {'params': self.actor.memory_parameters(), 'lr': memory_lr},
            ]
        else:
            params = [
                {'params': self.actor.not_memory_parameters(), 'lr': model_lr},
                {'params': self.actor.memory_parameters(), 'lr': memory_lr},
            ]

        return torch.optim.AdamW(params, weight_decay=self.optim_config['weight_decay'])


    def __call__(self, curriculum_config: list[CurriculumConfig], batch_size: int):
        """Start Memory Reinforcement Learning Curriculum."""

        # 0. Set global epoch count for all stages
        self.global_epochs_count = sum(stage['epochs'] for stage in curriculum_config)
        self.global_epoch = 0

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

            # 4. Freeze all components except memory attention and memory cross-attention layers in decoder/encoder
            if unfreeze_epoch != 0:
                self.actor.freeze_components('joint')
                if isinstance(unfreeze_epoch, tuple):
                    print(f"Starting training with unfreeze strategies - 'warmup' - mem-att/cross-att trainable / rest model frozen")
                else:
                    print(f"Starting training with simple unfreeze - 'joint' - mem-att/cross-att trainable / rest model frozen")

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
                # 7. Increment global epoch
                self.global_epoch += 1
                # 8. Run "on epoch start" callbacks (log info, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_start(self.actor, epoch, epochs, current_curriculum_step, self.global_epoch,
                                      self.global_epochs_count)

                # 9. Reset steps counter for epoch
                self.epoch_step = self._init_steps()

                # 10. Set random STM resets ratio from selected epoch
                if random_resets and random_resets_from <= epoch:
                    self.random_resets_ratio = random_resets_ratio
                else:
                    self.random_resets_ratio = 1.0

                # 11. Apply the unfreeze strategy
                self._apply_unfreeze_strategy(epoch, unfreeze_epoch)

                # 12. Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                # 13. Run reinforcement learning algorithms for current epoch
                policy_loss, critic_loss = self.train_epoch(dataloader, epoch, batch_size)

                # 14. If evaluation dataset is provided, run evaluation steps
                if self.eval_dataset:
                    should_stop_stage = self.evaluate(batch_size, epoch)
                else:
                    should_stop_stage = False

                # 15. Finally, run "on epoch end" callbacks (save models, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_end(self.actor, epoch, epochs, policy_loss, critic_loss, self.global_epoch,
                                    self.global_epochs_count)

                # 16. Synchronize TensorBoard writer
                if self.writer:
                    self.writer.flush()

                # 17. Synchronize devices in DDP mode
                if self.use_ddp:
                    dist.barrier()

                # 18. Finish curriculum stage if rewards are not increased or reached threshold point
                if should_stop_stage:
                    break

            # 19. Run "on_training_end" callbacks after each curriculum stage (they have own callbacks)
            for cb in self.callbacks:
                cb.on_training_end(self.actor, self.critic, current_curriculum_step)

        # 20. Training end - finish processes after all curriculum stages
        if self.use_ddp:
            dist.destroy_process_group()

        # 21. Close writer
        if self.writer:
            self.writer.close()
