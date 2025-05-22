import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, Union, TypedDict
from enum import Enum
import random
from ..transformers.sampler import BatchSampler, BatchSampleDecoder
from ..training.callbacks import MrlTrainerCallback
from ..training.dataset import MrlCurriculumDataset
from ..training.mrl import MrlActorModel, MrlCriticModel, MrlRewardModel, MrlConfig, CurriculumConfig, SamplerConfig, MrlStrategy, MrlActorAction


class MRLTrainerTextBased:
    """
    TextBased implementation of MRL Trainer, made to compare the performance. Base version is using tokenized dataset,
    and is working on token ids in environment state, while the TextBased version has text dataset, keep states in strings
    and list of strings, and is tokenizing data inside Trainer.
    Pros:
        - a lot easier merging/concatenation of queries and answers
        - less memory for environment states
    Cons:
        - performance - require tokenize/de-tokenize step for each sequence, while base version could be pre-tokenized once
        - converting generated answers to strings, only to tokenize them back for calculations


    TODO: Datasets, correct PPO algorithms and other utils
    """
    def __init__(
            self,
            actor: MrlActorModel,
            critic: MrlCriticModel,
            reward: MrlRewardModel,
            device: torch.device,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            config: MrlConfig,
            sampler_config: SamplerConfig,
            log_dir: str = None,
            end_token_id: int = 3,
    ):
        """
        TextBased (Tokenization inside) Trainer for Memory Reinforcement Learning (MRL) in Reactive Transformer.

        Args:
            actor: MRL Actor model with encoder, decoder and memory attention.
            critic: Critic network for advantage estimation.
            config: Configuration dictionary with hyperparameters.
            tokenizer: Tokenizer for text processing.
        """
        self.actor = actor
        self.critic = critic
        self.tokenizer = tokenizer
        self.reward = reward
        self.device = device
        self.max_seq_len = config.get('max_seq_len', 1024)
        self.critic_max_len = config.get('critic_max_len', 2048)

        # Move models to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # Batch Sampler for answer generation
        sampler = BatchSampler(self.actor, self.device, end_token_id=end_token_id)
        self.generator = BatchSampleDecoder(sampler, self.tokenizer)
        self.sampler_config = sampler_config

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

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Dynamic fields, updated for each curriculum step
        self.curriculum_step = 0
        self.train_dataset = None
        self.eval_dataset = None
        self.random_resets_ratio = 0.0
        self.long_range_ratio = 0.0
        self.callbacks = []


    def reset_stm(self):
        """Reset Short-Term Memory state."""
        if self.random_resets_ratio == 1.0:
            self.actor.reset_memory()
        else:
            rng = random.random()
            if rng <= self.random_resets_ratio:
                self.actor.reset_memory()

    def encode_and_update_stm(self, query: list[str], answer: list[str]):
        """Encode interaction and update STM."""
        # 1. Tokenize query and answer in interaction mode (two text inputs)
        inputs = self.tokenizer(
            query,
            answer,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

        # 2. Encode data and update memory
        self.actor(inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device), action=MrlActorAction.UPDATE)

    def generate_answer(self, query: list[str]) -> tuple[list[str], torch.Tensor]:
        """Generate response using decoder with current STM."""
        return self.generator.generate_with_log_probs(query, **self.sampler_config)

    def compute_reward(self, generated: list[str], reference: list[str]) -> float:
        """Compute reward based on memory retention (e.g., BLEU-4)."""
        reward = self.reward(generated, reference)
        for cb in self.callbacks:
            cb.on_reward(self.actor, generated, reference, reward)
        return reward

    def get_strategy(self):
        """Get MRL strategy for current episode."""
        if self.is_first_step:
            return MrlStrategy.SINGLE_STEP_STRATEGY
        else:
            rng = random.random()
            if rng <= self.long_range_ratio:
                return MrlStrategy.LONG_RANGE_STRATEGY
            else:
                return MrlStrategy.MULTI_STEP_STRATEGY

    def collect_trajectories(self, batch_size: int) -> list[list[dict]]:
        """Collect trajectories for PPO for current curriculum step."""
        # 1. Init PyTorch DataLoader and trajectories list
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        trajectories = []

        # 2. Collect episode trajectories for all batches in dataset
        for batch_idx, batch in enumerate(dataloader):
            # 4. Reset Short-Term Memory state (with random reset ratio - sometimes it will be good to build memory
            # state from existing one, instead of new random one
            self.reset_stm()

            # 5. Get first interaction (data to save) and follow-up interactions for current episode, based on curriculum step
            first_query, first_answer, interactions = batch["query"], batch["answer"], batch["interactions"]
            interactions = interactions[:self.curriculum_step]
            interactions_len = len(interactions)
            # 6. Encode and update STM with data to save from first interaction
            self.encode_and_update_stm(first_query, first_answer)

            # 7. Get MRL Strategy for current episode
            strategy = self.get_strategy()

            # 8. Save first interaction as data to save (for trajectory state)
            query, answer = first_query, first_answer
            # 9. Run training strategy for follow-up interactions
            episode_trajectories = []
            for i, interaction in enumerate(interactions):
                # 10. Generate answer based on follow-up query
                generated_answer, log_probs = self.generate_answer(interaction["query"])

                is_last_interaction = (i + 1) == interactions_len

                # 11. Depending on strategy compute reward
                if strategy is MrlStrategy.SINGLE_STEP_STRATEGY or strategy is MrlStrategy.MULTI_STEP_STRATEGY or is_last_interaction:
                    # 11a. Compute reward for all steps in SINGLE STEP and MULTI STEP strategies, and last reward for LONG RANGE
                    reward = self.compute_reward(generated_answer, interaction["answer"])
                else:
                    ## 11b. Skip reward calculation for middle interactions in LONG RANGE mode
                    reward = 1.0

                # 12. Update STM with generated response (skip for last interaction, it won't be used)
                if not is_last_interaction:
                    self.encode_and_update_stm(interaction["query"], generated_answer)

                # 14. Store trajectory
                trajectory = {
                    "state": (query, answer, interaction["query"]),
                    "action": generated_answer,
                    "log_probs": log_probs,
                    "reward": reward,
                    "reference": interaction["answer"],
                }
                episode_trajectories.append(trajectory)
                # 15. Set current interaction query and generated answer, as saved data for next interaction
                query, answer = interaction['query'], generated_answer
            # 16. Append current episode to trajectories
            trajectories.append(episode_trajectories)

            # 17. Run "on episode collected" callbacks
            for cb in self.callbacks:
                cb.on_batch_collected(self.actor, batch_idx, trajectories)

        return trajectories

    def update_critic(self, states: list[str], returns: torch.Tensor, epoch: int):
        """Update critic network using MSE loss."""
        # 1. Tokenize input states
        inputs = self.tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.critic_max_len,
        ).to(self.device)
        # 2. Reset critic gradients
        self.critic_optimizer.zero_grad()
        # 3. Calculate values with critic encoder
        values = self.critic(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).logits.squeeze()
        # 4. Calculate critic loss, run backpropagation and optimizer step
        loss = nn.MSELoss()(values, returns)
        loss.backward()
        self.critic_optimizer.step()
        critic_loss_item = loss.item()
        # 5. Run "on critic updated" callbacks
        for cb in self.callbacks:
            cb.on_critic_updated(self.actor, self.critic, epoch, critic_loss_item)
        return critic_loss_item

    def ppo_step(self, trajectories: list[list[dict]]):
        """Perform PPO update step using trajectories."""
        # 1. Run update separately for episodes in trajectory - we have to reset memory before each episode, and update
        # memory, based on collected episode data
        all_losses = []
        trajectories_len = len(trajectories)
        for episode_idx, episode in enumerate(trajectories):
            # 2. Convert episode to PPO-compatible format
            critic_states = [f"[Q]{s['state'][0]}[A]{s['state'][1]}[EOS][BOS][Q] {s['state'][2]}" for s in episode]
            states = [(s['state'][0], s['state'][1], s['state'][2]) for s in episode]
            actions = [s["action"] for s in episode]
            rewards = torch.tensor([s["reward"] for s in episode], device=self.device)

            # 3. Reset memory for current episode
            self.reset_stm()

            # 4. Compute advantages using critic
            with torch.no_grad():
                inputs = self.tokenizer(critic_states, return_tensors="pt", padding=True, truncation=True, max_length=self.critic_max_len).to(self.device)
                values = self.critic(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits.squeeze()
                advantages = rewards - values

            # 5. Policy gradient update (simplified)
            action_logits = []
            for state, action in zip(states, actions):
                query, answer, next_query = state
                # 6. Encode and update STM on each step, to include encoder and memory attention gradients in loss
                self.encode_and_update_stm(query, answer)
                # 7. Tokenize next query and action (generated answer) with interaction tokenizer (two text inputs)
                inputs = self.tokenizer(next_query, action, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len).to(self.device)
                # 8. Use decoder, with updated STM state and get action logits
                logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.DECODE)
                action_logits.append(logits)

            # 9. Calculate PPO loss
            # Placeholder for actual PPO loss calculation
            # This would involve computing log_probs, ratios, and clipped surrogate loss
            policy_loss = -torch.mean(torch.stack(action_logits).squeeze() * advantages)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            policy_loss_item = policy_loss.item()
            all_losses.append(policy_loss_item)

            # 10. Run "on episode updated" callback
            for cb in self.callbacks:
                cb.on_episode_updated(self.actor, episode_idx, trajectories_len, policy_loss_item)

        return torch.mean(torch.tensor(all_losses)).item()

    def train_epoch(self, batch_size: int, epoch: int):
        """Train for one epoch."""
        # 1. Collect trajectories for current epoch
        trajectories = self.collect_trajectories(batch_size)

        # 2. Flatten trajectories and collect state and rewards for critic update
        flat_trajectories = [t for episode in trajectories for t in episode]
        states = [f"[Q]{t['state'][0]}[A]{t['state'][1]}[EOS][BOS][Q] {t['state'][2]}" for t in flat_trajectories]
        rewards = torch.tensor([t["reward"] for t in flat_trajectories], device=self.device)
        # 3. Update critic model, based on states and rewards
        critic_loss = self.update_critic(states, rewards, epoch)

        # 4. Run PPO algorithm step
        policy_loss = self.ppo_step(trajectories)


    def evaluate(self, batch_size: int, epoch: int):
        """Evaluate model on validation dataset."""
        # 1. Init evaluation DataLoader
        dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        total_reward = 0
        count = 0

        # 2. Run evaluation on all batch episodes
        for batch in dataloader:
            for episode_idx, episode in enumerate(zip(batch['query'], batch['answer'], batch['interactions'])):
                with torch.no_grad():
                    self.reset_stm()

                    first_query, first_answer, interactions = episode
                    self.encode_and_update_stm(first_query, first_answer)

                    for interaction in interactions:
                        generated = self.generate_answer(interaction['query'])
                        reward = self.compute_reward(generated, interaction['answer'])
                        total_reward += reward
                        count += 1
                        self.encode_and_update_stm(interaction['query'], generated)

        avg_reward = total_reward / count if count > 0 else 0
        for cb in self.callbacks:
            cb.on_eval_end(self.actor, epoch, avg_reward)

    def __call__(self, curriculum_config: list[CurriculumConfig], batch_size: int):
        """Start Memory Reinforcement Learning Curriculum."""

        # 1. Run each curriculum step based on config
        for current_curriculum_step in curriculum_config:
            # 2. Setup training config for curriculum step
            epochs = current_curriculum_step.get('epochs', 5)
            self.curriculum_step = current_curriculum_step.get('steps', 1)
            print(f'Curriculum Steps Increased to {self.curriculum_step}')
            self.train_dataset = current_curriculum_step.get('dataset', None)
            assert self.train_dataset is not None
            self.eval_dataset = current_curriculum_step.get('eval_dataset', None)
            self.callbacks = current_curriculum_step.get('callbacks', [])
            random_resets = current_curriculum_step.get('random_resets', False)
            random_resets_from = current_curriculum_step.get('random_resets_from', None)
            random_resets_ratio = current_curriculum_step.get('random_resets_ratio', None)
            self.long_range_ratio = current_curriculum_step.get('long_range_ratio', 0.0)
            self.is_first_step = current_curriculum_step.get('is_first_step', False)

            # 3. Freeze all components except memory attention and memory cross-attention layers in decoder/encoder
            self.actor.freeze_components()

            # 4. Run selected number of epochs for given curriculum stage
            for epoch in range(epochs):
                # 5. Run "on epoch start" callbacks (log info, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_start(self.actor, epoch, epochs)

                # 6. Set random STM resets ratio from selected epoch
                if random_resets and random_resets_from <= epoch:
                    self.random_resets_ratio = random_resets_ratio
                else:
                    self.random_resets_ratio = 1.0

                # 7. Unfreeze all components before selected epoch
                if epoch == current_curriculum_step.get('unfreeze_epoch', 0):
                    self.actor.unfreeze_components()

                # 8. Run reinforcement learning algorithms for current epoch
                self.train_epoch(batch_size, epoch)

                # 9. If evaluation dataset is provided, run evaluation steps
                if self.eval_dataset:
                    self.evaluate(batch_size, epoch)

                # 10. Finally, run "on epoch end" callbacks (save models, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_end(self.actor, epoch, epochs)

