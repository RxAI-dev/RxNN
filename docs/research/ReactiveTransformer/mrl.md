# Reactive Transformer: Memory Reinforcement Learning for Attention-Based Memory System
Draft by Adam Filipek/Reactive AI (adamfilipek@rxai.dev)

> After MRL experiments this draft is little outdated - it will be updated soon

## Abstract
**Reactive Transformer (RxT)**, the world's first **Reactive Language Model (RxLM)** architecture, is a breakthrough in the way artificial
intelligence works, as well as in its computational cost. Real-time, stateful processing of individual interactions and transferring context
to dedicated memory layers is a crucial milestone for upcoming awareness models and AGI, but training such an advanced system is a much bigger
challenge than training stateless **Large Language Models (LLM)**. The initial, supervised training stages were designed to maximize preparation
for the final stages of reinforcement learning. These stages alone are insufficient to achieve the full functionality of an **Attention-Based Memory System (ABMS)**,
as supervised training cannot replicate the full dynamics of **Reactive Neural Networks (RxNN)**. The concepts of **Event-Driven AI** and **RxNN**
implemented in **RxT** are strongly linked to the theoretical foundations of reinforcement learning – the model operates according to the principles
of a _Markov chain_, with each subsequent interaction dependent on the previous state.

In this research, we present "Memory Reinforcement Learning", the most important step in training **RxLMs**, that transforms a "weak" supervised
memory system into a fully functional **ABMS**, together with the **Implicit Memory Policy Optimization (IMPO)** algorithm, which is an extension
of **PPO** dedicated to memory learning.

### Memory Reinforcement Learning (MRL)
**Memory Reinforcement Learning** is a crucial step for all the reactive models, as it finally training models for inter-sequence memory
retention. It's based on _Curriculum Learning_ concept, starting from single topic multistep retention, up to long-range multi-topic retention.
**Short-Term Memory (STM)** is treated as model's internal state, that's not observable by the RL environment - its influence is implicitly
deducted from processed interactions. Then, rewards and advantages are not calculated from STM states, but from input/output sequences (interactions).

It's inspired by RLHF, but it has different goals and rewards - remember the previous interactions, instead of more human-like text processing in RLHF.

#### MRL Training Curriculum
The flow of Memory Reinforcement Learning is rather simple and has a little reversed order. We are starting from passing a complete interaction
with question and answer about data that we want to save in memory, to the components responsible for update: encoder and memory-attention layers.
Encoder will transform the interaction with data to its abstract representation in the memory latent spaces, then the results are passed to each
memory attention network's layer, to be combined with previous layer's STM state . New updated data is then dynamically combined with current STM
state by Residual Gate - memory should be updated (of course not at the beginning of training).

Then, the process depends on the curriculum step:
- on the start, it's a single interaction retention, so decoder takes the question about saved data as an input and have to generate the response,
  that's based on those data - it will be used to calculate rewards (based on number of informational content from saved data in generated answer)
- in the Nth step, there are 2 training strategies:
  - _Cumulative Retention_ - each from Nth processed interactions is connected to the same topic, extending it in next steps, with immediate rewards,
    that are calculated in each step, and the one delayed final reward, calculated in last step from question asking of all data from previous steps
  - _Long-term Retention_ - decoder could be used Nth times to process an interaction with a topic not related to saved data, and only the last
    question is connected to saved data. Reward is delayed and calculated only for the last step, so the model is learning to remember data after
    a lot of not connected steps
- both strategies should be mixed, because they are useful for different reasons - first one, with delayed reward, is training model to be able to
  go back to some very old steps, while second approach is training model in progressive modifications of the same topic memories.

In first curriculum steps, only the memory cross-attention and memory-attention layers should be trained, all the other parts of model(s) should be frozen.
After model will start getting some real rewards, it could be slowly unfrozen, to fit the other parts to new layers.

So, in each episode the first full interaction is passed only to encoder and memory attention (it acts like its response is generated
by the decoder). Then all the following interactions are processed by all components - query/question is passed to decoder, that have
to generate the answer, based on the previous interactions from episode - just like in regular chat.

First experimental reactive models should be able to accomplish 16 MRL Curriculum Steps, then it may be increased to 32 steps for production-ready models

> ### Update: Abandoning Single-Step Strategy
> During first tests, we noticed that when using only the single step, not all the model can learn correctly:
> - we are starting from encoding initial interaction and updating memory
> - before the first step, there's memory reset - it starts from random normal distribution
> - in first encoding, encoder's memory cross-attention has access only to initial random STM state
> - then memory is updated by memory-attention and decoder's cross-attention has access to correctly updated memory
> 
> So, in single step, the encoder's memory cross-attention cannot learn, because it has no access to any meaningful signals.
> 
> Additionally, with Single-Step Strategy, every step is final (terminal), so in PPO it's using only simplified GAE calculation.
> It seems, that we should start from 2 or even from 4 steps - we will check what's better here.

> ##### Increasing Curriculum Steps
> Curriculum steps could be increased linearly (1, 2, 3, ..., 15, 16), exponentially (1, 2, 4, 8, 16) or with custom
> schedule (like 1, 2, 3, 4, 6, 8, 12, 16). Exponential or custom schedules should be recommended, as it require simpler and
> smaller dataset, and is more challenging for the model. Linear growth may be too easy in late steps (remembering 13 vs 14
> interactions shouldn't be a big difference), causing model to waste some time on learning tasks on same difficulty level,
> but the exponential growth may be too hard on the other hand (8 vs 16 steps to remember is a big difference). Then, custom
> schedule (like one from example) could be a good balance.

##### Memory resets
By default, we have to reset the Short-Term Memory to its initial (random) state, before each episode. In practice, however,
reactive models will often start with pre-trained STM states that are completely different from the initial ones, so we
should take this into account in training. After some initial steps starting from random normal (0.02 variance) STM, we
could switch to random resets with some probability - model should learn how to build memory states from different starting
points.

#### MRL Datasets
Datasets for the Memory Reinforcement Learning are based on sequences (lists) of connected interactions, depending on the
curriculum step and training strategy. First interaction (not processed by decoder) is the most important in case of the data
to save, while following interactions are either exploring the topic from first one or changing it, and going back to it at the end.
All the interactions in the dataset (not only first) are including both query and response - only query is processed by the model (except
first interaction), but response from dataset could be used in some variants of rewards calculations, as a ground truth.

This kind of dataset is something new, there aren't existing ones, so we have to create them from scratch - in first experiments,
they will be based on synthetic data. Thanks to this we can ensure custom structure - each interaction should be divided into
query and response parts, for easier processing in MRL.

> MRL Datasets contain sequences of interactions representing conversation. Then, they could be similar to chat/dialog datasets
> and could be derived from those kinds of datasets. We just have to ensure, that the dialog content is corresponding with
> MRL goals - concentrated on memory retention, and they have to be divided into single interaction

##### Datasets format
MRL Datasets should include separate subset for each curriculum step. Every example should have `query` and `answer` fields,
for first interaction (and `think` for reasoning models), that will be passed only to encoder and memory attention. Additionally,
it will have `interactions` field, that will be the list of follow-up interactions in the same format - `query`/`answer`/`think`.

#### RL Environment Details
MRL Environment is based on model's interactions:
- saved data and input queries are the environment states - saved data could be just the first interaction (in Single/Long-Term
  Retention modes) or any previous interaction (in Cumulative Retention mode)
- model responses are the agent actions
- rewards are based on memory retention - number of information about previous interactions in model responses (actions)

That makes MRL similar to RLHF, with different rewards, but it's still based on dialog-like interaction. Access to Short-Term
Memory is implicit, through memory cross-attention and memory attention layers in model, so the STM states aren't included
in the environment state.

Collected trajectories includes current environment state (input query), sequence of actions (generated response) and calculated reward.

#### Rewards
MRL awards are mainly based on the model's ability to generate interconnected answers to questions, based on information
from previous interactions, in the form of a fluid dialogue, but the quality of the dialogue itself is not the main focus
of the evaluation (this is the focus of the next stage - RxRLHF). Rewards are always calculated using saved data (interaction),
reference answers and generated response/answer (combined, they are second interaction), but they can represent different parts
of environment state, depending on curriculum step and strategy:
- for single step retention, it's just a first interaction and first query/response
- for long-term retention, it's the first interaction and last query/response
- for cumulative retention, it's previous interaction (including answer generated by model) and current query/response. We
  could also set a step greater than 1, then it will be Nth previous interaction and current query/response

#### RL Algorithms
When we have defined environment, it's time to connect the Reinforcement Learning algorithms and model's loss calculation, to
enable learning. We are basing on Advantage Actor Critic (A2C) algorithm variants, especially Proximal Policy Optimization (PPO),
which is one of the industry standards for RL and language models.

##### Critic network
For MRL's PPO the critic network has to be text based, so it will be the Transformer encoder with a _Value head_ (returning single
predicted number). It has to predict the advantage, based on the environment state - saved data and question about this data.

Critic could be _memory-aware_ - based on pre-trained (on MLM) **Reactive Transformer Encoder** with implicit access to STM
through memory cross-attention, that will learn in MRL process just as in Actor. It could help it to calculate the advantages.
Otherwise, critic could use **Classic Transformer Encoder** too. As we have pre-trained encoder from the previous stages, it
could be reused for the Critic network, either with or without memory cross-attention (with frozen memory cross-attention, it
acts like the regular transformer)

> As _Critic_ has to process both saved data (full interaction) and question from some next interaction, it should have increased
> context length

#### Implementation
Our **RxNN** library (based on **PyTorch**) is already integrated with **HuggingFace** ecosystem - `transformers`, `tokenizers`,
`datasets` and `huggingface-hub`, but MRL is rather not compatible with basic `trl`, because it expects classic transformer
decoders. We provide our custom implementation. Our datasets (connected to specialized trainer) are extending **PyTorch Datasets**
and some features from **HuggingFace Datasets**, especially for loading them from [Hub](https://huggingface.co).
Trainers are specialized for training tasks, like autoregressive/masked language modeling - for MRL there's separate module `rxnn.training.mrl`.

### Research in progress