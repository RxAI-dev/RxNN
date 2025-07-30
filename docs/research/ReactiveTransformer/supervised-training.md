# Reactive Transformer: Supervised Training of Language Components and Attention-Based Memory System
Draft by Adam Filipek/Reactive AI (adamfilipek@rxai.dev)

> This is the second article from Reactive Transformer series and is introducing supervised training stages. For the first
> article - architecture introduction, [check draft](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
>
> In next article, we will describe [Reinforcement Learning stages](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/mrl.md)

## Abstract
Our **Reactive Transformer** architecture is introducing real-time processing of single interactions with dedicated short-term
memory layers for the model context. While it's based on the **Transformer** architecture, it completely redefines how the
model handle conversations - processing single messages with access to STM, that's updated between interactions, instead of
processing full conversation history everytime, like LLM. In result, **Reactive Language Models (RxLM)** are about `N` times
faster and cheaper in inference, where `N` is the number of interactions/messages in the conversation. However, the architecture
is a lot more complex than typical decoder-only transformer - it contains decoder, encoder and memory attention network, called
components. Additional challenge is the asynchronous nature of memory updates - it's done after generating the answer, when
most of the training is based on generated sequences. Fortunately, first stages based on masked and autoregressive language
modelling are similar to original encoder-decoder transformer training. Then, when memory components are included, it's becoming
more complex.

In this article we will describe the supervised stages of **Reactive Transformer** - pre-training and fine-tuning of language
components (decoder and encoder), and memory components pre-training.

## Training stages
Generally, **Reactive Transformer** training has six stages - four supervised stages and two reinforcement learning stages:
1. Joint LM Pre-Training for encoder and decoder, on autoregressive language modeling and masked language modeling at once
2. Joint LM Interaction Supervised Fine-Tuning (SFT) for encoder and decoder
3. Memory Attention Self-Supervised Pre-Training
4. Supervised Memory-Aware Training
5. Memory Reinforcement Learning (MRL) for Short-Term Memory
6. Reinforcement Learning from Human Feedback for Reactive Models (RxRLHF)

First two stages are concentrated on decoder and encoder language modelling and connecting vector spaces between component's
layers - it's based on modified original encoder-decoder training formula. Encoder layers results are saved as **Short-Term
Memory** and used by decoder's **Memory Cross-Attention**. The difference is that encoder is using masked input sequences, has
separate MLM loss calculated and its layers results are detached before saving as memory.

Third stage is for memory attention and memory updates. It's connecting memory attention layers to the same vector spaces
as in language components and is training the model to combine different encoder's results.

When we have pre-trained encoder, decoder and memory attention, in fourth stage we are going back to supervised fine-tuning.
In this stage encoder and memory attention could be trained with decoder or frozen (with gradual unfreezing), but now decoder's
cross-attention input is not from the same data as decoder's main input (as before) - it's combined state of the previous
interactions. It of course require, that decoder's main input is some follow-up interaction, connected with previous ones stored in STM.

Both third and fourth stages are using **Memory Reinforcement Learning** curriculum datasets, based on series of interconnected
interactions.

## Joint Language Models Pre-Training
First training stage is traditionally made for base language modelling. We have to train the model for basic knowledge on
big text corpora - it's using exactly the same datasets as for LLMs pre-training. In this stage we are treating the model
as original encoder-decoder transformer, but it's trained used two objectives - Masked Language Modelling (MLM) and Autoregressive
Modelling. It requires including additional MLM head for encoder that's not a part of the final model. Both encoder and
decoder are using the same input sequences, but for the encoder it's masked. Results of each encoder's layer are detached,
saved as Short-Term Memory and accessed by decoder's Memory Cross-Attention.

The algorithm flow:
1. Get batch of inputs from datasets
2. Process masked inputs with encoder and it results with MLM head
3. Detach encoder layers results and save it as STM
4. Process inputs with decoder, using saved encoder layers results as **Memory Cross-Attention** inputs
5. Calculate MLM cross-entropy loss from encoder/MLM head logits
6. Calculate autoregressive cross-entropy loss from decoder's logits
7. Add both losses, run backpropagation and optimizer step

This training results in very low losses and high accuracy (~90-95%) for decoder, because it has explicit, not masked access
to all next tokens with cross-attention. It may result on too much concentration on cross-attention and weak self-attention,
so it may be good to run some part of the training with decoder only and frozen cross-attention to reach stronger self-attention
before. Generally that behavior is incorrect for **Reactive Transformer**, because finally it should access previous states,
not the current one, but it connects the vector spaces between components and will be refined in later stages.

> In our **RxNN framework** this stage is implemented in:
> - `JointLMTrainer` from `rxnn.training.bml`
> - `JointLMDataset` from `rxnn.training.dataset`
> - `JointTrainingModel` from `rxnn.training.models`

### Joint Interaction Supervised Fine-Tuning (SFT)
In this stage, we have to fit the models (both decoder and encoder) to process the data in single interaction format - question
and answer. We have to provide the examples of interactions from the start of conversation and from the middle of conversation,
but it doesn't require any connections between consecutive interactions. Each example interaction could be independent, but there
should be examples that looks as taken from different stage of conversation, like follow-up questions or topic changes. The goal
is to tune the model to basic conversational interaction format and keep connections between questions and answers. The inter-sequence
memory dependencies are added in the next stages.

The algorithm is exactly the same as for the first pre-training stage, only the dataset and the tokenization inside it are
different - now we have to include special query/answer tokens and prepare inputs from query/answer pairs.

> In **RxNN*, SFT stage is handled by the same `JointLMTrainer` and `JointTrainingModel`, only the `JointLMDataset` is
> replaced by `JointSftDataset` (from `rxnn.training.dataset`)

#### Interaction SFT Datasets
Dataset for Supervised Fine-Tuning for Reactive Transformer includes simple Question-Answer (Query-Response) interactions, from
different stages of dialog/conversation. They don't need to be connected, because they will be just shuffled in different epochs.

Single record - interaction - could be a single text sequence in format `[Q] User's question [A] Model's answer` or it could
be divided into two separate fields. First option is simpler, but the second is easier to customize, so it's recommended.

### Memory Attention Self-Supervised Pre-Training
The main challenge for supervised training of memory attention network is the fact that memory state is not human-interpretable,
so we don't have the idea how it should look like and cannot create labels. But the main feature of memory attention is to
combine data from two different sources - current STM state and new encoded interaction. We could use some arbitrary weighted
mean of both input data sources as a self-supervised labels and maximize their cosine similarity with memory attention results.
The weighted mean factors depends on step:
- starting from random STM state and first interaction, we want the result to include about 90% of the first interaction
- in all next steps it should go in opposite direction - only 5-50% for new encoded interaction and the rest for accumulated state
- in each follow-up step, the factor for new encoded data should be reduced (depending on total number of steps)
- each step result, is one of the inputs for the next step - accumulated memory state

To increase randomness we could add some random noise to inputs or labels - it should result in a little more diversified
combinations instead of static ones based only on factors.

Algorithm flow (require pre-trained and fine-tuned encoder):
1. Starting from random (normal) STM state, encode first interaction
2. Process encoded interaction with memory attention, using random STM state
3. Create self-supervised labels from random STM state and encoded interaction - weighted mean with factors like 0.9 for new encoded data
4. Calculate cosine similarity between labels (weighted mean) and memory attention results, use it's negative value as a loss and run optimization step
5. For `N` steps:
   - use previous memory attention result as STM state and encode next interaction
   - process next encoded interaction with memory attention, using previous results as STM state
   - create self-supervised labels, but now with different factors - `0.5 - N * step_decay` as new data factor and `1.0 - new_data_factor` for accumulated STM state
   - calculate cosine similarity, use it's negative value as loss and run optimization step

This will of course result in very weak memory updates, that are not based on data that's really needed for real interactions,
but it will be then refined in **Memory Reinforcement Learning**. This stage is designed mainly to fix the cold start problem
in MRL - randomly initialized memory attention completely breaks generated answers. In tests using only the previous encoded
interaction as decoder's memory cross-attention input results in initial rewards in the middle of the scale (about ~4.5-5.5/10
in **RxT-Alpha-Micro**), but with added memory attention it dropped almost to zero, so we have to connect vector spaces before.

> In **RxNN framework** this stage is implemented in:
> - `SupervisedMemoryAttentionTrainer` from `rxnn.training.smst`
> - `MrlCurriculumDataset` from `rxnn.training.dataset`
> - `MemoryAttentionTrainingModel` from `rxnn.training.models`

#### MRL Curriculum Datasets
MRL Datasets, originally designed for Memory Reinforcement Learning are based on series of `N` interconnected interactions,
where each interaction includes some questions about previous interaction. `N` depending on the curriculum stage - starting
from small number of steps, it's increased in each stage. It will be described with details in next article.

### Supervised Memory-Aware Training
In first stages, decoder and encoder are using the same input (masked for encoder), so initially memory cross-attention is not
working correctly, using current interaction data instead of previous ones. In fourth stage, when we have pre-trained memory
attention, we have to refine memory cross-attention layers to use accumulated memory states instead. So now, components will
have different inputs - processing previous interaction with encoder, then combining it with previous state by memory attention
and finally decoder is processing next interaction - connected to previous ones.

Again, it's made for cold start problem in MRL and generate better initial answers. After this training, memory system will
be still "weak", but all the vector spaces are connected and model could be finally refined in reinforcement learning stages
to provide full functionality.

> Without 3rd and 4th stages, with new initialized memory attention, initial reward (65% BLEU, 15% cosine, 10% length) was
> on ~0.6/10.0 level and improved only to ~0.9/10.0 - the generated answers were completely unreadable. After new supervised
> stages, it's starting from ~2.0/10.0 and generated answers are mostly correct, including some information from previous steps.

Algorithm steps:
1. Starting from random (normal) STM state, save first interaction from batch as previous interaction
2. For `N` steps:
   - get next interaction from batch
   - process previous interaction with encoder and save results in STM with memory attention
   - process next interaction with decoder to get logits
   - calculate cross entropy loss, run backpropagation and optimization step
   - save next interaction as previous one

> In our **RxNN framework** this stage is implemented in:
> - `SupervisedMemoryAwareTrainer` from `rxnn.training.smst`
> - `MrlCurriculumDataset` from `rxnn.training.dataset`
> - `SupervisedMemoryAwareModel` from `rxnn.training.models`

## Summary
**Reactive Transformer** supervised training is a more complex process than training decoder-only LLM. The main challenge
is to correctly connect vector spaces between components and work with not interpretable memory. Described training stages
should result in partially working memory system, that is ready for the final reinforcement learning stages.