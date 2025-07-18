---
license: apache-2.0
pipeline_tag: fill-mask
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- RxNN
- ReactiveTransformer
- MLM
- lm_head
language:
- en
datasets:
- roneneldan/TinyStories
library_name: RxNN
---

# RxT-Alpha Micro MLM
Masked Language Modelling head for [**RxT-Alpha-Micro-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder) pre-training. In
**Reactive Transformer** architecture, final **Memory Encoder** is using only **Transformer** layers, but it has to be pre-trained using standard
MLM training, so we have to include additional head model.

### MLM Head Details:
- one linear layer (dim -> dim / 128 -> 128)
- GELU activation layer
- output linear layer (dim -> vocab / 128 -> 5000)
- size: ~660k Params


## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

During first stage, Memory Cross-Attention layers are frozen and STM is in default initial random state (normal distribution with 0 mean and almost 0 variance),
to not disturb basic language modelling training. We are training decoder and encoder separately with shared embeddings. Then, in second stage - Memory Reinforcement
Learning, they will be connected into bigger ensemble with additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

This model (MLM Head) is not used in final Reactive Transformer system. It's made only for first stage of training - base encoder model pre-training.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

## RxT-Alpha Micro Encoder + MLM Head Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

[**RxT-Alpha-Micro-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder) was trained on Masked Language Modelling task with MLM head,
on [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, using **2.5B total tokens** and reached **~81.7% accuracy**.

Pre-trained embeddings were then used for [**RxT-Alpha-Micro-Decoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Decoder) training.

### Supervised Fine-Tuning
**RxT-Alpha-Micro** models were fine-tuned to generate real-time interactions (sequences) on our synthetic dataset, inspired by TinyStories - [**ReactiveAI/TinyStories-Interaction-SFT**](https://huggingface.co/datasets/ReactiveAI/TinyStories-Interaction-SFT).

Encoder reached the best validation loss after full 30 epochs (~433M processed tokens)

#### Details
- GPU: 1x L4
- epochs: full 30/30
- lr: 3e-4 peak, cosine annealing schedule
- batch size: 256
- processed tokens: ~433M
- loss: 0.6366 (validation) / 0.7131 (train)
- accuracy: **85.69%**

### Encoder architecture details:
- dim: 128
- layers: 6
- heads: 8
- self-attention: symmetric Sparse Query Attention
  - query/key/value groups: 4
- memory cross-attention: Sparse Query Attention
  - query groups: 4
  - key/value groups: 2
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- message length: 256
- STM size: 256 * 6 layers
- size: ~1.88M (+ ~660k MLM Head = ~2.5M for pre-training)
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
