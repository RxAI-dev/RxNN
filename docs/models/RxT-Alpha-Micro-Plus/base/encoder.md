---
license: apache-2.0
pipeline_tag: fill-mask
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- RxNN
- ReactiveTransformer
language:
- en
datasets:
- roneneldan/TinyStories
library_name: RxNN
---

# RxT-Alpha Micro Plus Encoder (Base)
## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

> ### RxT-Alpha Micro Plus
> Extended architecture to improve results in Memory Reinforcement Learning. Changes vs. base RxT-Alpha Micro:
> - 10 layers instead of 6
> - symmetric SQA for memory cross-attention (4 key/value heads instead of 2)
> - 7.5k vocab instead of 5k
> - better tokenizer training
> 
> Model has ~3.1M vs ~1.8M in base version.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (encoder) is a memory encoder for Reactive Transformer system and is made for first stage of training - base model pre-training.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

During first stage, Memory Cross-Attention layers are frozen and STM is in default initial random state (normal distribution with 0 mean and almost 0 variance),
to not disturb basic language modelling training. We are training decoder and encoder separately with shared embeddings. Then, in second stage - Memory Reinforcement
Learning, they will be connected into bigger ensemble with additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">

Compared to decoder, encoder is using dense model, while decoder is Mixture-of-Experts (~4.5x bigger)

## RxT-Alpha Micro Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

Encoder was trained on Masked Language Modelling task with additional MLM head model [**RxT-Alpha-Micro-Plus-MLM**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Plus-MLM),
with [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, using **4B total tokens** and reached **~81% accuracy** on
validation dataset.

Pre-trained embeddings were then used for [**RxT-Alpha-Micro-Plus-Decoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Plus-Decoder) training.

### Encoder architecture details:
- dim: 128
- layers: 10
- heads: 8
- self-attention: symmetric Sparse Query Attention
  - query/key/value groups: 4
- memory cross-attention: symmetric Sparse Query Attention
  - query/key/value groups: 4
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 7.5k (english only)
- message length: 256
- STM size: 256 * 10 layers
- size: ~3.1M
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
