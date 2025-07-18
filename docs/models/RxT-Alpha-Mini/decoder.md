---
license: apache-2.0
pipeline_tag: text-generation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- RxNN
- ReactiveTransformer
language:
- en
datasets:
- HuggingFaceFW/fineweb
- HuggingFaceFW/fineweb-edu
- wikimedia/wikipedia
library_name: RxNN
---

# RxT-Alpha Mini Decoder (Base)
## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (decoder) is a generator decoder for Reactive Transformer system and is made for first stage of training - base model pre-training.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

During first stage, Memory Cross-Attention layers are frozen and STM is in default initial random state (normal distribution with 0 mean and almost 0 variance),
to not disturb basic language modelling training. We are training decoder and encoder separately with shared embeddings. Then, in second stage, we are fine-tuning models
to interaction format (processing single messages), before the third stage - Memory Reinforcement Learning, they will be connected into bigger ensemble with
additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (1024 tokens for Mini), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">

Decoder is based on Mixture-of-Experts architecture with 16 experts and 2 active ones.

## RxT-Alpha Mini Training
Mini models from RxT-Alpha series are second PoC (after micro-scale models) for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning.

Decoder was trained on Autoregressive Language Modelling task with embedding from [encoder pre-training](https://huggingface.co/ReactiveAI/RxT-Alpha-Mini-Encoder),
with Fineweb/Fineweb-edu/Wikipedia datasets, using **20B total tokens** and reached **~67% accuracy**.


### Decoder architecture details:
- dim: 256
- layers: 8
- heads: 16
- self-attention: symmetric Sparse Query Attention
  - query/key/value groups: 8
- memory cross-attention: symmetric Sparse Query Attention
  - query/key/value groups: 8
- Mixture-of-Experts Feed Forward
  - experts: 16
  - active experts: 2
  - SwiGLU feed forward with 512 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- message length: 1024
- STM size: 1024 * 8 layers
- size: ~60M
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
