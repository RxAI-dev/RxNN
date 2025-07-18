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
- HuggingFaceFW/fineweb
- HuggingFaceFW/fineweb-edu
- wikimedia/wikipedia
library_name: RxNN
---

# RxT-Alpha Mini MLM
Masked Language Modelling head for [**RxT-Alpha-Mini-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Mini-Encoder) pre-training. In
**Reactive Transformer** architecture, final **Memory Encoder** is using only **Transformer** layers, but it has to be pre-trained using standard
MLM training, so we have to include additional head model.

### MLM Head Details:
- one linear layer (dim -> dim / 256 -> 256)
- GELU activation layer
- output linear layer (dim -> vocab / 256 -> 10 000)
- size: ~2.64M Params


## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

During first stage, Memory Cross-Attention layers are frozen and STM is in default initial random state (normal distribution with 0 mean and almost 0 variance),
to not disturb basic language modelling training. We are training decoder and encoder separately with shared embeddings. Then, in second stage, we are fine-tuning models
to interaction format (processing single messages), before the third stage - Memory Reinforcement Learning, they will be connected into bigger ensemble with
additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (1024 tokens for Mini), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

This model (MLM Head) is not used in final Reactive Transformer system. It's made only for first stage of training - base encoder model pre-training.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

## RxT-Alpha Mini Encoder + MLM Head Training
Mini models from RxT-Alpha series are the second PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning.

[**RxT-Alpha-Mini-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Mini-Encoder) was trained on Masked Language Modelling task with MLM head,
on Fineweb/Fineweb-edu/Wikipedia datasets, using **20B total tokens** and reached **~62% accuracy**.

Pre-trained embeddings were then used for [**RxT-Alpha-Mini-Decoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Mini-Decoder) training.

### Encoder architecture details:
- dim: 256
- layers: 8
- heads: 16
- self-attention: symmetric Sparse Query Attention
  - query/key/value groups: 8
- memory cross-attention: Sparse Query Attention
  - query groups: 8
  - key/value groups: 4
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- message length: 1024
- STM size: 1024 * 8 layers
- size: ~11M
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
