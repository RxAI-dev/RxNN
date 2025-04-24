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
- roneneldan/TinyStories
library_name: RxNN
---

# RxT-Alpha Micro MoE Decoder
## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (decoder) is a generator decoder for Reactive Transformer system and is made for first stage of training - base model pre-training.

During first stage, Memory Cross-Attention layers are frozen and STM is in default initial random state (normal distribution with 0 mean and almost 0 variance),
to not disturb basic language modelling training. We are training decoder and encoder separately with shared embeddings. Then, in second stage - Memory Reinforcement
Learning, they will be connected into bigger ensemble with additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

## RxT-Alpha Micro Training
Micro models from RxT-Alpha series are the first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

This model is made to test Mixture-of-Experts models in the RxNN library and to improve regular [RxT-Alpha Micro Decoder](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Decoder).

Decoder was trained on Autoregressive Language Modelling task with embedding from [encoder pre-training](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder),
with [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, using **3B total tokens** and reached **~69.7% accuracy**.

## Next Stage: Memory Reinforcement Learning
Model is able to generate meaningful short stories and should be ready for the memory training in the next stage. More info soon.

### Decoder Mixture-of-Experts architecture details:
- dim: 128
- layers: 6
- heads: 8
- self-attention: GQA with 2 groups
- memory cross-attention: MQA
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- message length: 256
- STM size: 256 * 6 layers
- size: ~5.3M
- Library: RxNN
- Docs: More info soon