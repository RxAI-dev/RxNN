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

# RxT-Alpha Micro Decoder (Base)
World's first experimental **Reactive/Real-Time Language Model** based on revolutional **Reactive Transformer** architecture - processing only single interactions/messages,
with all the context moved to **Short-Term Memory**, managed by **Attention-Based Memory System**.

> This is _Base_ version of the model, still not able to update memory - it will be available from _MRL_ version (in training)

## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (decoder) is a generator decoder for Reactive Transformer system and is made for first stage of training - base model pre-training.

Decoder is based on Mixture-of-Experts architecture with 12 experts and 2 active ones.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

In first two stages - pre-training and supervised fine-tuning, decoder and encoder are trained together - encoder layer's results are used as decoder's memory
cross-attention key/value inputs to align vector spaces between components. Then, in third stage - Memory Reinforcement Learning, they are connected with Memory Attention
layers, and full model is trained update and use memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">

## RxT-Alpha Micro Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

Decoder was trained with [**RxT-Alpha-Micro-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder) and additional MLM head model [**RxT-Alpha-Micro-MLM**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-MLM),
using Joint LM Training (with MLM and Autoregressive loss) and [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
Both encoder and decoder are using shared embedding layer

## Next Stage: Interaction Supervised Fine-Tuning
The model is able to generate meaningful short stories, using grammatically correct sentences, and is ready for the fine-tuning to interaction (single query + answer)
fine-tuning in next stage.

### Decoder architecture details:
- dim: 128
- layers: 6
- heads: 8
- self-attention: symmetric Sparse Query Attention
  - query/key/value heads: 4
- memory cross-attention: symmetric Sparse Query Attention
  - query/key/value heads: 4
- Mixture-of-Experts Feed Forward
  - experts: 12
  - active experts: 2
  - SwiGLU feed forward with 256 dim
- RoPE
- RMS Norm
- vocab: 7.5k (english only)
- message length: 256
- STM size: 256 * 6 layers
- size: ~9M (~3.5M Activated)
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
