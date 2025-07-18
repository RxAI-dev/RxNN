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
- ReactiveAI/TinyStories-Plus-Interaction-SFT
library_name: RxNN
base_model:
- ReactiveAI/RxT-Alpha-Micro-Encoder
---

# RxT-Alpha Micro Encoder (SFT)
World's first experimental **Reactive/Real-Time Language Model** based on revolutional **Reactive Transformer** architecture - processing only single interactions/messages,
with all the context moved to **Short-Term Memory**, managed by **Attention-Based Memory System**.

> This is _SFT_ version of the model, still not able to update memory - it will be available from _MRL_ version (in training)

## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (encoder) is the fine-tuned memory encoder for Reactive Transformer system, trained to process single interactions (sequences) in real-time.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

In first two stages - pre-training and supervised fine-tuning, decoder and encoder are trained together - encoder layer's results are used as decoder's memory
cross-attention key/value inputs to align vector spaces between components. Then, in third stage - Memory Reinforcement Learning, they are connected with Memory Attention
layers, and full model is trained update and use memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">

Compared to decoder, encoder is using dense model, while decoder is Mixture-of-Experts (~4.5x bigger)

## RxT-Alpha Micro Training

### Pre-Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

Encoder was trained with additional MLM head model [**RxT-Alpha-Micro-MLM**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-MLM) and [**RxT-Alpha-Micro-Decoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Decoder),
using Joint LM Training (with MLM and Autoregressive loss) and [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
Both encoder and decoder are using shared embedding layer

### Supervised Fine-Tuning
**RxT-Alpha-Micro** models were fine-tuned to generate real-time interactions (sequences) on our synthetic dataset (improved in v3), inspired by TinyStories - [**ReactiveAI/TinyStories-Plus-Interaction-SFT**](https://huggingface.co/datasets/ReactiveAI/TinyStories-Plus-Interaction-SFT).

Models were fine-tuned using Joint LM Training mode (for memory cross-attention pre-training):
- encode data with encoder and calculate MLM loss for it
- save encoder layer's results as Short-Term Memory (available for decoder by memory cross-attention)
- process data with decoder and calculate autoregressive loss

That training results in decoder with ~95% accuracy, because it has access to all next tokens information with memory cross-attention. In next training stages it
will access previous interactions data with those layers. After this training model is partially able to read information from memory, before **Memory Reinforcement Learning**

#### Details
- GPU: 1x L40S
- epochs: full 8/8
- lr: 2e-4 peak, cosine annealing schedule
- batch size: 256
- processed tokens: ~200M
- encoder accuracy: ~77%

### Encoder architecture details:
- dim: 128
- layers: 6
- heads: 8
- self-attention: symmetric Sparse Query Attention
  - query/key/value heads: 4
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 7.5k (english only)
- message length: 256
- STM size: 256 * 6 layers
- size: ~2.1M
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
