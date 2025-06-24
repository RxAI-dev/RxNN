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
- ReactiveAI/TinyStories-Interaction-SFT
library_name: RxNN
---

# RxT-Alpha Micro Decoder (SFT)
## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

This model (decoder) is the fine-tuned generator decoder for Reactive Transformer system, trained to process/generate single interactions (sequences) in real-time.

Decoder is based on Mixture-of-Experts architecture with 12 experts and 2 active ones.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-moe.png" width="800" />

Same as in the first stage, in the second stage (Supervised Fine-Tuning) Memory Cross-Attention layers are frozen and STM is in default initial random
state (normal distribution with 0 mean and almost 0 variance), to not disturb interaction query-answer modeling. We are training decoder and encoder
separately, using shared embeddings from encoder training. Then, in third stage - Memory Reinforcement Learning, they will be connected into bigger
ensemble with additional Memory Norm and Memory Attention layers, and will learn how to keep and update memory.

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilites. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">



## RxT-Alpha Micro Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

Decoder was trained on Autoregressive Language Modelling task with embedding from [encoder pre-training](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder),
with [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, using **2.5B total tokens** and reached **~70.7% accuracy**.

### Supervised Fine-Tuning
**RxT-Alpha-Micro** models were fine-tuned to generate real-time interactions (sequences) on our synthetic dataset,
inspired by TinyStories - [**ReactiveAI/TinyStories-Interaction-SFT**](https://huggingface.co/datasets/ReactiveAI/TinyStories-Interaction-SFT).

Decoder reached the best validation loss after full 30 epochs (~433M processed tokens)

#### Details
- GPU: 1x L4
- epochs: full 30/30
- lr: 3e-4 peak, cosine annealing schedule
- batch size: 256
- processed tokens: ~433M
- loss: 0.5985 (validation) / 0.5865 (train)
- accuracy: **85.84%**

## Next Stage: Memory Reinforcement Learning
The model is able to generate meaningful interactions, using grammatically correct sentences, and is ready for the memory training in the next stage. More info soon.

### Decoder architecture details:
- dim: 128
- layers: 6
- heads: 8
- self-attention: symmetric Sparse Query Attention
  - query/key/value groups: 4
- memory cross-attention: Sparse Query Attention
  - query groups: 4
  - key/value groups: 2
- Mixture-of-Experts Feed Forward
  - experts: 12
  - active experts: 2
  - SwiGLU feed forward with 256 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- message length: 256
- STM size: 256 * 6 layers
- size: ~8.77M
- Library: RxNN
- Docs: [draft/in progress](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)

### Usage
Model requires [RxNN framework](https://github.com/RxAI-dev/RxNN) for training/inference. It's integrated with HuggingFace Hub and libraries.

#### Inference:
- Install RxNN, PyTorch and dependencies: `pip install rxnn torch transformers tokenizers`
- Install Flash Attention (optional, but recommended) - details in [RxNN framework docs](https://github.com/RxAI-dev/RxNN)
```python
import torch
from rxnn.rxt.models import RxTAlphaDecoder
from rxnn.transformers.sampler import Sampler, SampleDecoder
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub

model = RxTAlphaDecoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Decoder-SFT')
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/RxT-Alpha-Micro-Decoder-SFT')
sampler = Sampler(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), end_token_id=3)
sample = SampleDecoder(sampler, tokenizer)

# 0.1 and 0.9 are default values for temperature and top_p
generated = sample('[Q] Tell me a story about a little black dog [A]', temperature=0.1, top_p=0.9, max_seq_len=256)
sample('[Q] Tell me a story about a little black dog [A]', temperature=0.1, top_p=0.9, max_seq_len=256, print_stream=True)
```