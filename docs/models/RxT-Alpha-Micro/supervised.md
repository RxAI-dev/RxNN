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
- ReactiveAI/TinyStories-Plus-Interaction-SFT
- ReactiveAI/TinyStories-MRL
library_name: RxNN
base_model:
- ReactiveAI/RxT-Alpha-Micro-Decoder
- ReactiveAI/RxT-Alpha-Micro-Encoder
- ReactiveAI/RxT-Alpha-Micro-Decoder-SFT
- ReactiveAI/RxT-Alpha-Micro-Encoder-SFT
- ReactiveAI/RxT-Alpha-Micro-MemAttn-Interlayer
- ReactiveAI/RxT-Alpha-Micro-Encoder-SMAT
- ReactiveAI/RxT-Alpha-Micro-Decoder-SMAT
- ReactiveAI/RxT-Alpha-Micro-MemAttn-SMAT
---

# RxT-Alpha Micro (Supervised)
World's first experimental real-time **Reactive Language Model (RxLM)** based on revolutionary **Reactive Transformer**
architecture - processing only single interactions/messages, with all the context moved to **Short-Term Memory**,
managed by **Attention-Based Memory System**.

**RxLMs** have linear computational/inference cost scaling (`O(NT)`) compared to **LLMs** quadratic growth (`O(N²T)`),
where `N` is the number of messages in conversation and `T` is the number of tokens in single interaction. Thanks to that
scaling, they are just `N` times faster and cheaper than **LLMs**.

That's not all from the advantages - event-driven real-time processing with memory is a lot more natural and human-like,
than LLMs data-driven approach (processing full conversation history everytime). It's a crucial milestone in development
of AGI and awareness models.

> This is _Supervised_ version of the model with "weak" memory system - result of Supervised Memory System Training. It's
> able to remember some information between interactions (without passing it explicitly in prompt/chat template), but it
> has to be refined in next Memory Reinforcement Learning stage for full functionality.

## Reactive Transformer Architecture
Experimental research model made to test our Reactive Transformer architecture and Attention-based Memory System.

Reactive Transformer has additional Short-Term Memory layers, connected to model with Memory Cross-Attention, and updated by Memory Encoder and Memory Attention.
Short-Term Memory state is kept between interactions/event (single message), not between tokens in sequence - that's key difference between RxNNs and RNNs.

The goal of the architecture is to process only single messages and keep conversation history in Short-Term Memory - we believe, that this is the key requirement
for awareness and AGI. Processing all the chat history on every interaction is not natural and that's not how human awareness is working. Then, Reactive Transformer
architecture is a first step in transition from language models to awareness models.

To balance number of the parameters, decoder is based on Mixture-of-Experts architecture, while the encoder is using regular
dense feed forward layers. This model is using interlayer version of memory attention networks with sigmoid residual gates.

### Architecture details:
- dim: 128
- layers: 6
- heads (for split): 8
- **Decoder:**
  - self-attention: symmetric Sparse Query Attention
    - query/key/value heads: 4/8
  - memory cross-attention: symmetric Sparse Query Attention
    - query/key/value heads: 4/8
  - Mixture-of-Experts Feed Forward
    - experts: 12
    - active experts: 2
    - SwiGLU feed forward with 256 dim
  - size: ~9M (~3.5M Activated)
- **Encoder:**
  - self-attention: symmetric Sparse Query Attention
    - query/key/value heads: 4/8
  - SwiGLU feed forward with 384 dim
  - size: ~1.9M
- **Memory Attention:**
  - variant: **Interlayer Memory Attention**
  - attention layers: symmetric Sparse Query Attention
    - query/key/value heads: 4/8
  - residual gate: linear with sigmoid activation (per STM slot)
  - size: ~1.2M
- RoPE for self-attention, memory cross-attention (query only) and memory attention (key only)
- RMS Norm for all normalization layers
- vocab: 7.5k (english only)
- interaction (query + answer) length: 256 tokens
- STM size: 6 layers * 256 slots (* 128 dim) 
- size: ~12M
- Library: RxNN

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" width="800" />

> RxT-Alpha models intentionally use very short sequence length and STM size (256 tokens for Micro), but that isn't their "full" context size - it's only for single
> message. "Full" context is theoretically infinite, restricted by STM size and memory abilities. For PoC models we want to
> reach 16 steps in Memory Reinforcement Learning curriculum, which should enable fluent conversations for 4k tokens context
> for this model. That sizes are good for research, final models will handle SOTA contexts.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png" width="800">


## RxT-Alpha Micro Training
Micro models from RxT-Alpha series are first PoC for Reactive Transformer, Attention-Based Memory System and Memory Reinforcement Learning,
used mainly to test library and architecture basics, before training bigger models (that are still relatively small, as it's PoC).

They are trained to generate simple stories based on [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories),
and follow-up answers to question about those stories.

Supervised Memory System Training includes 4 steps, before proceeding to Reinforcement Learning stages.

### Base Models Pre-Training
[**RxT-Alpha-Micro-Decoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder) was trained with [**RxT-Alpha-Micro-Encoder**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-Encoder)
and additional MLM head model [**RxT-Alpha-Micro-MLM**](https://huggingface.co/ReactiveAI/RxT-Alpha-Micro-MLM), using
Joint LM Training (with MLM and Autoregressive loss) and [**roneneldan/TinyStories**](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
Both encoder and decoder are using shared embedding layer

### Supervised Fine-Tuning
**RxT-Alpha-Micro** models were fine-tuned to generate real-time interactions (sequences) on our synthetic dataset (improved in v3),
inspired by TinyStories - [**ReactiveAI/TinyStories-Plus-Interaction-SFT**](https://huggingface.co/datasets/ReactiveAI/TinyStories-Plus-Interaction-SFT).

Models were fine-tuned using Joint LM Training mode (for memory cross-attention pre-training):
- encode data with encoder and calculate MLM loss for it
- save encoder layer's results as Short-Term Memory (available for decoder by memory cross-attention)
- process data with decoder and calculate autoregressive loss

That training results in decoder with ~95% accuracy, because it has access to all next tokens information with memory cross-attention. In next training stages it
will access previous interactions data with those layers.

### Self-Supervised Memory Attention Pre-Training
Memory Attention was pre-trained to combine accumulated Short-Term Memory states with next interaction data processed by the
encoder, using weighted mean (with randomized arbitrary weights) as labels and negative cosine similarity as loss. Label weights
depending on inner step:
- first step, when STM is in initial random normal state, using 90% of new encoded data
- follow-up steps are using `50% - step * 5%` of new encoded data
- each step could have 0-15% random differences in weights

Additionally, random noise is added to both inputs and labels.

This model was trained on six arbitrary selected steps using [**ReactiveAI/TinyStories-MRL**](https://huggingface.co/datasets/ReactiveAI/TinyStories-MRL)
dataset - `steps-6` subset and `supervised` split.

> This stage is fast and could reach convergence after even single epoch

### Supervised Memory-Aware Training
Finally, with pre-trained/fine-tuned components, in last supervised stage, model is trained to use previous/accumulated STM
states as memory cross-attention input, instead of the same sequences as decoder's input:
- previous (or first) interaction is processed by encoder and used to update memory
- next interaction is processed by decoder, using related information from STM
- loss is calculated from decoder's logits and gradients propagate through memory attention to encoder

In this stage we are using gradual unfreeze strategy:
- start from training only decoder
- after N epochs unfreeze memory attention
- after another K epochs unfreeze encoder

## Next Stage: Memory Reinforcement Learning
The model is able to generate grammatically correct answers with basic retention between interaction, and is ready for the
**Memory Reinforcement Learning** in the next stage. More info soon.

## Research in progress
Research and papers are in progress, drafts could be checked in RxNN docs:
- [Architecture introduction](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
- [Supervised Training stages](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/supervised-training.md)
- [Reinforcement Learning stages](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/mrl.md)

## Usage
_Supervised_ model is made mainly to be a starting point for **Memory Reinforcement Learning** - basic memory retention
after supervised stages is too weak for real usage (this model at all is not made for real usage, but as Proof-of-Concept).

It's still could be loaded and used for interactions (could run on CPU):
```python
from rxnn.rxt.models import RxTAlpha

model = RxTAlpha.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Supervised')
model.share_components() # currently required to connect embeddings/STM

seq_len = 256

# Memory init - could be used as "system prompt" in LLMs
stm_init_state = model.tokenize_full_interaction('System prompt like', 'Initial memory for the model', max_seq_len=seq_len)
model.init_stm_state(**stm_init_state)

# Helper function
def interaction(query: str):
  tokenized_query = model.tokenize_query(query, max_seq_len=seq_len)
  for token_id in model.interact(**tokenized_query, max_seq_len=seq_len, temperature=1.0):
    if token_id == -1: print('\n', '[Start memory update...]')
    elif token_id == -2: print('[Memory updated]')
    else:
      txt_token = model.stringify_token(token_id)
      print(txt_token, end='')

# Process first interaction
interaction('Tell me a story about...')    
# Process follow-up interaction      
interaction('What was that story about?')

```
