<span>
  <img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/logo/logo_rxai_v2.png" width="400" />
  <img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/logo/logo_rxnn_v2.png" width="400" />
</span>

# Reactive AI - RxNN
## Reactive Neural Networks Platform

RxNN is AI/Deep Learning development platform made for Reactive Neural Networks and Event-driven AI, introduced by Reactive AI.

## Reactive Neural Networks and Event-driven AI
Reactive neural networks (RxNN) are a new family of memory-augmented neural networks that combine classical deep learning
algorithms with reactive communication patterns. In Event-driven AI, input data (sequence) is treated as event, and memory
state has to be kept between events/interactions. Technically, it's a specific kind of RNN that's storing data between
processed sequences, instead of between sequence elements like in regular RNN. Then, their recurrence is on a higher level.
In the case of reactive communication patterns, RxRNNs are stateful reactive data sources that you have to connect before
you can send and receive messages.
While RxNNs are using some RNN concepts, they are rather made to extend Transformer language/multi-modal models. In our
opinion, the biggest downside of current LLMs is their stateless nature - conversational models have to process full chat
history on every interaction! That's not real-time processing, and it's not how human's awareness is working. In RxNN based
transformers, model is processing single messages, while all the previous interactions history should be saved and read
from memory. That features are required for **Weak** Reactive Neural Networks specification, and it will be the first major
step in transition from language models to awareness models - in Reactive AI ecosystem, it will be introduced in Reactive
Transformer architecture.

Additionally, to achieve awareness, **Strong** Reactive Neural Networks are working in reactive infinite reasoning loop,
that's generating Infinite Chain-of-Thoughts and is communicating in push-based mode (model decides if and when return output).

Reactive communication patterns in RxNN models are adapted to handle asynchronous nature of model - after it finish generating
sequence, it has to process it and save it in memory, but it could be done in background.

## Release plan
We are working on three new reactive architectures, that progressively advance from language models to awareness models:
- Reactive Transformer: Reactive Language Model (RLM) with Short-Term Memory
- Preactor: extending Reactive Transformer with additional Long-Term Memory, providing theoretically infinite context (only
  single message length is limited) and the ability to learn from interactions (Live Learning)
- Reactor: AGI awareness model & Strong Reactive Neural Network, that's working in infinite reasoning loop and doesn't require explicit human commands

Each new architecture is based on the previous one and adding new features/abilities. They will be progressively
released with next versions of **RxNN** framework:
- 0.1.x (Released): Reactive Transformer base models, Base Model Learning (pre-training/fine-tuning) & Transformers extensions (MoE Attention, Short-Term Memory, etc.)
- 0.2.x (Released): Memory Reinforcement Learning (MRL) for Short-Term Memory & Reactive Transformer, Attention-based Memory System details
- 0.3.x: Reinforcement Learning from Human Feedback for Reactive models (RxRLHF), basic Tensor Reactive
  Extensions (TRX/Rust) for full Reactive Transformer, RxT-Alpha release (+following models - RxT-Beta, etc.)
- 0.4.x: Preactor base models, Tensor Database (TDB/Rust) for Long-Term Memory, mxRAG/revRAG subsystems
- 0.5.x: MRL for Long-Term Memory & Preactor, Live Learning for Preactor, PRx-Alpha release (+following models - PRx-Beta, etc.)
- 0.6.x: Reactor base models, TRX full implementation, Receptors & Effectors Reactive RNNs
- 0.7.x: Behavioral Reinforcement Learning (BRL) for Reactor's Infinite Chain-of-Thoughts, Continuous Live Learning for Reactor
- 0.8.x: Rx-Alpha release
- 0.9.x: Rx-Beta release
- 1.0.0: Reactor AGI official release (Expert, Assistant & Utility class models)
- 1.x.x: Multimodal reactive models (could be released earlier, depending on progress)
- 2.0.0: Real-Time Vision Reactor - Worker class models
- x.x.x: ...and more!

## Usage
**RxNN** is made to train models based on reactive architectures, as well as transformer language models. Current version
is based on PyTorch and HuggingFace libraries (Transformers/Datasets/Tokenizer/Hub), and is integrated with [HuggingFace Hub](https://hugginface.co)
and [TensorBoard](https://github.com/tensorflow/tensorboard).

> We are also planning a version for **TensorFlow**, more info soon

### Install library and dependencies
- RxNN and required deps: `pip install rxnn torch transformers tokenizers huggingface_hub`
- Datasets are required only for training: `pip install datasets`
- TensorBoard is optional: `pip install tensorboard`
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) is recommended for faster training/inference (required for models with explicit `use_flash_attention=True`) - check its separate [installation guide](#installing-flash-attention)
- **NumPy** should be installed too: `pip install numpy`

> ### Installing Flash Attention
> Installing `flash-attn` could be very frustrating and may take hours (with standard method), only to result in some incompatibility
> error. Fortunately, the prebuilt versions could be downloaded from GitHub and installed just in seconds. However, you should choose
> the compatible version based on:
> - Python version
> - CUDA version
> - PyTorch version (2.7 is currently not supported)
> - ABI
>
> #### Steps
> 1. Choose your version from [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)
> 2. Download prebuilt release, in example: `wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
> 3. Install it, in example: `pip install --no-dependencies --upgrade flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
> 4. Verify: `flash_attn.__version__` (an incorrect version will cause the error when importing)
> 
> #### Note on `use_flash_attention` option in models/layers
> Explicit `use_flash_attention` option is made to enable direct calls to `flash_attn_func` without using **PyTorch** `scaled_dot_product_attention`. Even
> if it's set to `False`, when `flash-attn` library is installed, **PyTorch** will try to use it implicitly through _SDPA backend_. It's better to set it
> to `False` and use automatically, because of better compatibility. Explicit options could be used for research

### Modules
**RxNN** framework has multiple modules with models, layers, training and inference tools, made for complete development
of _reactive models_, and could be also used for regular **Transformers**.

#### Transformers
Transformers module includes classes for models and layers. It includes **Reactive Transformers** as well as **Classic Transformers**

Submodules:
- `rxnn.transformers.attention` - basic, most common attention layers - `MultiHeadAttention`, `GroupedQueryAttention` and `MultiQueryAttention`
  - additional attention layers, especially `SparseQueryAttention` could be found in `rxnn.experimental.attention` module
  - `SparseQueryAttention` will be moved to `rxnn.transformers.attention` in 0.2.x version
- `rxnn.transformers.positional` - positional encoding layers - `RotaryPositionalEmbedding` and legacy ones - `AbsolutePositionalEmbedding`/`RelativePositionalEmbedding`
- `rxnn.transformers.ff` - dense feed forward layers, including gated layers (_SwiGLU_, etc.) - `FeedForward` & `GatedFeedForward` (recommended)
- `rxnn.transformers.moe` - Mixture-of-Experts feed forward layers - `MoeFeedForward` & `GatedMoeFeedForward` (recommended)
- `rxnn.transformer.layers` - complete reactive/classic transformer layers - `ReactiveTransformerLayer` & `ClassicTransformerLayer`
- `rxnn.transformer.models` - reactive/classic transformer models - `ReactiveTransformerEncoder`, `ReactiveTransformerDecoder` & `ClassicTransformerEncoder`, `ClassicTransformerDecoder`
- `rxnn.transformer.sampler` - samplers for reactive models (Sampler is the integral part of reactive architectures) - `Sampler`, `SampleDecoder`, `BatchSampler` & `BatchSampleDecoder`

In **RxNN** models are initialized in declarative style by class composition, but then they are wrapped in imperative classes,
to be compatible with HuggingFace **JSON** config. In example:

```python
from typing import TypedDict
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from rxnn.transformers.attention import GroupedQueryAttention
from rxnn.transformers.positional import RotaryPositionalEmbedding
from rxnn.transformers.layers import ReactiveTransformerLayer
from rxnn.transformers.models import ReactiveTransformerDecoder
from rxnn.memory.stm import ShortTermMemory

class YourReactiveTransformerConfig(TypedDict):
    num_layers: int
    vocab_size: int
    embed_dim: int
    ff_dim: int
    att_heads: int
    seq_len: int
    stm_size: int
    att_groups: int
    cross_att_groups: int


class YourReactiveTransformerDecoder(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            config: YourReactiveTransformerConfig,
            **kwargs
    ):
        super(YourReactiveTransformerDecoder, self).__init__(**kwargs)

        embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        rope = RotaryPositionalEmbedding(config['embed_dim'] // config['att_heads'], config['seq_len'])
        stm = ShortTermMemory(config['num_layers'], config['embed_dim'], config['stm_size'])

        self.model = ReactiveTransformerDecoder(
            stm=stm,
            embedding=embedding,
            own_layers=nn.ModuleList([
                ReactiveTransformerLayer(
                    config['embed_dim'],
                    config['ff_dim'],
                    use_gated=True,
                    use_moe=False,
                    ff_activation=nn.GELU(),
                    ff_dropout=0.1,
                    use_rms_norm=True,
                    self_attention=GroupedQueryAttention(
                        config['embed_dim'],
                        config['att_heads'],
                        config['att_groups'],
                        rope=rope,
                        dropout=0.1,
                        max_seq_len=config['seq_len'],
                        is_causal=True,
                    ),
                    memory_cross_attention=GroupedQueryAttention(
                        config['embed_dim'],
                        config['att_heads'],
                        config['att_groups'],
                        rope=rope,
                        dropout=0.1,
                        max_seq_len=config['seq_len'],
                        is_causal=True,
                        rope_only_for_query=True
                    ),
                ) for _ in range(config['num_layers'])
            ])
        )
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        return self.model(x, attention_mask=attention_mask)
```

#### Memory
The _memory_ module includes **Short-Term Memory** and layers responsible for its update. In future versions it will also
include **Long-Term Memory**.

The main `ShortTermMemory` class is located in `rxnn.memory.stm` module - the usage example is in Transformers module description.

> 0.2.x Memory modules docs in progress - will be released soon

#### Training
Training module includes **Trainers** for different training stages of reactive models and shared training utils.

Submodules:
- `rxnn.training.tokenizer` - custom Trainer for **HuggingFace** `tokenizers` and utils to load tokenizer from Hub
  - Tokenizer could be loaded from Hub with `load_tokenizer_from_hf_hub(repo_id)`
- `rxnn.training.dataset` - datasets for different training stages:
  - `MaskedLMDataset` & `AutoregressiveLMDataset` are made for base models pre-training
  - `EncoderSftDataset` & `DecoderSftDataset` are made for Interaction Supervised Fine-Tuning for reactive models
  - `MrlCurriculumDataset` is the dataset for single MRL Curriculum step
  - `MrlDatasets` is wrapping MRL datasets for all curriculum steps
  - each dataset has `from_hf_hub` class method to load dataset from Hub
  - they have also `concat_from_hf_hub` class method to load multiple Hub datasets into single training dataset
  - if dataset has no validation/test split, each dataset has `get_subset(subset_size, from_start=False)` method - it
    returns new subset and modifying existing one - i.e. `valid_dataset = train_dataset.get_subset(0.1)`
  - for concatenated datasets, validation/test split could be created with `concat_from_hf_hub_with_subset` - it cuts the
    same percentage of each loaded dataset
- `rxnn.training.callbacks` contain Trainer callbacks, for different kind of utils (more info below)
- `rxnn.training.scheduler` includes learning rate scheduler for training
- `rxnn.training.bml` - Base Model Learning module with Trainers for pre-training and fine-tuning
- `rxnn.training.mrl` - Memory Reinforcement Learning module with Trainers for MRL
- `rxnn.training.rxrlhf` - Reinforcement Learning from Human Feedback for Reactive Models module (from 0.3.x)
- `rxnn.training.brl` - Behavioral Reinforcement Learning module (Reactor / from 0.7.x)

##### Base Model Learning
Docs in progress

