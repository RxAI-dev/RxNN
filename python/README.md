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
- **Reactive Transformer**: Reactive Language Model (RLM) with Short-Term Memory. [Research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md)
- **Preactor**: extending Reactive Transformer with additional Long-Term Memory, providing theoretically infinite context (only
  single message length is limited) and the ability to learn from interactions (Live Learning)
- **Reactor**: AGI awareness model & Strong Reactive Neural Network, that's working in infinite reasoning loop and doesn't require explicit human commands

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

## Modules
**RxNN** framework has multiple modules with models, layers, training and inference tools, made for complete development
of _reactive models_, and could be also used for regular **Transformers**.

### Transformers
Transformers module includes classes for models and layers. It includes **Reactive Transformers** as well as **Classic Transformers**

Submodules:
- `rxnn.transformers.attention` - basic, most common attention layers - `MultiHeadAttention`, `GroupedQueryAttention` and `MultiQueryAttention`
  - additional attention layers, especially `SparseQueryAttention` could be found in `rxnn.experimental.attention` module
  - `SparseQueryAttention` will be moved to `rxnn.transformers.attention` in 0.3.x version
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
                        config['cross_att_groups'] if 'cross_att_groups' in config else config['att_groups'],
                        rope=rope,
                        dropout=0.1,
                        max_seq_len=config['seq_len'],
                        is_causal=False,
                        rope_only_for_query=True
                    ),
                ) for _ in range(config['num_layers'])
            ])
        )
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        return self.model(x, attention_mask=attention_mask)
```

#### RxT-Alpha
`RxTAlphaEncoder` and `RxTAlphaDecoder` are ready to use **Reactive Transformer** components, compatible with Hugging Face
Hub (the above example is based on their code), so it could be used instead of creating custom class. Example usage could
be found in [pre-training docs](#pre-training)

### Memory
The _memory_ module includes **Short-Term Memory (STM)** and layers responsible for its update. In future versions it will also
include **Long-Term Memory (LTM)**.

#### Short Term Memory
The main `ShortTermMemory` class is located in `rxnn.memory.stm` module. As described in [Reactive Transformer research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/reactive-transformer.md),
each transformer (encoder and decoder) layer has its own **STM** layer of shape `[batch_size, stm_size, embed_dim]`. Initially,
for the first training stages (pre-training and supervised fine-tuning), **STM** is in "single/no batch" mode (`batch_size = 1`),
because it's not used. For reinforcement learning stages (**MRL/RxRLHF/BRL**), we have to switch short-term memory to batch
mode, because items in batches are independent. After training, we could switch back to "single/no batch" mode. Example:
```python
from rxnn.memory.stm import ShortTermMemory

num_layers = 10
stm_size = 256
embed_dim = 128
batch_size = 32

# 1. Init STM
stm = ShortTermMemory(
  num_layers, embed_dim, stm_size,
  init_type='normal' # memory init type, 'normal' is default and means normal distribution with 0.0 mean and 0.02 std
)

# 2. Set "batch" mode for MRL
stm.batched_memory(
  batch_size,
  init_type='standard' # init type could be changed for batch mode, 'standard' is normal distribution with 0.0 mean and 1.0 std
)

# 3. Reset STM with optional init type change
stm.reset(init_type='uniform') # init type could be also 'ones' or 'zeros', but it's not recommended

# 4. Back to "single" mode for inference (optionally using mean value from batch)
stm.single_memory(
  init_type='standard', # we could change init type again
  use_mean_from_batch=True # use mean values from batch as new memory
)
```

> ##### Other utils
> `ShortTermMemory` could be also resized with `stm.resize(new_stm_size, init_type)` method, detached and cloned
> with `stm.clone_detach_reset()` (used in MRL), or could be made trainable (experimental option):
> - could be initialized as trainable - `stm = ShortTermMemory(num_layers, embed_dim, stm_size, is_trainable=True)`
> - could be switched to trainable - `stm.make_trainable()`
> - and switched back to buffer - `stm.freeze()`

#### Memory Attention Network
**Memory Attention Network** is responsible for memory layers update. It includes memory attention layers, with normalization
and residual connection (with optional gated residual). **Memory Attention Network** should have the same number of layers
as other components (encoder & decoder). It takes the results from each encoder layer (hidden states), and combine them
with actual memory state.

You can create your own **Memory Attention Network**, integrated with **HuggingFace Hub**, same way as reactive transformers:
```python
from typing import TypedDict
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from rxnn.transformers.attention import GroupedQueryAttention
from rxnn.transformers.positional import RotaryPositionalEmbedding
from rxnn.memory.stm import ShortTermMemory
from rxnn.memory.attention import StmMemoryAttention

class YourMemoryAttentionConfig(TypedDict):
    num_layers: int
    vocab_size: int
    embed_dim: int
    ff_dim: int
    att_heads: int
    seq_len: int
    stm_size: int
    att_groups: int

class YourMemoryAttention(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    """RxT-Alpha (Reactive Transformer) memory attention model"""

    def __init__(
            self,
            config: YourMemoryAttentionConfig,
            **kwargs,
    ):
        super(YourMemoryAttention, self).__init__(**kwargs)

        rope = RotaryPositionalEmbedding(config['embed_dim'] // config['att_heads'], config['seq_len'])
        # This separately initialized STM will be replaced by shared instance with `load_shared_memory` call
        stm = ShortTermMemory(config['num_layers'], config['embed_dim'], config['stm_size'])

        self.model = StmMemoryAttention(
            stm,
            attention_layers=nn.ModuleList([
                GroupedQueryAttention(
                    config['embed_dim'],
                    config['att_heads'],
                    config['att_groups'],
                    rope=rope,
                    dropout=0.1,
                    is_causal=False,
                    rope_only_for_keys=True
                ) for _ in range(config['num_layers'])
            ]),
            memory_norm_layers=nn.ModuleList([
              nn.RMSNorm(config['embed_dim']) for _ in range(config['num_layers'])
            ]),
            use_gated_residual=False, # memory attention residual gate
            per_slot_gate=False, # gate per memory slot, otherwise it's per layer
            init_gate=None, # initial value for gate weights
            use_dynamic_gate=False, # dynamic gate calculated from weights and memory state, otherwise it's calculated only from weights
            use_tanh_gate=False, # use tanh gate, otherwise it's sigmoid
        )

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)
```

> #### Gated residual
> Optional gated residual could be used to improve Memory Attention expressiveness. It's using gate (sigmoid or tanh)
> with trainable weights, to decide how much information from old and new updated memory state should be stored. Depending
> on params weights are declared per layer or per memory slot (more expressive). It could work in two modes, that could
> be switched, because they are using the same weights shape:
> - static - gate values calculated only from weights (`gate = torch.sigmoid(weights)`) - enable explicit control with `init_gate` param
> - dynamic - gate values calculated from weights and updated memory state (`gate = torch.sigmoid(weights * (new_layer_stm + layer_stm).mean(dim=-1, keepdim=True))`)
> 
> Depending on `use_tanh_gate` param, final gated residual connection is calculated with different formulas:
> - sigmoid gate - `stm[i] = layer_gate * new_layer_stm + (1 - layer_gate) * layer_stm`
> - tanh gate - `stm[i] = (1 + layer_gate) * new_layer_stm + (1 - layer_gate) * layer_stm`
> - tanh gate preserve residual connection scale, while sigmoid gate result is equivalent to `(new_layer_stm + layer_stm) / 2`
>
> **Gated residual** is currently in tests - we are not sure if it will provide better results, so **it's not recommended**

##### RxT-Alpha Memory Attention
`RxTAlphaMemoryAttention` is ready to use Memory Attention network for **Reactive Transformer** Proof-of-Concept, that
could be used instead of creating custom class. Example is in [Memory Reinforcement Learning docs](#memory-reinforcement-learning)

### Training
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
  - each dataset has `pre_tokenize` method, to tokenize all items before the training (otherwise they are tokenized on
    dynamically on item access). It's recommended for smaller datasets (fine-tuning, MRL, etc.) and not recommended for
    very big datasets (pre-training), because it's using a lot of RAM (CPU)
- `rxnn.training.callbacks` contain Trainer callbacks, for different kind of utils (more info below)
- `rxnn.training.scheduler` includes learning rate scheduler for training
- `rxnn.training.bml` - Base Model Learning module with Trainers for pre-training and fine-tuning
- `rxnn.training.mrl` - Memory Reinforcement Learning module with Trainers for MRL
- `rxnn.training.rxrlhf` - Reinforcement Learning from Human Feedback for Reactive Models module (from 0.3.x)
- `rxnn.training.brl` - Behavioral Reinforcement Learning module (Reactor / from 0.7.x)

#### Base Model Learning
**Base Model Learning (BML)** module is made for both pre-training and fine-tuning base models, that will be used as components
in reactive models. Generally the only two differences between pre-training and supervised fine-tuning are different dataset
classes and trainer/callbacks hyperparams config.

Reactive models are based on transformer decoder and encoder, with shared embeddings and memory layers - it require special
handling in first training stages:
- layers connected with memory - **Memory Cross-Attention** are frozen during pre-training and fine-tuning, and they are
  skipped by residual connections
- as encoder is able to learn little better embeddings, because of bidirectional modelling, it's pre-trained first, then
  decoder is trained with frozen embeddings from encoder
- in **Reactive Transformer** fine-tuning, both encoder and decoder are fit to interaction format (single query and answer), the
  training order is the same as for pre-training
- in **Preactor** architecture there are 2 encoders and single decoder. Encoders are fine-tuned from single pre-trained
  encoder - first one is processing only queries and second one only the answers. More info soon
- in **Reactor** architecture there are 2 encoders and 2 decoders. Both encoders and decoders are fine-tuned from single
  pre-trained encoder and decoder. Each component is fine-tuned to their specific task. More info soon

##### Pre-training
We have to start with importing required modules/libraries, initializing the models and loading the tokenized - I will
use _RxT-Alpha-Micro-Plus_ models as example:
```python
import torch
from rxnn.rxt.models import RxTAlphaDecoder, RxTAlphaEncoder
from rxnn.training.dataset import AutoregressiveLMDataset, MaskedLMDataset
from rxnn.training.bml import AutoregressiveTrainer, MLMTrainer
from rxnn.training.models import MLMHead, MLMTrainingModel
from rxnn.training.scheduler import get_transformer_lr_scheduler, calculate_steps
from rxnn.training.callbacks import PrintLossCallback, PrintAccuracyCallback, TokenCounterCallback, ModelSaveCallback, JointModelSaveCallback
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub
from rxnn.utils import set_random_seed, cache_clean

embed_dim = 128
vocab_size = 7_500
seq_len = 256

set_random_seed(42)

config = {
  'num_layers': 10,
  'vocab_size': vocab_size,
  'embed_dim': embed_dim,
  'att_heads': 16, # attention heads, in SQA it's used only for dimension split
  'att_groups': 8, # key/value groups for GQA/SQA
  'seq_len': seq_len,
  'stm_size': seq_len,
  'use_flash_attention': False, # explicitly use flash-attn function (otherwise it's used through PyTorch backend) - not recommended
  'use_gated': True, # use Gated Linear Units in feed forward, True by default
  'ff_activation': 'silu', # feed forward activation, 'silu' is default for SwiGLU layers
  'ff_dropout': 0.1,
  'self_att_type': 'sqa', # self attention could be 'sqa', 'gqa', 'mqa' or 'mha'
  'cross_att_type': 'sqa', # self attention could be 'sqa', 'gqa', 'mqa' or 'mha'
  'att_query_groups': 8, # query groups for SQA
}

encoder_config = {
  'ff_dim': 384,
  **config
}

decoder_config = {
  'ff_dim': 256,
  'use_moe': True, # use Mixture-of-Experts feed forward
  'num_experts': 20, # number of experts
  'moe_top_k': 4, # number of activated experts (per token)
  **config
}

encoder = RxTAlphaEncoder(**encoder_config)
decoder = RxTAlphaDecoder(**decoder_config)
head = MLMHead(embed_dim, vocab_size)

# Tokenizer is the same for encoder and decoder
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/RxT-Alpha-Micro-Plus-Encoder', token='HF_TOKEN')
```
Then, we have to load MLM datasets, set callbacks and run encoder training:
```python
# 1. Load datasets
load_kwargs = {
    'trust_remote_code': True
}

train_dataset = MaskedLMDataset.from_hf_hub('roneneldan/TinyStories', load_kwargs=load_kwargs, tokenizer=tokenizer, max_seq_len=seq_len)
valid_dataset = MaskedLMDataset.from_hf_hub('roneneldan/TinyStories', split="validation", load_kwargs=load_kwargs, tokenizer=tokenizer, max_seq_len=seq_len)

# 2. Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Clean GPU cache (optional)
cache_clean()

# 4. Set training config variables
batch_size = 256
epochs = 8
gradient_acc_steps = 1
peak_lr = 1e-3 * gradient_acc_steps

# 5. Get number of steps for scheduler
steps_config = calculate_steps(len(train_dataset), epochs, batch_size, warmup_ratio=0.05, verbose=True)
steps_per_epoch, total_steps, warmup_steps = steps_config['epoch'], steps_config['total'], steps_config['warmup']

# 6. Freeze memory cross-attention layers
encoder.freeze_memory()

# 7. Select directory for TensorBoard logs
logs_dir = './micro/tensorboard_logs/encoder-plus-sft'

# 8. Basic callbacks - print loss, accuracy and number of processed tokens
print_cb = PrintLossCallback(batches_per_epoch=steps_per_epoch)
count_cb = TokenCounterCallback(3_000_000_000)
acc_cb = PrintAccuracyCallback()

# 9. Joint model save callback - used to save encoder and MLM head, and push them to HuggingFace Hub 
save_cb = JointModelSaveCallback(
  './micro/encoder-plus-sft',
  push_to_hub=True,
  hub_model_decoder=None,
  hub_model_encoder='Your encoder model id',
  hub_model_head='Your mlm model id',
  push_checkpoint_weights=True, # push epoch checkpoints to hub
  final_commit_message='Final commit message',
  private_repo=False, # use HF private repository
  save_checkpoint_after_n_batches=1000, # save model after N batches in epoch (batch checkpoint)
  push_batch_checkpoint=True, # push batch checkpoints to HF Hub
  mlm_mode=True, # use MLM mode
  hf_token='HF_TOKEN',
  use_ddp=False, # use distributed training mode
)

# 10. Init training model - encoder + head
model = MLMTrainingModel(encoder, head)

# 11. Init MLM Trainer
trainer = MLMTrainer(
  model,
  device,
  dataset=train_dataset,
  validation_dataset=valid_dataset,
  vocab_size=vocab_size,
  callbacks=[print_cb, acc_cb, count_cb, save_cb],
  use_amp=True, # use autocast
  dtype=torch.bfloat16, # data type for training
  log_dir=logs_dir,
  use_ddp=False, # use distributed training mode
)

# 12. Init optimizer and cosine annealing scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.02)
scheduler = get_transformer_lr_scheduler(
    optimizer,
    warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 13. Run the training for the selected number of epochs
trainer(epochs=epochs, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler)
```
After the encoder's training, we have to train decoder:
```python
# 1. Load datasets
load_kwargs = {
    'trust_remote_code': True
}

train_dataset = AutoregressiveLMDataset.from_hf_hub('roneneldan/TinyStories', load_kwargs=load_kwargs, tokenizer=tokenizer, max_seq_len=seq_len)
valid_dataset = AutoregressiveLMDataset.from_hf_hub('roneneldan/TinyStories', split="validation", load_kwargs=load_kwargs, tokenizer=tokenizer, max_seq_len=seq_len)

# 2. Load shared embedding and memory, then freeze embedding and memory cross-attention
decoder.load_shared_embedding(encoder.model.embedding)
decoder.load_shared_memory(encoder.model.stm)

decoder.model.embedding.requires_grad_(False)
decoder.freeze_memory()

# 3. Clean GPU cache (optional)
cache_clean()

# 4. Set training config variables
batch_size = 256
epochs = 8
gradient_acc_steps = 1
peak_lr = 1e-3 * gradient_acc_steps

# 5. Get number of steps for scheduler
steps_config = calculate_steps(len(train_dataset), epochs, batch_size, warmup_ratio=0.05, verbose=True)
steps_per_epoch, total_steps, warmup_steps = steps_config['epoch'], steps_config['total'], steps_config['warmup']

# 6. Select directory for TensorBoard logs
logs_dir = './micro/tensorboard_logs/decoder-plus-sft'

# 7. Basic callbacks - print loss, accuracy and number of processed tokens
print_cb = PrintLossCallback(batches_per_epoch=steps_per_epoch)
count_cb = TokenCounterCallback(5_000_000_000)
acc_cb = PrintAccuracyCallback()

# 8. Model save callback - used to save decoder and push it to HuggingFace Hub 
save_cb = ModelSaveCallback(
  './micro/decoder-plus-sft',
  push_to_hub=True,
  hub_model_id='Your decoder model id',
  push_checkpoint_weights=True, # push epoch checkpoints to hub
  final_commit_message='Final commit message',
  private_repo=False, # use HF private repository
  save_checkpoint_after_n_batches=1000, # save model after N batches in epoch (batch checkpoint)
  push_batch_checkpoint=True, # push batch checkpoints to HF Hub
  hf_token='HF_TOKEN',
  use_ddp=False, # use distributed training mode
)

# 9. Init Autoregressive Trainer
trainer = AutoregressiveTrainer(
  decoder,
  device,
  dataset=train_dataset,
  validation_dataset=valid_dataset,
  vocab_size=vocab_size,
  callbacks=[print_cb, acc_cb, count_cb, save_cb],
  use_amp=True,
  dtype=torch.bfloat16,
  log_dir=logs_dir,
  use_moe_aux_loss=True, # Add MoE Router auxiliary loss to main loss
  moe_aux_loss_scale=0.02, # MoE Router aux loss scale
  use_ddp=False, # use distributed training mode
)

# 10. Init optimizer and cosine annealing scheduler
optimizer = torch.optim.AdamW(decoder.parameters(), lr=peak_lr, weight_decay=0.02)
scheduler = get_transformer_lr_scheduler(
    optimizer,
    warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 11. Run the training for the selected number of epochs
trainer(epochs=epochs, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler)
```

##### Fine-tuning
For _**Interaction Supervised Fine-Tuning**_, the code is almost the same as for pre-training, with some small changes.

First, we have to load pre-trained models, instead of initializing them with configs:
```python
encoder = RxTAlphaEncoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-Encoder', token='HF_TOKEN')
decoder = RxTAlphaDecoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-Decoder', token='HF_TOKEN')
head = MLMHead.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-MLM', token='HF_TOKEN')
```

Then, we have to change the datasets loading part. For encoder:
```python
# 1. Load datasets
train_dataset = EncoderSftDataset.from_hf_hub('ReactiveAI/TinyStories-Plus-Interaction-SFT', tokenizer=tokenizer, max_seq_len=seq_len)
valid_dataset = EncoderSftDataset.from_hf_hub('ReactiveAI/TinyStories-Plus-Interaction-SFT', split="validation", tokenizer=tokenizer, max_seq_len=seq_len)

# 2. Pre-tokenize dataset with verbose logging (optional)
train_dataset.pre_tokenize(verbose=True, log_interval=5000)
valid_dataset.pre_tokenize(verbose=True, log_interval=1000)
```
And the same for decoder:
```python
# 1. Load datasets
train_dataset = DecoderSftDataset.from_hf_hub('ReactiveAI/TinyStories-Plus-Interaction-SFT', tokenizer=tokenizer, max_seq_len=seq_len)
valid_dataset = DecoderSftDataset.from_hf_hub('ReactiveAI/TinyStories-Plus-Interaction-SFT', split="validation", tokenizer=tokenizer, max_seq_len=seq_len)

# 2. Pre-tokenize dataset with verbose logging (optional)
train_dataset.pre_tokenize(verbose=True, log_interval=5000)
valid_dataset.pre_tokenize(verbose=True, log_interval=1000)
```

We could also add early stoppage callback:
```python
from rxnn.training.callbacks import EarlyStoppageCallback

stop_cb = EarlyStoppageCallback(num_plateau_epochs=5)
```

Additionally, in fine-tuning we will rather use different config for number of epochs, steps, learning rate, etc.

> #### Classic Transformer Training
> The same code could be used also to train classic decoder-only or encoder-only transformers, the only difference is
> that they don't require memory cross-attention freezing.

##### Joint Training
There are also `JointLMDataset` and `JointLMTrainer` classes to train encoder and decoder at once. In that case, embeddings
are updated from both encoder and decoder optimization. However, I noticed some issues with balancing training in that mode,
so it's **not recommended** now, until it will be tested and fixed

#### Memory Reinforcement Learning
**Memory Reinforcement Learning (MRL)** is the most important training stage for reactive model's **Attention-Based Memory System**.
In this stage we are training model to remember information between multiple interactions, with different curriculum stage
configs. Theoretical foundations are described in [research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/mrl.md).

> **MRL** algorithm is currently in tests and still a lot of things could be changed!

In practice, algorithm has over 50 hyperparams, so it require careful handling. We start from importing modules, loading
pre-trained models from SFT stage, initializing new Memory Attention, and actor and critic models:
```python
import torch
from rxnn.rxt.models import RxTAlphaDecoder, RxTAlphaEncoder, RxTAlphaMemoryAttention
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub
from rxnn.training.dataset import MrlDatasets
from rxnn.training.models import MrlActorModel, MrlCriticModel
from rxnn.training.reward import MrlRewardModel
from rxnn.training.mrl import MRLTrainer, CurriculumConfig, MrlStrategy, MrlConfig
from rxnn.training.rl import PPOAlgorithm, PPOConfig
from rxnn.training.callbacks import MrlPrintCallback, MrlEarlyStoppageCallback, MrlModelSaveCallback, MrlGeneratedTokensCallback
from rxnn.utils import set_random_seed

# 1. Set random seed, batch size and embed dim
set_random_seed(42)
batch_size = 64
embed_dim = 128

# 2. Get pre-trained microscale PoC models
decoder = RxTAlphaDecoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-Decoder-SFT', token='HF_TOKEN')
encoder = RxTAlphaEncoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-Encoder-SFT', token='HF_TOKEN')
# 3. Init Memory Attention Network
mem_attn = RxTAlphaMemoryAttention(
    num_layers=10,
    embed_dim=embed_dim,
    att_heads=8,
    seq_len=256,
    stm_size=256,
    use_flash_attention=False, # explicitly use flash-attn function (otherwise it's used through PyTorch backend)
    norm_type='classic-rms', # memory norm type
    att_groups=4, # key/value groups for SQA/GQA
    att_type='sqa', # attention type, could be 'sqa', 'gqa', 'mqa' or 'mha'
    att_query_groups=4, # query groups for SQA
)

# 4. Load shared embedding and memory from encoder to other models
decoder.load_shared_embedding(encoder.model.embedding)
encoder.model.stm.batched_memory(batch_size=batch_size, init_type='standard')
decoder.load_shared_memory(encoder.model.stm)
mem_attn.load_shared_memory(encoder.model.stm)

# 5. Init Actor model
actor = MrlActorModel(encoder, decoder, mem_attn)

# 6. Get pre-trained encoder, extend its context size, freeze memory and use as a body for Critic model
critic_encoder = RxTAlphaEncoder.from_pretrained('ReactiveAI/RxT-Alpha-Micro-Plus-Encoder-SFT', token='HF_TOKEN')

critic_encoder.update_max_len(512)
critic_encoder.freeze_memory()
# 7. Init Critic model
critic = MrlCriticModel(critic_encoder, embed_dim)
```

Then, we have to load tokenizer and MRL Datasets, and create _curriculum config_:
```python
# 1. Load tokenizer
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/RxT-Alpha-Micro-Plus-Decoder', token='HF_TOKEN')

# 2. Load PoC TinyStories based MRL Dataset, starting from 4 steps to 16 in long range, and pre-tokenize it
mrl_datasets = MrlDatasets.from_hf_hub(
    'ReactiveAI/TinyStories-MRL',
    tokenizer,
    mrl_curriculum_steps=[
        { 'subset_name': 'steps-4', 'steps': 4, 'is_long_range': False },
        { 'subset_name': 'steps-6', 'steps': 6, 'is_long_range': False },
        { 'subset_name': 'steps-8', 'steps': 8, 'is_long_range': False },
        { 'subset_name': 'steps-8-lr', 'steps': 8, 'is_long_range': True },
        { 'subset_name': 'steps-12', 'steps': 12, 'is_long_range': True },
        { 'subset_name': 'steps-16', 'steps': 16, 'is_long_range': True },
    ],
    eval_split='validation',
    max_seq_len=256,
)

mrl_datasets.pre_tokenize(verbose=True, log_interval=100)

# 3. Create curriculum stages config
curriculum_stages = [CurriculumConfig(
    steps=item['steps'], # number of steps in curriculum stage
    epochs=10 if item['steps'] == 4 else 5, # number of epochs in curriculum stage 
    dataset=item['dataset'],
    eval_dataset=item['eval_dataset'],
    callbacks=[
        MrlPrintCallback(), # Print loss/reward callback
        MrlModelSaveCallback(
            './models',
            push_to_hub=True,
            hub_model_critic='Your critic model hub id',
            hub_model_decoder='Your MRL decoder model hub id',
            hub_model_encoder='Your MRL encoder model hub id',
            hub_model_memory_attention='Your memory-attention model hub id',
            private_repo=True,
            hf_token='HF_TOKEN',
            final_commit_message=f"MRL steps: {item['steps']} {'lr' if item['is_long_range'] else ''}",
            push_checkpoint_weights=True,
        ) # MRL Model save callback - save and push to hub critic model and actor components
    ],
    strategy=MrlStrategy.LONG_RANGE_STRATEGY if item['is_long_range'] else MrlStrategy.MULTI_STEP_STRATEGY, # strategy for curriculum stage
    unfreeze_epoch=((2, 2e-5), (4, 8e-5), (6, 1e-5), 8) if item['steps'] == 4 else (0, 1, (2, 1e-6), 4), # unfreeze strategy config
    random_resets=item['steps'] > 4, # enable random memory resets
    random_resets_from=2, # epoch when random resets starts
    random_resets_ratio=0.4 if item['steps'] != 4 else None, # probability of STM reset before episode
    separate_memory_lr=True, # use separate memory LR in current curriculum stage
    memory_lr=6e-4 if item['steps'] == 4 else None, # memory LR for curriculum stage, if None, use global config
    lr=3e-4 if item['steps'] == 4 else None, # model LR for curriculum stage, if None, use global config
    critic_lr=4e-4 if item['steps'] == 4 else None, # critic (head) LR for curriculum stage, if None, use global config
    critic_encoder_lr=2e-4  if item['steps'] == 4 else None, # critic (encoder) LR for curriculum stage, if None, use global config
    teacher_forcing=item['steps'] <= 8, # use teacher forcing - save reference answers from dataset in memory instead of generated ones
) for item in mrl_datasets]
```

After that, we have to configure reward model. It's based on BLEU scores and cosine similarity between generated answers
and saved data from previous steps and reference answers from dataset. Cosine similarity is also calculated from running
mean embedding of previous steps. Reward model also includes optional length reward. It's config includes a lot of option
to set different factors for different reward parts.
```python
# 1. Init GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Create reward model
reward_model = MrlRewardModel(
    encoder.model.embedding,
    device,
    bleu_with_saved_data=True, # use saved data (previous or first interaction) in BLEU calculation
    reward_len=True, # use length reward in calculation (answer_len / target_len)
    max_rewarded_len=None, # target length awarded as 1.0
    neg_reward_len=True, # negative length reward - lower reward when answer is too long (target_len / answer_len)
    target_len_as_ref=True, # use reference answer len as target
    use_running_mean=True, # use running mean embedding of all previous answers in cosine similarity calculation
    allow_not_summing_factors=False, # if True sum of reward factors could be different from 1.0, it's False by default
    bleu_factor=0.4, # factor for BLEU score in standard reward
    cos_factor=0.5, # factor for cosine similarity score in standard reward
    len_factor=0.1, # factor for length reward score in standard reward
    bleu_ref_factor=0.4, # factor for reference answer score in BLEU calculation (standard mode)
    bleu_saved_factor=0.6, # factor for saved data score in BLEU calculation (standard mode)
    cos_ref_factor=0.35, # factor for reference answer score in cosine sim calculation (standard mode)
    cos_saved_factor=0.65, # factor for saved data score in cosine sim calculation (standard mode)
    multi_cos_ref_factor=0.3, # factor for reference answer in multi-step cosine sim calculation
    multi_cos_saved_factor= 0.5, # factor for saved data in multi-step cosine sim calculation
    multi_cos_running_mean_factor = 0.2, # factor for previous answers running mean in multi-step cosine sim calculation
    neg_bleu_factor=0.45, # factor for BLEU score in negative reward
    neg_cos_factor=0.45, # factor for cosine similarity score in negative reward
    neg_bleu_ref_factor=0.3, # factor for reference answer score in BLEU calculation (negative mode)
    neg_bleu_saved_factor=0.7, # factor for saved data score in BLEU calculation (negative mode)
    neg_cos_ref_factor=0.3, # factor for reference answer score in cosine sim calculation (negative mode)
    neg_cos_saved_factor=0.7, # factor for saved data score in cosine sim calculation (negative mode)
    bleu_ref_weights=(0.2, 0.2, 0.3, 0.3), # weights for n-grams in NLTK BLEU calculation for reference answers
    bleu_saved_weights=(0.2, 0.2, 0.3, 0.3), # weights for n-grams in NLTK BLEU calculation for saved data
    tanh_reward_scale=False, # scale rewards to -1.0 to 1.0 range, instead of standard 0.0-1.0
    rewards_scale=1.0, # rewards scaling factor (reward * rewards_scale)
)
```

And finally, we could create the MRL Trainer with RL algorithm (currently only PPO available) and start the training:
```python
# 1. Init PPO Algorithm
algorithm = PPOAlgorithm(
  PPOConfig(clip_eps=0.2, gae_lambda=0.95, gae_gamma=0.99, entropy_coef=0.01, critic_value_clip=50.0)
)

# 2. Create config for MRLTrainer (most of MrlConfig fields could be overwritten in each curriculum stage)
mrl_config = MrlConfig(
    lr=1e-4, # main LR, used for decoder layers
    encoder_lr=2e-4, # encoder LR, used for encoder layers (if None, lr is used)
    critic_lr=2e-4, # critic LR, used for critic value head
    critic_encoder_lr=1e-4, # critic encoder LR (if not set, critic_lr is used)
    separate_memory_lr=True, # use separate LR for memory attention and memory cross-attention
    encoder_memory_lr=5e-4, # LR for encoder memory cross-attention (if None, memory_lr is used)
    memory_lr=3e-4, # memory LR, used for decoder memory cross-attention
    memory_attn_lr=5e-4, # memory attention LR (if None, memory_lr is used)
    max_seq_len=256, # maximum length of single interaction
    critic_max_len=512, # maximum length of critic sequence (have to be longer than actor's context)
    weight_decay=0.01, # weight decay for actor AdamW optimizer
    critic_weight_decay=0.01, # weight decay for critic AdamW optimizer
    update_epochs=10, # inner PPO update epochs
    pad_token_id=0, # tokenizer padding token id
    end_token_id=3, # tokenizer EOS token id
    use_moe_aux_loss=True, # add Mixture-of-Experts Router auxiliary loss to policy loss
    freeze_embeddings=False, # freeze pre-trained embeddings for MRL training
    embedding_lr=5e-6, # LR for embeddings, if not frozen (if None, lr is used)
    use_memory_warmup=False, # memory warmup - update memory with first interaction in no grad mode, before episode, for better initialization
)

# 3. Initialize MRL Trainer
trainer = MRLTrainer(
    actor, critic, reward_model, device, mrl_config, algorithm,
    use_amp=True, # use autocast in MRL Training
    dtype=torch.bfloat16, # data type for MRL
    use_ddp=False, # use distributed training with DDP
)

# 4. Train with curriculum stages config
trainer(curriculum_stages, batch_size=batch_size)
```

## Experimental attention layers
While working on reactive architectures, we also developed several new types of attention layers, some of which achieve
very promising results. Even considering that reactive models, processing single interactions, have much lower computational
requirements, we need the most efficient attention mechanisms, consistent with memory requirements. Since memory is not a
sequence but a set, spatial sparsity is probably not a good solution here, so we were looking for an efficient alternative
to Flex Attention with full access to all memory positions. New attention layers are implemented in `rxnn.experimental.attention`
module:
- **Grouped Mixture-of-Experts Attention (GMA)** - use MoE routing to dynamically select K active key/value heads for each token, instead
  of using static selection in **GQA**. While it's theoretically interesting, in practice, it achieved worse results than **GQA**,
  and even **MQA**, in all test, and is a lot slower because of routing overhead, so we abandoned further research. More details
  in [research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/moe_attention.md)
- **Deep Mixture-of-Experts Attention (DMA)** - extends **GMA** with the same MoE routing for query heads. Like **GMA**,
  it gives even worse results, and all the computational performance benefits from the sparse query heads (like in
  **SQA**) are lost by routing overhead (lack of specialized kernels for heads selection), so the further research is also
  abandoned. [Research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/moe_attention.md)
- **Hierarchical MoE Attention (HMA)** - extends **DMA/GMA**, using different number of query/key/value heads for tokens with
  different priority. It's only the idea and is not implemented, because of poor results of GMA/DMA. [More info](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/hierarchical_moe_attention.md)
- **Sparse Query Attention (SQA)** - the most trivial extension to GQA, reducing not only the number of key/value heads, but
  also the number of query heads. It results in even 2-3x faster model (for 32k/131k tokens). **SQA** is the fastest attention
  mechanism for 0-131k sequence length, for longer sequences **Flex Attention** becomes faster. That's ideal for reactive models,
  that doesn't need a million token context for single interaction processing. In tested cases **SQA** models results (loss/accuracy)
  were close to GQA, differences were almost unnoticeable, but it still requires more tests. [Research docs](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/sparse_query_attention.md)
- **Flex Sparse Query Attention (Flex-SQA)** - **Flex Attention** combined with **SQA** - enable handling 4-8x longer sliding
  windows, in shorter time, than base **Flex**, so it should result in better results. **Flex-SQA** should be the fastest
  attention mechanism for sequences longer than 131k tokens and is made for classic transformers, or potentially self-attention
  in bigger reactive models. Currently, it's viable only with symmetric variants of **SQA** (same number of used query
  and key/value heads), because kernels aren't compatible with GQA in sliding windows and not symmetric variants is 2x slower,
  than it should be. Docs and tests in progress

### Test usage
Experimental attention layers could be tested with `ExperimentalAttentionTransformer` model from `rxnn.experimental.models`,
Usage example could be found in our notebooks repository - [RxNN Notebooks](https://github.com/RxAI-dev/rxnn-notebooks)
