---
license: apache-2.0
pipeline_tag: text-generation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- RxNN
- SparseQueryAttention
- SQA
- GroupedQueryAttention
- MultiQueryAttention
language:
- en
datasets:
- roneneldan/TinyStories
library_name: RxNN
---

# SQAT-m: symmetric Sparse Query Attention Transformer Micro-MoE
Research model for [**Sparse Query Attention (SQA)**](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/sparse_query_attention.md)
research - extension to **Grouped Query Attention (GQA)**, that's also reducing the number of used query heads, instead of further
reducing key/value heads count, up to **Multi Query Attention (MQA)**. That approach results in huge computational complexity reduction
and much faster training, while the performance stays between **GQA** and **MQA** level.

> Symmetric **SQA** variant, is using exactly 50% of both query and kv heads. It has performance on reference GQA level, but
> is noticeable faster. [Check other variants](#compared-models)

##### Research paper in progress

### Architecture details:
- trainable params: ~8.62M
- dim: 128
- layers: 6
- self-attention: symmetric Sparse Query Attention (sSQA)
  - heads: 8 (for dimension split)
  - query groups: 4
  - key/value groups: 4
- Mixture-of-Experts Feed Forward
  - experts: 12
  - active experts: 2
  - SwiGLU feed forward with 256 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- context length: 256
- Library: RxNN

### Training details:
This microscale model was trained on 5 epochs on simple synthetic dataset, and is able to generate simple stories. The
main training goal is to compare it with reference GQA/MQA models and other SQA variants
- dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- 5 epochs
- 2.3B processed tokens
- learning rate: 2e-3, cosine annealing scheduler without warmup

### Compared models
- [GQA-Ref-Micro](https://huggingface.co/ReactiveAI/GQA-Ref-Micro): 8 query heads, 2/8 kv heads
- [MQA-Ref-Micro](https://huggingface.co/ReactiveAI/MQA-Ref-Micro): 8 query heads, 1/8 kv heads
- [SQAT-mm](https://huggingface.co/ReactiveAI/SQAT-mm): 4/8 query heads, 2/8 kv heads
- [sSQAT-mm](https://huggingface.co/ReactiveAI/sSQAT-mm): 4/8 query heads, 4/8 kv heads
- [xSQAT-mm](https://huggingface.co/ReactiveAI/xSQAT-mm): 2/8 query heads, 2/8 kv heads


### Results
Validation mean loss/accuracy:
- GQA: 1.139 / ~70.66%
- MQA: 1.158 / ~70.33%
- **SQA: 1.159 / ~70.32%**
- **sSQA: 1.142 / ~70.63%** <-
- **xSQA: 1.169 / ~70.12%**

Total training time:
- GQA: ~398 min
- MQA: ~399 min
- **SQA: ~387 min**
- **sSQA: ~390 min** <-
- **xSQA: ~383 min**

That results suggest that even with very short sequences (256) the computational benefits are noticeable (~3%), while
the performance differences are very small (~1%). **sSQA** configuration has only ~0.3% worse loss, while it's 2% faster.
However, in bigger models with 1024 context size, the computational differences were greater (~10%), while most **SQA**
variants were closer to GQA than MQA in performance

### Computational complexity comparison
- MHA: `O(N*d * N*d)`
- GQA `O(N*d * N*(d/heads*groups))`
- MQA `O(N*d * N*(d/heads))`
- SQA `O(N*(d/heads*query_groups) * N*(d/heads*groups))`

SQA has reduced two factors instead of one. That means it will better scale for longer sequences and training time gains
will be even greater, what's confirmed in little bigger models - [ReactiveAI/SQAT-m](https://huggingface.co/ReactiveAI/SQAT-m).

> Some **SQA** variants have theoretically higher complexity than MQA, but they are still faster. It's probably caused by
> a fact that for MQA/GQA, both matrix multiplications are working in full dimensional spaces - first factor in both multiplications
> has the same shape as full query heads. In the opposite, in SQA both multiplications are in reduced dimensions, result of the
> first multiplication has reduced dimensionality, what leads to a more efficient GPU utilization. Additionally, variants with
> the same number of used query and key/value heads could use most mature full Multi Head Attention optimizations. It's confirmed
> by all the computational performance benchmarks - **SQA is always faster**.

Even _the extreme version_ of **SQA** with only 2/8 used query heads (and also 2/8 key/value heads), seems to have similar performance
as a reference MQA model, with even shorter training times. However, further reduction below this level (~25% of heads used), doesn't
reduce training time/cost and noticeable decreasing performance, so there is some limitation. It suggests that **SQA** could be a
viable alternative to spatially sparse attention. More info in [ReactiveAI/xSQAT-mm](https://huggingface.co/ReactiveAI/xSQAT-mm).

### Model size difference
SQA has reduced dimensions of query heads linear projection and output projection, which results in a little smaller model size:
- GQA: 8.67M Params
- MQA: 8.64M Params
- **SQA: 8.57M Params**
- **sSQA: 8.62M Params** <-
- **xSQA: 8.52M Params**

> In these models, size difference is small because of MoE. In dense models the difference is more noticeable, check [ReactiveAI/SQAT-m](https://huggingface.co/ReactiveAI/SQAT-m)

### Usage
Model requires [RxNN framework](https://github.com/RxAI-dev/RxNN) for training/inference. It's integrated with HuggingFace Hub and libraries.

#### Inference:
- Install RxNN, PyTorch and dependencies: `pip install rxnn torch transformers tokenizers`
```python
import torch
from rxnn.experimental.models import ExperimentalAttentionTransformer
from rxnn.transformers.sampler import Sampler, SampleDecoder
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub

model = ExperimentalAttentionTransformer.from_pretrained('ReactiveAI/sSQAT-mm')
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/sSQAT-mm')
sampler = Sampler(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), end_token_id=3)
sample = SampleDecoder(sampler, tokenizer)

# 0.1 and 0.9 are default values for temperature and top_p
generated = sample('Example model input for text generation...', temperature=0.1, top_p=0.9, max_seq_len=1024)
sample('Example model input for text generation - print streamed response...', temperature=0.1, top_p=0.9, max_seq_len=1024, print_stream=True)
```

#### Train:
- Install RxNN, PyTorch and dependencies: `pip install rxnn torch transformers tokenizers tensorboard` (`tensorboard` is optional)
```python
import torch
from rxnn.experimental.models import ExperimentalAttentionTransformer
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub
from rxnn.training.dataset import AutoregressiveLMDataset
from rxnn.training.bml import AutoregressiveTrainer
from rxnn.training.callbacks import PrintLossCallback, PrintAccuracyCallback, TokenCounterCallback, ModelSaveCallback
from rxnn.training.scheduler import get_transformer_lr_scheduler

model = ExperimentalAttentionTransformer.from_pretrained('ReactiveAI/sSQAT-mm')
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/sSQAT-mm')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256
epochs = 5
gradient_acc_steps = 1
seq_len = 1024
vocab_size = 10_000

peak_lr = 2e-3 * gradient_acc_steps

train_dataset = AutoregressiveLMDataset.from_hf_hub('hf-dataset-id', 'subset', tokenizer=tokenizer, max_seq_len=seq_len) # split is 'train' by default
valid_dataset = AutoregressiveLMDataset.from_hf_hub('hf-dataset-id', split='validation', tokenizer=tokenizer, max_seq_len=seq_len)

dataset_len = len(train_dataset)

steps_per_epoch = int(dataset_len / batch_size - 1)
total_steps = int((epochs * steps_per_epoch) / gradient_acc_steps)
warmup_steps = 0


logs_dir = './tensorboard_logs' # require tensorboard `pip install tensorboard`

print_cb = PrintLossCallback(batches_per_epoch=steps_per_epoch)
count_cb = TokenCounterCallback()
acc_cb = PrintAccuracyCallback()
save_cb = ModelSaveCallback('./path/to/save', push_to_hub=True,
                            hub_model_id='your-model-id', private_repo=True,
                            push_checkpoint_weights=True, final_commit_message='Final commit message', hf_token=YOUR_HF_TOKEN)

trainer = AutoregressiveTrainer(model, device, dataset=train_dataset, validation_dataset=valid_dataset,
                         vocab_size=vocab_size, callbacks=[print_cb, acc_cb, count_cb, save_cb], use_amp=True,
                         dtype=torch.bfloat16, log_dir=logs_dir, gradient_accumulation_steps=gradient_acc_steps,
                         use_moe_aux_loss=True, moe_aux_loss_scale=0.01)

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
scheduler = get_transformer_lr_scheduler(
    optimizer,
    warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

trainer(epochs=epochs, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler)
```

## Summary
According to experiment results, **Sparse Query Attention** seems to be the most cost-effective variant of **Grouped Query Attention**,
leading to noticeable training time reduction (even for very small context) and is a promising research direction. It should be tested
on very long context models, but this was out of scope of the current research. We will surely continue exploring SQA, but now we are
mostly concentrated on out reactive architectures.

Currently, for our **Reactive Tranformer** architectures that were initially designed with GQA for self-attention and MQA for memory-attention,
we consider using SQA variants instead, for all attention layer types. More info will be released soon.
