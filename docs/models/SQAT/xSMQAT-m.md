---
license: apache-2.0
pipeline_tag: text-generation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- RxNN
- SparseQueryAttention
- SQA
language:
- en
datasets:
- wikimedia/wikipedia
library_name: RxNN
---

# xSMQAT-m: extremely Sparse Multi Query Attention Transformer mini
Research model for [Sparse Query Attention](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/sparse_query_attention.md)
experiments - extension to Grouped Query Attention, that's also reducing the number of used query heads, instead of further
reducing key/value heads count (up to Multi Query Attention). That approach results in huge computational complexity reduction
and much faster training, while the performance stays on GQA level (almost unnoticeable decrease, when compared to GQA, and
noticeable better than MQA).

> This version is using an extreme 4x query heads reduction factor with a single key/value group to combine Multi Query
> Attention with Sparse Query Attention. However, further key/value head reduction doesn't bring any training time reduction,
> compared to [xSQA](https://huggingface.co/ReactiveAI/xSQAT-m) and noticeably decreasing performance. So this solution is
> not recommended.

### Architecture details:
- trainable params: ~10.2M
- dim: 256
- layers: 8
- self-attention: Sparse Query Attention
  - heads: 16 (for dimension split)
  - query groups: 4
  - key/value groups: 1
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- message length: 1024
- Library: RxNN

### Training details:
This model was only trained for research purposes, on a small number of training steps.
- dataset: 50% from english subset of [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (45% train / 5% validation)
- single epoch
- 1.5B processed tokens
- learning rate: 5e-4, cosine annealing scheduler with 25% warmup steps

### Results
Validation mean loss/accuracy:
- MHA: 1.1976 / ~77.35%
- GQA: 1.2177 / ~77.12%
- MQA: 1.2497 / ~76.64%
- SQA: 1.2272 / ~76.97%
- _xSQA: 1.2428 / ~76.74%_
- **xSMQA: 1.2815 / ~76.22%**

Training time / time per batch:
- MHA: ~269 min / 0.7173s
- GQA: ~258 min / 0.6877s
- MQA: ~261 min / 0.6947s
- SQA: ~241 min / 0.6417s
- _xSQA: ~235 min / 0.6251s_
- **xSMQA: ~235 min / 0.6250s**

### Computational complexity comparison
- MHA: `O(N*d * N*d)`
- GQA `O(N*d * N*(d/heads*groups))`
- MQA `O(N*d * N*(d/heads))`
- SQA/xSQA `O(N*(d/heads*query_groups) * N*(d/heads*groups))`
- xSMQA `O(N*(d/heads*query_groups) * N*(d/heads))`

SQA has reduced two factors instead of one. That means it will better scale for longer sequences and training time gains
will be even greater.

### Model size difference
SQA has reduced dimensions of query heads linear projection and output projection, which results in a little smaller model sizes:
- MHA: 12M Params
- GQA: 11.2M Params
- MQA: 11M Params
- SQA: 10.7M Params
- xSQA: 10.4M Params
- **xSMQA: 10.2M Params**

### Usage
Model requires [RxNN framework](https://github.com/RxAI-dev/RxNN) for training/inference. It's integrated with HuggingFace Hub and libraries.

#### Inference:
- Install RxNN, PyTorch and dependencies: `pip install rxnn torch transformers tokenizers`
```python
import torch
from rxnn.experimental.models import ExperimentalAttentionTransformer
from rxnn.transformers.sampler import Sampler, SampleDecoder
from rxnn.training.tokenizer import load_tokenizer_from_hf_hub

model = ExperimentalAttentionTransformer.from_pretrained('ReactiveAI/xSMQAT-m')
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/xSMQAT-m')
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

model = ExperimentalAttentionTransformer.from_pretrained('ReactiveAI/xSMQAT-m')
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/xSMQAT-m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128 # Require ~40GB GPU Memory (trained on L40S)
epochs = 1
gradient_acc_steps = 1
seq_len = 1024
vocab_size = 10_000

peak_lr = 5e-4 * gradient_acc_steps

train_dataset = AutoregressiveLMDataset.from_hf_hub('hf-dataset-id', 'subset', tokenizer=tokenizer, max_seq_len=seq_len) # split is 'train' by default
valid_dataset = AutoregressiveLMDataset.from_hf_hub('hf-dataset-id', split='validation', tokenizer=tokenizer, max_seq_len=seq_len)

dataset_len = len(train_dataset)

steps_per_epoch = int(dataset_len / batch_size - 1)
total_steps = int((epochs * steps_per_epoch) / gradient_acc_steps)
warmup_steps = int(0.25 * steps_per_epoch)


logs_dir = './tensorboard_logs' # require tensorboard `pip install tensorboard`

print_cb = PrintLossCallback(batches_per_epoch=steps_per_epoch)
count_cb = TokenCounterCallback()
acc_cb = PrintAccuracyCallback()
save_cb = ModelSaveCallback('./path/to/save', push_to_hub=True,
                            hub_model_id='your-model-id', private_repo=True,
                            push_checkpoint_weights=True, final_commit_message='Final commit message', hf_token=YOUR_HF_TOKEN)

trainer = AutoregressiveTrainer(model, device, dataset=train_dataset, validation_dataset=valid_dataset,
                         vocab_size=vocab_size, callbacks=[print_cb, acc_cb, count_cb, save_cb], use_amp=True,
                         dtype=torch.bfloat16, log_dir=logs_dir, gradient_accumulation_steps=gradient_acc_steps)

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
scheduler = get_transformer_lr_scheduler(
    optimizer,
    warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

trainer(epochs=epochs, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler)
```

## Summary
According to experiment results, this variant of SparseQueryAttention has greater training time reduction gains and still
a little better performance than MQA. However, the regular and [symmetric](https://huggingface.co/ReactiveAI/sSQAT-m) version
of SQA seems to provide better cost-effective results - slightly slower, but almost the GQA level performance


