# Sparse Query Attention - the most cost-effective variant of grouped attention
by Adam Filipek/Reactive AI

## Abstract
In my experiments with new attention layer types, I accidentally discovered that reducing used query heads count, instead
of further reduction of key/value heads (from GQA as a starting point, up to MQA) leads to much better results. In tests,
models with different variants of **Sparse Query Attention (SQA)** achieved performance between GQA and MQA (with some variants
on GQA performance level), but they are always noticeable faster, and the computational efficiency difference is getting bigger
for models with longer contexts. **SQA** has also smaller number of params, because of additional reduction of output projection
dimensions.

> Due to limited budget, I was only able to train (on a very limited dataset) and test two groups of small models with
> short context. Apart from that, I only tested longer sequences for computational efficiency, in order to compare with
> trained models with short sequences. I'm planning to further explore the topic in future research.

Generally, in training tests with short context lengths (256/1024), the computational performance difference was really
noticeable - ~3-10%, while the efficiency decrease (compared to GQA, when comparing to MQA, it's increase) is rather
unnoticeable (~0.1–0.2% accuracy).

However, in pure computational efficiency benchmarks with current State-of-the-Art context sizes (32k/128k), **Sparse
Query Attention** is even 2-3x faster than reference **GQA**/**MQA** and **MHA**. This results suggests, that **SQA**
could be a gamechanger and viable alternative to **Flex Attention** (it could be also combined) without _spatial sparsity_.

### Computational Efficiency - Theory and Practice
The results of the study challenge the generally accepted claim that reducing the number of key/value heads (or further
reducing from GQA to MQA) in the attention layers leads to greater computational benefits, than reducing the number of
query heads. This was explained by the fact that the number of key/value heads affects two matrix multiplications in
the attention operation, while the query heads only affect one. This is completely untrue - it seems that the influence
of query heads is crucial for both multiplications, due to the reduction in the dimensionality of the result of the
first multiplication. Thanks to this, both multiplications are performed in smaller dimensions. In traditional GQA and MQA,
regardless of the number of key/value heads, both multiplications are performed in full dimensionality, because the result
of the first multiplication takes over the dimensions of the query heads.

It seems that reducing the dimensionality of both matrix multiplication operations in _scaled dot product attention_ is
crucial for the speed of operation in practice. Even SQA variants with theoretically higher computational complexity
than MQA are much faster than MQA.

The difference in computational efficiency between SQA and GQA/MQA is also larger than the difference between
GQA/MQA and classical MHA

#### Computational complexity comparison
- MHA: `O(N*d * N*d)`
- GQA `O(N*d * N*(d/heads*groups))`
- MQA `O(N*d * N*(d/heads))`
- SQA `O(N*(d/heads*query_groups) * N*(d/heads*groups))`

### Difference from GQA
**Sparse Query Attention** layer is almost the same as **Grouped Query Attention** with some noticeable differences:
- Additional hyperparam for number of used query heads
- Query Linear projection has reduced output dimensions (`model_dim / num_heads * num_query_heads`)
- Output Linear projection has reduced input dimensions (`model_dim / num_heads * num_query_heads`)
- As _scaled dot product attention_ is calculated in smaller dimensionality, the output projection has to transform results back to original dimensions
- Symmetric variants could use standard **MHA** optimizations (same number of query and key/value heads)

> In **SQA** "number of heads" is used only for model dimensions split - number of all used/active query/key/value heads is reduced

### Variants
I tried different variants, that I named:
- SQA: standard variant, GQA extension, with 2x less active query heads
- sSQA: symmetric variant, using 50% of heads for both query and key/value heads
- xSQA: extreme variant, reducing the number of query heads even more than 2x
- xSMQA: extreme Sparse Multi Query Attention, extreme variant of SQA with only single key/value head

## Tests
As I mentioned, I tested two micro-size model groups. First case was on dense models with ~10-12M params and 1024 context size,
trained on single epoch (~22k steps) on 50% of `wikimedia/wikipedia` dataset (english subset) - it includes also _Multi Head Attention_
and 4 variants of **SQA**. Second one was with smaller Mixture-of-Experts models with ~8.5M params and 256 context size, that
were trained on 5 epochs (~40k steps), using small synthetic `roneneldan/TinyStories` dataset - I skipped MHA and **xSMQA** variant
in this case.

### Dense models (~10-12M)

#### Architecture details:
- dim: 256
- layers: 8
- heads: 16 (for dimension split)
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- context size: 1024
- self-attention: MHA/GQA/MQA + SQA/sSQA/xSQA/sSMQA
  - 4 key/value heads for GQA/SQA/xSQA
  - 1 key/value head for MQA/xSMQA
  - 8 key/value heads for sSQA
  - 8 query heads for SQA/sSQA
  - 4 query heads for xSQA/xSMQA
- sizes: 12M MHA / 11.2M GQA / 11M MQA / 10.7M SQA / 10.9M sSQA / 10.4M xSQA / 10.2M xSMQA
- dataset: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (50% - 45% train/5% validation)
- single epoch, 5e-4 LR, cosine annealing schedule with 25% warmup steps
- GPU: 1 x L40S 48GB GPU

#### Results
Validation mean loss/accuracy:
- MHA: 1.1976 / ~77.35%
- GQA: 1.2177 / ~77.12%
- MQA: 1.2497 / ~76.64%
- SQA: 1.2272 / ~76.97%
- sSQA: 1.2201 / ~77.05%
- xSQA: 1.2428 / ~76.74%
- xSMQA: 1.2815 / ~76.22%

Training time / time per batch:
- MHA: ~269 min / 0.7173s
- GQA: ~258 min / 0.6877s
- MQA: ~261 min / 0.6947s
- SQA: ~241 min / 0.6417s
- sSQA: ~243 min / 0.6468s
- xSQA: ~235 min / 0.6251s
- xSMQA: ~235 min / 0.6250s

All variants of SQA achieved about 20-min shorter training time compared to GQA/MQA. The same training time for xSQA and xSMQA
suggests that there's no sense to reduce key/value heads count after some point. The extreme version of SQA seems to be the fastest,
with performance still little better than MQA, while both SQA and sSQA have the best cost-effective results, with performance almost
the same as GQA.

#### Models
Models from experiments are published on HuggingFace Hub:
- [SQAT-m](https://huggingface.co/ReactiveAI/SQAT-m): 8/16 query heads, 4/16 kv heads
- [sSQAT-m](https://huggingface.co/ReactiveAI/sSQAT-m): 8/16 query heads, 8/16 kv heads
- [xSQAT-m](https://huggingface.co/ReactiveAI/xSQAT-m): 4/16 query heads, 4/16 kv heads
- [xSMQAT-m](https://huggingface.co/ReactiveAI/xSMQAT-m): 4/16 query heads, 1/16 kv heads

### Micro Mixture-of-Experts models (~10-12M)

#### Architecture details:
- dim: 128
- layers: 6
- heads: 8 (for dimension split)
- Mixture-of-Experts Feed Forward
  - experts: 12
  - active experts: 2
  - SwiGLU feed forward with 256 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- context size: 256
- self-attention: GQA/MQA + SQA/sSQA/xSQA
  - 2 key/value heads for GQA/SQA/xSQA
  - 1 key/value head for MQA
  - 4 key/value heads for sSQA
  - 4 query heads for SQA/sSQA
  - 2 query heads for xSQA
- sizes: 8.67M GQA / 8.64M MQA / 8.57M SQA / 8.62M sSQA / 8.52M xSQA - with Mixture-of-Experts size differences are a lot smaller
- dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- 5 epochs, 2e-3 LR, cosine annealing schedule without warmup
- GPU: 1 x L4 24GB GPU

#### Results
Validation mean loss/accuracy:
- GQA: 1.139 / ~70.66%
- MQA: 1.158 / ~70.33%
- **SQA: 1.159 / ~70.32%**
- **sSQA: 1.142 / ~70.63%**
- **xSQA: 1.169 / ~70.12%**

Total training time:
- GQA: ~398 min
- MQA: ~399 min
- **SQA: ~387 min**
- **sSQA: ~390 min**
- **xSQA: ~383 min**

That results suggest that even with very short sequences (256) the computational benefits are noticeable (~3%), while
the performance differences are very small (~1%). **sSQA** configuration has only ~0.3% worse loss, while it's 2% faster.
However, in bigger models with 1024 context size, the computational differences were greater (~10%), and **SQA** performance
was even better

#### Models
- [GQA-Ref-Micro](https://huggingface.co/ReactiveAI/GQA-Ref-Micro): 8 query heads, 2/8 kv heads
- [MQA-Ref-Micro](https://huggingface.co/ReactiveAI/MQA-Ref-Micro): 8 query heads, 1/8 kv heads
- [SQAT-mm](https://huggingface.co/ReactiveAI/SQAT-mm): 4/8 query heads, 2/8 kv heads
- [sSQAT-mm](https://huggingface.co/ReactiveAI/sSQAT-mm): 4/8 query heads, 4/8 kv heads
- [xSQAT-mm](https://huggingface.co/ReactiveAI/xSQAT-mm): 2/8 query heads, 2/8 kv heads

### Computational efficiency benchmarks
Tested on Dense models (~10-12M params) with different sequence length, on single L4 24 GB GPU.

> Results are sorted from the fastest one. Time (in seconds) is for 50 steps and per single step

#### 1024 Sequence / 128 batch size / 50 steps
1. xSQA: 4.7669 / 0.2383
2. SQA: 5.2710 / 0.2635
3. sSQA: 5.3359 / 0.2668
4. MQA: 5.7171 / 0.2859
5. GQA: 5.8207 / 0.2910
6. MHA: 6.4760 / 0.3238

#### 4096 Sequence / 32 batch size / 50 steps
1. xSQA: 5.1942 / 0.2597
2. SQA: 6.0414 / 0.3021
3. sSQA: 6.0466 / 0.3023
4. MQA: 7.0939 / 0.3547
5. GQA: 7.1983 / 0.3599
6. MHA: 7.9115 / 0.3956

#### 32k Sequence / 4 batch size / 50 steps
1. **xSQA: 8.6792 / 0.4340**
2. **sSQA: 12.2262 / 0.6113**
3. **SQA: 12.9228 / 0.6461**
4. MQA: 19.2118 / 0.9606
5. GQA: 19.4045 / 0.9702
6. MHA: 20.0299/ 1.0015

#### 128k Sequence / 1 batch size / 50 steps
1. **xSQA: 20.3782 / 1.0189**
2. **sSQA: 32.9969 / 1.6498**
3. **SQA: 36.0181 / 1.8009**
4. MQA: 60.5910 / 3.0295
5. GQA: 60.7724 / 3.0386
6. MHA: 61.4984 / 3.0749

Benchmark results are really surprising - from tests on shorter sequences, where it was up to 10% difference, I expected
that for longer ones it will be up to 20-30% difference, but as you can see in the results, for 128k sequence, **xSQA**
is about 3x faster! If we confirm that performance results for that sequence lengths are also on **GQA/MQA** level, it
could be a gamechanger for training costs.

> It's interesting, that for longer sequences **sSQA** is becoming faster than **SQA**, while it has 2x more key/value
> heads - that's probably caused by better optimization for the same number of query and key/value heads, like in **MHA**.

## Summary
According to experiment results, **Sparse Query Attention** seems to be the most cost-effective variant of **Grouped Query Attention**,
leading to noticeable training time reduction (even for very small context) and is a promising research direction. It should be tested
on very long context models, but this was out of scope of the current research. We will surely continue exploring SQA, but now we are
mostly concentrated on out reactive architectures.

I'm really surprised that this isn't a widely used attention variant yet. In terms of computational complexity and training time/cost
reduction, it seems that the biggest difference is made by reducing the number of query heads used by a factor of 2. At the same time,
this provides a much smaller performance/quality penalty than further reducing the number of key/value heads. It requires further research,
but the first results are suggesting that it could be viable, great performance alternative to classic GQA and MQA

#### Implementation & Test Code
**Sparse Query Attention** implementation (for **PyTorch**) is available in [RxNN framework](https://github.com/RxAI-dev/RxNN)
from `0.1.55` version, in `rxnn.experimental.attention` module. Tests were performed in `0.1.59` version, with **PyTorch** `2.6.0`
and **Flash Attention** `2.7.4.post1`.

**Jupyter** notebook with experiments is available in [rxnn-notebook](https://github.com/RxAI-dev/rxnn-notebooks) repository (`Experimental-Attention-Check.ipynb`).
The notebook includes also the code for abandoned [Grouped Mixture-of-Experts Attention research](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/moe_attention.md).