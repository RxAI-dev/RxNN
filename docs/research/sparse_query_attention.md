# SparseQueryAttention as a viable and much faster alternative to GQA/MQA
by Adam Filipek/Reactive AI
## Abstract
In my experiments with new attention layer types, I accidentally discovered that reducing used query heads count, instead
of further reduction of key/value heads (from GQA as a starting point, up to MQA) leads to much better results. While the
model achieved better performance than the reference MQA model, close to the original GQA, it was also much faster, because
of the higher computational complexity reduction and has the smallest number of params, because of additional output layer
dimensionality reduction. Generally, the computational performance difference was really noticeable, while the efficiency
decrease (compared to GQA, when comparing to MQA, it's increase) is rather unnoticeable (~0.1–0.2% accuracy). It seems to be
a very promising direction, so I decided to run more experiments concentrated directly on SparseQueryAttention variants

### Variants
I tried different variants, that I named:
- SQA: standard variant, GQA extension, with 2x less active query heads
- sSQA: symmetric variant, using 50% of heads for both query and key/value heads
- xSQA: extreme variant, reducing the number of query heads even more than 2x
- xSMQA: extreme Sparse Multi Query Attention, extreme variant of SQA with only single key/value head

### Computational complexity comparison
- MHA: `O(N*d * N*d)`
- GQA `O(N*d * N*(d/heads*groups))`
- MQA `O(N*d * N*(d/heads))`
- SQA/sSQA/xSQA `O(N*(d/heads*query_groups) * N*(d/heads*groups))`
- xSMQA `O(N*(d/heads*query_groups) * N*(d/heads))`

### Architecture details:
- dim: 256
- layers: 8
- heads: 16
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- context size: 1024
- self-attention: MHA/GQA/MQA + SQA/sSQA/xSQA/sSMQA
  - 4 key/value groups for GQA/SQA/sSQA/xSQA
  - 8 query groups for SQA/sSQA
  - 4 query groups for xSQA/xSMQA
- dataset: `wikimedia/wikipedia` (50% - 45% train/5% validation)
- single epoch, 5e-4 LR, cosine annealing schedule with 25% warmup steps
- sizes: 12M MHA / 11.2M GQA / 11M MQA / 10.7M SQA / 10.9M sSQA / 10.4M xSQA / 10.2M xSMQA

### Results
Validation loss/accuracy:
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
the same as GQA

#### Models
Models from experiments are published on HuggingFace Hub:
- [SQAT-m](https://huggingface.co/ReactiveAI/SQAT-m)
- [sSQAT-m](https://huggingface.co/ReactiveAI/sSQAT-m)
- [xSQAT-m](https://huggingface.co/ReactiveAI/xSQAT-m)
- [xSMQAT-m](https://huggingface.co/ReactiveAI/xSMQAT-m)

## Summary
I'm really surprised that this isn't a widely used attention variant yet. In terms of computational complexity and training time/cost
reduction, it seems that the biggest difference is made by reducing the number of query heads used by a factor of 2. At the same time,
this provides a much smaller performance/quality penalty than further reducing the number of key/value heads. It requires further research,
 but the first results are suggesting that it could be viable, great performance alternative to classic GQA and MQA

## RESEARCH IN PROGRESS, RESULTS WILL BE PUBLISHED SOON
