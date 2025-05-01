# SparseQueryAttention as a viable and much faster alternative to GQA/MQA
by Adam Filipek/Reactive AI
## Abstract
In my experiments with new attention layer types, I accidentally discovered that reducing used query heads count, instead
of further reduction of key/value heads (from GQA as a starting point, up to MQA) leads to much better results. While the
model achieved better performance than the reference MQA model, close to the original GQA, it was also much faster, because
of the higher computational complexity reduction and has the smallest number of params, because of additional output layer
dimensionality reduction. Generally, the computational performance difference was really noticeable, while the efficiency
decrease (compared to GQA, when comparing to MQA, it's increase) is rather unnoticeable (~0.1–0.2% accuracy). It seems to be
a very promising direction, so I decided to run more experiments concentrated directly on SparseQueryAttention

## Results from previous experiments
Architecture details:
- dim: 256
- layers: 8
- heads: 16
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- context size: 1024
- self-attention: MHA/GQA/MQA + SQA + MoE (GMA/DMA)
- gqa groups: 4
- sqa query groups: 8
- dataset: `wikimedia/wikipedia` (50% - 45% train/5% validation)
- single epoch, 5e-4 LR, cosine annealing schedule with 25% warmup steps
- sizes: 12M MHA / 11.2M GQA / 11M MQA / 10.7M SQA 

Validation mean loss/accuracy:
- MHA: 1.1976 / ~77.35%
- GQA: 1.2177 / ~77.12%
- MQA: 1.2497 / ~76.64%
- SQA: 1.2272 / ~76.97%

Training time / time per batch:
- MHA: ~269 min / 0.7173s
- GQA: ~258 min / 0.6877s
- MQA: ~261 min / 0.6947s
- SQA: ~241 min / 0.6417s

## Computational complexity comparison
- MHA: `O(N*d * N*d)`
- GQA `O(N*d * N*(d/heads*groups))`
- MQA `O(N*d * N*(d/heads))`
- SQA `O(N*(d/heads*query_groups) * N*(d/heads*groups))`

## RESEARCH IN PROGRESS