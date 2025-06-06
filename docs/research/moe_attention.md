# Extending Grouped Query Attention with Mixture-of-Experts
by Adam Filipek/Reactive AI
## Abstract
The main problem with Multi-Head Attention (MHA) is that it is not scalable for long sequences, because of its quadratic computational
complexity. Grouped Query Attention (GQA) and Multi Query Attention (MQA) are the most popular solutions to reduce this overhead.
Standard MHA is splitting the input tokens into N heads and applying an attention mechanism to each head. GQA and MQA are using
a smaller number of key/value heads than query heads. For MQA it's a single head, while for GQA it's a group of heads. Thanks
to it, scaled dot product attention is calculated between smaller dimensions, which makes it much more efficient, reducing its
computational complexity to the linear scale.

However, while GQA and MQA provide performance comparable to standard MHA, part of the information is lost. Smaller, static
key/value heads cannot provide full information, because of dimensionality reduction.

Mixture-of-Experts (MoE) is commonly used to dynamically select the weights (expert) that will be used to process each token.

In this research, we propose two new attention mechanisms that are using Mixture-of-Experts to dynamically select active
attention heads that will be used to process each token.

## Grouped Mixture-of-Experts Attention (Grouped MoE Attention/GMA)
Grouped MoE Attention is a simple extension of GQA/MQA. Its main goal is to reduce the information loss for key/value heads
without increasing the computational complexity. The only difference is that it has the same number of total key/value heads
as query heads, but only a group of selected active heads is used to calculate the attention for each token. Active heads are
selected using a Mixture-of-Experts routing mechanism. The rest is the same as in GQA/MQA.

### Algorithm steps
1. Process `query` tensor with Linear projection and split it into `num_heads` for attention calculation.
2. Process `key` tensor with MoE Router - Linear layer with Softmax activation, which selects `num_groups` active heads for each token.
3. Process `key` and `value` tensors with MoE layers - each token is processed by `num_groups` active heads selected by router.
4. Aggregate processed tensors for attention calculation (each position should have `num_groups` results from correct expert heads)
5. (Optional) Apply RoPE to `query` and `key` tensors.
6. Calculate attention results using scaled dot product attention.
7. Apply output Linear projection.

## Deep Mixture-of-Experts Attention (Deep MoE Attention/DMA)
Deep MoE Attention is extending GMA furthermore by using the same dynamic head selection also for query heads. However, it has
a different goal — reduce the computational complexity even more, without decreasing the performance. In this approach, only a
part of query heads are used to find corresponding keys/values. Each token could attend to every other token, but using only a
dynamically selected part of its information. Because only active query heads are used, results will have reduced dimensions,
so the output projection should transform it back to the original dimensions.

This solution could be a great option for very large context sizes, because of its computational complexity reduction. Compared
to MHA `O(N*d * N*d)`, GQA `O(N*d * N*(d/heads*groups))` and MQA `O(N*d * N*(d/heads))` complexity, it's `O(N*(d/heads*query_groups) * N*(d/heads*groups))`.

> I considered naming it Sparse MoE Attention, but sparse attention is rather known from spatial sparsity, so it could be misleading.

### Algorithm steps
1. Process `query` tensor with MoE Router (query router) - Linear layer with Softmax activation, which selects `num_query_groups` active query heads for each token.
2. Process `query` tensor with MoE layers - each token is processed by `num_query_groups` active query heads selected by router.
3. Aggregate processed tensors for attention calculation (each position should have `num_query_groups` results from correct expert query heads)
4. Process `key` tensor with MoE Router - Linear layer with Softmax activation, which selects `num_groups` active key/value heads for each token.
5. Process `key` and `value` tensors with MoE layers - each token is processed by `num_groups` active key/value heads selected by router.
6. Aggregate processed tensors for attention calculation (each position should have `num_groups` results from correct expert heads)
7. (Optional) Apply RoPE to `query` and `key` tensors.
8. Calculate attention results using scaled dot product attention.
9. Apply output Linear projection - transform from reduced dimensions (`model_dim` / `num_heads` * `num_query_groups`) to original.

## Implementation
We have implemented new attention layers in our [RxNN framework](https://github.com/RxAI-dev/RxNN) in `rxnn.experimental.attention`
module. It's compatible with any PyTorch model (it's recommended to use RxNN RoPE implementation).

## Computational complexity & optimization
Generally, using 2x less active query heads, like in our test case, reducing scaled dot product attention complexity x2, but it's
adding additional routing and gathering overhead. Execution time reduction is clearly visible when running the models on CPU, the DMA
version is ~2x faster than GQA, but when it comes to GPU real training time was even ~3% worse than GQA. The reason is that GPU kernels
are directly optimized for attention calculation, but not for routing and gathering operations, therefore, their overhead completely
reduces the benefits. Then it needs further optimization, which is out of the scope of this research.

## Evaluation
Because of the limited budget I only checked new layers on very small models:
- Micro - ~2.5M Params
- Mini - ~11-12M Params

### Micro architecture details:
- dim: 128
- layers: 6
- heads: 8
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- context size: 256
- self-attention: MHA/GQA/MQA or GMA/DMA
- gqa groups: 2
- dma query groups: 4

### Layer sizes
- GMA - 66.7k params
- DMA - 59.5k params
- MQA - 37k params
- GQA - 41k params
- MHA - 65.7k params

### Execution times on CPU (100 batches * 32 sequences)
- GMA - 41.19s
- DMA - 20.80s
- MQA — 39.83s
- GQA - 46.63s
- MHA - 47.05s

### Full models sizes
- GMA - 2.58M
- DMA - 2.53M
- MQA — 2.4M
- GQA - 2.42M
- MHA - 2.57M

### Results for micro models
Trained on single epoch, using `roneneldan/TinyStories` dataset (totally 8279 steps), 5e-4 lr with cosine annealing schedule and 25% warmup steps

- Model: validation loss / accuracy / time
- MHA: 1.5569 / 63.08% / 29.5 min
- GQA: 1.5668 / 62.94% / 28.8 min
- MQA: 1.6054 / 62.21% / 28.9 min
- GMA: 1.6384 / 61.85% / 29.6 min
- DMA: 1.6665 / 61.50% / 29.9 min

I tried also extended version, with 2 x more experts than heads (8 heads / 16 experts) and active 8 query/key/value heads, but it performs
even worse:
- xDMA: 1.7029 / 60.90% / 32 min

Then I proceed to bigger models

### Mini architecture details:
- dim: 256
- layers: 8
- heads: 16
- SwiGLU feed forward with 768 dim
- RoPE
- RMS Norm
- vocab: 10k (english only)
- context size: 1024
- self-attention: MHA/GQA/MQA or GMA/DMA
- gqa groups: 4
- dma query groups: 8

### Full models sizes
- GMA - 12M
- DMA - 11.8M
- MQA — 11M
- GQA - 11.2M
- MHA - 12M

### Results for mini models
Trained on single epoch, using 50% of `wikimedia/wikipedia` dataset (45% for training, 5% for validation, totally 22526
training steps), 5e-4 lr with cosine annealing schedule and 25% warmup steps

- Model: validation loss / accuracy / time
- MHA: 1.1976 / 77.35% / 269.3 min
- GQA: 1.2177 / 77.12% / 258.2 min
- MQA: 1.2497 / 76.64% / 260.8 min
- GMA: 1.2977 / 76.04% / 272.7 min
- DMA: 1.3660 / 75.09% / 266.9 min

## Summary
According to results, using Mixture-of-Experts to dynamically select active attention heads is not leading to better results.
Regular GQA and even MQA are achieving better results, without additional routing overhead and finally are also faster.

However, as reduced number of active query heads in DMA has small performance difference, compared to GMA, I tried also
the other variant, that I'm calling **Sparse Query Attention**, that reduces query heads count statically. It achieved
probably the best possible cost-effective results from all the GroupedQueryAttention: outperforming MQA, with almost
GQA-level performance (0.01 loss difference) and ~10% training time/cost reduction. [More details](./sparse_query_attention.md)
