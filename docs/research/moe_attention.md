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

In this paper, we propose three new attention mechanisms that are using Mixture-of-Experts to dynamically select active
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
to GQA `O(N * N/heads*groups)` and MQA `O(N * N/heads)` complexity, it's `O(N/heads*query_groups * N/heads*groups)`.

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

## Hierarchical Mixture-of-Experts Attention (Hierarchical MoE Attention/HMA)
Hierarchical MoE Attention is the most complex proposition that should provide the best results, with DMA level computational efficiency.
All previous solutions treat each token the same - use the same amount of information to calculate attention. That could be not optimal.
For example, stop words are not important at all, so maybe we could use less information for those tokens. On the opposite, some crucial
tokens could be processed with full information access. Hierarchical MoE Attention is based on this idea. In HMA routers are not only
selecting active query heads, but also deciding how many active heads should be used for each token. The possible choices are predefined,
separately for query and key/value heads. After routing, query and key/value tokens are divided into groups (with saved original order),
that are processed separately by different number of expert heads, selected by the router. Then, we have to calculate attention. For each
query group we have to calculate attention with each key/value group. Each result group has a different last dimension, so we need separate
output projections, that transform it back to the original dimensions. After that, the last thing to do is to merge the results back into
a single sequence (for each batch) with the original order.

With this approach, we could use full MHA level information for the most important tokens and only a small subset of heads for less
important tokens like stop words. With correct router load balancing, it will still have DMA level computational efficiency.

Generally, I'm expecting that DMA could have some small performance drop because of the smaller number of used query heads, even if
they are dynamic. HMA should fix this problem with a dynamic number of active heads, resulting in the best possible performance.

On the other hand, it's the hardest to implement and train, because of the additional routing complexity and handling multiple
groups of different dimensions.

### Algorithm steps
1. Process `query` tensor with MoE Router (query router) - Linear layer with Softmax activation, which decides how many query heads should be used
	for each token, selects those query heads and divides a sequence into groups (based on number of heads for token). Save original order.
2. Process divided `query` groups with MoE layers - each token is processed by the number of active query heads selected by router.
3. Aggregate processed tensors for attention calculation in groups
4. Process `key` tensor with MoE Router - Linear layer with Softmax activation, which decides how many key/value heads should be used
	for each token, selects those query heads and divides a sequence into groups (based on number of heads for token), with corresponding `value`
  tensors. Save original order.
5. Process divided `key` and `value` groups with MoE layers - each token is processed by the number of active key/value heads selected by router.
6. Aggregate processed tensors for attention calculation in groups
7. (Optional) Apply RoPE to `query` and `key` tensors in groups, but based on the saved original order, instead of the order in groups.
8. For each `query` group calculate (and sum) attention with each `key` and `value` group
9. Apply separate output Linear projections for each result group, transforming it back to the original dimensions.
10. Merge results back into a single sequence (for each batch) with the original order.

## Implementation
We have implemented all new three attention layers in our [RxNN framework](https://github.com/RxAI-dev/RxNN) in `rxnn.experimental.attention`
module. It's compatible with any PyTorch model (it's recommended to use RxNN RoPE implementation).

## Evaluation
Because of the limited budget I only checked new layers on micro size transformer models (~2.5M Params).

### Test architecture details:
- dim: 128
- layers: 6
- heads: 16
- SwiGLU feed forward with 384 dim
- RoPE
- RMS Norm
- vocab: 5k (english only)
- context size: 256
- self-attention: MHA/GQA/MQA or GMA/DMA/HMA

# RESEARCH IN PROGRESS