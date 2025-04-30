# Hierarchical Mixture-of-Experts Attention - dynamic attention head selection based on token importance
by Adam Filipek/Reactive AI

> Because of poor results of Mixture-of-Experts based attention ([Group MoE Attention and Deep MoE Attention](./moe_attention.md)), I will
> not work on Hierarchical variant at this time. Maybe I'll back to it some time, but for now I'll leave the algorithm description, maybe
> someone will try to experiment with it. Anyway, according to my experiment's results I'm recommending trying
> [SparseQueryAttention (SQA)](./sparse_query_attention.md) as a viable alternative for GQA and GQA with the biggest computational
> complexity reduction (better results than MQA, ~10% faster).

## Abstract
In my previous research, I used Mixture-of-Expert routing for dynamic attention heads (experts) selection. In that approach, and in all other
popular solutions, all tokens are treated the same, with the same importance. However, in Natural Language Processing, not all tokens have
the same importance - some words are critical for a sequence, but others, like stop words, are not important at all. Using the same amount of
information for each token may be not optimal. Hierarchical Mixture-of-Experts Attention is introducing more dynamic hierarchical routing,
assigning different number of query and key/value heads to each token, based on its importance


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
