# Reactive Transformer - Memory Attention: Architecture variants
by Adam Filipek/Reactive AI

## Abstract
In **Reactive Transformer** architecture, **Memory Attention** layers are responsible for updating **Short-Term Memory (STM)**
layers. Each transformer's (encoder and decoder) layer has its own separate memory layer, accessed by **Memory Cross-Attention**,
that require dedicated **Memory Attention** layer for updates. All those layers, combined with normalization and residual
connections are called **Memory Attention Network**, that is one of the main components of Reactive Transformer.
As **Memory Attention** is made for processing not human-interpretable memory states, it cannot be pre-trained with supervised
learning (unlike other components) and is included in training from reinforcement learning stages, especially **Memory Reinforcement
Learning (MRL)**.
During the MRL algorithms development, we designed the extensions of different complexity and expressiveness, made especially
to integrate the attention layers with memory system, that should improve the training and model's memory retention abilities.

> Different variants are currently in tests, so the research is still in progress

## Attention layers configuration
Base variant contain single memory attention layer, where current memory states are used (as query) to get connected information
from encoded interaction (keys/values), with pre-norm applied to both inputs and final residual connection, that's adding the
attention result to current STM state.
This configuration may lead to concentrating too much on new information from encoded interaction, overriding useful data
from current memory state. Adding additional attention layer (**Memory Self-Attention**), to select information from STM
layer itself, before looking for new encoded data, should help balancing the data sources in update. However, it should
be partially enabled by residual connection in base variant, so we should check, if more complex variant will provide
meaningful improvements.

Last variant is based on interlayer dependencies - normally, all the memory layers are independent and contain data on
different abstraction levels. The downside of that architecture design, is that memory layers may include duplicated
information, and the only connection between memory layers is through encoder's results. **Interlayer Memory Attention**
calculates attention between each STM layer (query) and mean from all memory layers (key/value) - it will provide the
information about the global state of memory to each layer. Interlayer mean includes also information from the layer itself,
so it may be partially complementary with memory self-attention. However, training should be strictly monitored, as it may
accidentally increase duplications instead reducing them.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-memory-attention.png" />

All the layers configurations could be combined with gated residual. More info below

## Residual gate
**Residual gate** is introducing additional control mechanism after the attention layers, and is made to improve the
balance of new and previous states in updates, and to improve the gradients flow in training. The gate decides how much
data from new updates state and previous state it should use for new memory state - it could be per layer or per memory
slot (recommended).

The most simple static gate result is based only on trained weights, and could be used to explicitly enforce the flow
through memory-connected components in first training stages, by setting initial weights values.

Dynamic gates are more complex and expressive - they calculate gate results from updated memory states and trained weights,
either by elementwise multiplication or by linear layer.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/residual-gate.png" />

#### Work in progress, more info soon
