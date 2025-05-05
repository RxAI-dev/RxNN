# Reactive Transformer: Real-time processing for language models
by Adam Filipek/Reactive AI

## Abstract
Language models based on Transformer architecture are industry standard in Natural Language Processing. They are great
in language modelling and generative tasks. However, Transformers are completely stateless, and they are only emulating Short-Term
Memory with their long contexts — every processed sequence is completely separate. Conversational models based on this concept,
are processing all the conversation history on each message/interaction. Despite it being extremely inefficient, it's not how humans
thinking and communicating. Awareness is a stateful and continuous process that requires _Real-time Processing_, which is completely
opposite to how transformers working.

Previous approaches to memory in NLP, based on RNNs or even Neural Turing Machines, were concentrated on keeping state between
tokens in sequence, because it was the main problem for research in pre-transformer times. After transformer architecture release,
it no longer requires memory, thanks to attention layers. Then, it looks like the research community forgot about memory and
agreed that it's no longer necessary.

Someone even believed that transformers would achieve awareness only by scaling it furthermore to even bigger sizes. No, they
wouldn't, it's **impossible**. The key "feature" of awareness is that I rather know what I was doing or thinking 10 minutes
ago, without a need to read my whole-day history.

Awareness requires keeping state between sequences/interactions instead and processing only single messages in real-time,
with access to previous interactions by memory. It's only the first step to awareness, but this step is crucial.

In this research, we are introducing **Reactive Transformer** architecture and **Attention-based Short-Term Memory System**,
that's processing only single messages and moving conversation history into separate memory layers, accessed and updated
by specialized attention layers.

## Architecture
**Reactive Transformer** includes encoder and decoder, as in original transformer, but the execution order is reversed. First,
a message should be processed by decoder with the previous STM state, generating full response streamed out to
environment. Then, a concatenated message and response is passed to encoder, transforming it to latent memory spaces on
each layer. Finally, memory attention network is using encoded data to update Short-Term Memory layers. Both encoder and
decoder have memory-cross attention layers, placed between self-attention and feed forward, used to access memory state.

In base version, Generator Decoder is using Mixture-of-Experts (MoE) Feed Forward layers, while Memory Encoder is dense, to
balance models sizes - decoder autoregressive generation will take a lot much more computation time, than single memory
encoding, so decoder could use much bigger model. As number of layers, dimensions and embeddings should be the same for both
encoder and decoder, MoE is great option to balance sizes.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-moe.png" />

Alternative version could use regular dense decoder and could have similar sizes for both encoder and decoder. It has much
less params, but also worse performance and the training time is only a little shorter than MoE (that's becoming industry
standard), so it's less recommended. Mixture-of-Experts architecture seems to be a lot more natural - in biological neural
networks, only some part of neuron connections are activated, same as in MoE.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer.png" />

## Attention-based Memory System
According to the name, **Attention-based Memory System** is based on the idea of updating memory layers with attention
mechanisms. Fetch and update parts are separated and handled with dedicated components:
- Memory Cross-Attention is combining the processed sequence (query) with data from memory (key/values)
- Memory-Attention is combining memory layer's state (query) with an encoded message and response (key/values)

Each transformer's layer has its own connected memory layer, so encoder and decoder should have the same number of layers.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png">

# WORK IN PROGRESS
