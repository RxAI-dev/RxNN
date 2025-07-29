# Reactive Transformer: Real-time processing for language models
## Architecture introduction
Draft by Adam Filipek/Reactive AI (adamfilipek@rxai.dev)

> This is the first article from Reactive Transformer series and is introducing the architecture. In next articles, we
> will describe all the training stages and inference process. Drafts are available in RxNN docs:
> - [Supervised Training stages](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/supervised-training.md)
> - [Reinforcement Learning stages](https://github.com/RxAI-dev/RxNN/blob/main/docs/research/ReactiveTransformer/mrl.md)

## Abstract
Large Language Models (LLM) based on Transformer architecture are industry standard in Natural Language Processing. They are great
in language modelling and generative tasks. However, Transformers are completely stateless, and they are only emulating Short-Term
Memory with their long contexts — every processed sequence is completely separate. Conversational models based on this concept,
are processing all the conversation history on each message/interaction. Despite it being extremely inefficient, it's not how humans
thinking and communicating. Awareness is a stateful and continuous process that requires _Real-Time Processing_, which is completely
opposite to how transformers are working.

Previous approaches to memory in NLP, based on RNNs or even Neural Turing Machines, were concentrated on keeping state between
tokens in sequence, because it was the main problem for research in pre-transformer times. After transformer architecture release,
it no longer requires memory, thanks to attention layers. Then, it looks like the research community forgot about memory and
agreed that it's no longer necessary.

Instead, the research community focused on continually increasing context length, which only made sense up to a point. Due to
the quadratic growth (`O(N²T)` for conversation, where `N` is the number of messages/interactions and `T` is the number of
tokens in interaction) of LLM inference costs (each message is the cost of that message and all previous ones), in a very
long conversation, each subsequent message, even the shortest, carries a disproportionately large cost. This makes context
lengths of a million tokens or more completely pointless, because by the time you reach those million tokens in a conversation,
you'll have lost your entire budget.

Someone even believed that transformers would achieve awareness only by scaling it furthermore to even bigger sizes. No, they
wouldn't, it's **impossible**. The key "feature" of awareness is that I rather know what I was doing or thinking 10 minutes
ago, without a need to read my whole-day history. Awareness is a continuous, stateful and real-time process, while LLMs are
neither continuous, stateful nor real-time.

Awareness requires keeping state between sequences/interactions instead, and processing only single messages in real-time,
with access to previous interactions by memory. It's only the first step to awareness, but this step is crucial.

Those features are partially covered by agentic frameworks, like LangChain, that are integrating LLM models with external
tools, including memory or databases, through _prompt engineering_. But the models, even if agents are called "reactive",
still processing whole histories, instead working in real time. Their memory is not the model's memory, but the agent's memory.
Our new **Reactive Neural Networks** and **Event-driven AI** paradigms are moving those agentic features into the models architectures.

In this research, we are introducing **Reactive Transformer** architecture with **Attention-based Memory System for Short-Term
Memory**, that's processing only single messages in real time and moving conversation history into separate memory layers,
accessed and updated by specialized attention layers. **Reactive Language Models** based on this architecture have linear
costs scaling (`O(NT)`) and are `N` times cheaper and faster than LLMs (`N` is the number of messages/interactions).

## Reactive Neural Networks and Event-driven AI
_Reactive neural networks (RxNN)_ are event-driven, memory-aware neural networks with stateful real-time processing and infinite
inter-sequence recurrence (while RNN recurrence is mainly intra-sequence). While our new reactive architectures are based
on Transformers (that handle intra-sequence dependencies), it's not a strict requirement - it's possible for RNN to act
as RxNN, and it should be possible to create reactive Diffusion Language Model or other architecture - it's not a part of this research.

Reactive Neural Networks are made especially for conversational Natural Language Processing (and Multi-Modal Processing) tasks,
and their goal is to **progress from language models to awareness models**.

The related _"Reactivity Hypothesis"_ posits that in order to achieve **Artificial General Intelligence (AGI) and consciousness**,
the implementation of reactive neural network specifications is **required**.

_**Event-driven AI**_ is a radical paradigm shift from dominating _data-driven_ approach.

In _data-driven_ paradigm models are just concentrated on data processing - everything is a data - image, text sequence or
whole conversation history. Each data processing step is not connected with other steps (not counting recurrency in autoregressive
sequence generation - it's treated as single step), as the models are stateless entities used as a tools to pull the processed
data from them (pull-based processing).

The _event-driven_ approach treats all data processing as discrete or continuous events, with state preserved between events.
Models listen for environmental events (such as a new incoming message), processing data in response to those events and
emitting (pushing) new events to the environment (push-based processing) - we call it the **interaction**.  
According to this naming:
- **event** is a single part of **interaction** (model is getting input query or model is returning the response)
- **interaction** is a process of emitting the output **event** in reaction to the input **event**

> **Reactive language models** such as **Reactive Transformer** and **Preactor** (upcoming RxT extension) operate in discrete
> time steps (infinite), while the upcoming AGI awareness model **Reactor** operates in continuous time.

### Live Learning
**Event-driven AI** should be able to learn not only from data during training, but also from interactions during inference – this
concept is called **Live Learning**, and in the case of **Reactor**, it is extended to **Continuous Live Learning**, where the model
is even able to initiate interactions on its own and learn from them autonomously, without human intervention.

> **Strong Reactive Neural Networks (called also Infinite Reactive Neural Networks)**, like Reactor, have additional internal
> event loop, where they are both listeners and emitters. They are listening to both external and internal query events
> at once, and they are also emitting both external and internal response events at once, combining and processing
> everything in single interaction. External events are interpreted as a spoken words, while the internal events
> are representing model thoughts. Emitting external event may or may not result in next external query, but emitting
> internal event always results in new internal query, because the model itself is the only listener for those events.
> It means that every processed interaction, interpreted as thought process, is automatically initializing next
> interaction - the another thought process. It's the logical model of awareness, called _**Infinite Chain-of-Thoughts**_.

## Architecture
According to the name, **Reactive Transformer (RxT)** is a _**Reactive Neural Network (RxNN)**_ designed for stateful, conversational
_Natural Language Processing (NLP)_, that's extending the base _**Transformer**_ architecture. Just like the first, original
Transformer, it has encoder and decoder, but they have different roles and architectures. It has also additional _**Memory
Attention**_ network (Memory Attention layers with corresponding Memory (Pre-)Norm), so it's a little bigger ensemble. 

> Decoder, encoder and memory-attention network are called **components** in **Reactive Transformer** architecture

Compared to original Transformer, **Reactive Transformer** has reversed execution order:
- first, decoder is generating _answer_, based on input _query_ and _STM state_ (from previous interaction)
- then, encoder is transforming both _query_ and _answer_ to latent memory spaces and memory attention is updating memory

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-moe.png" />

Models based on **Reactive Transformer** architecture are called **Reactive Language Models (RxLM)**

### Shared Embedding
Both encoder and decoder have to share the same _embedding_ layer - they both have access to STM, that's connected to the
same _embedding_ space. It will reduce the parameter count in the final architecture and slightly improve a decoder quality (using
bidirectional MLM embeddings from encoder), but it's making the _tied embedding_ concept hard to implement (we're not using
it in our models). Embedding layer is pre-trained and fine-tuned with decoder and encoder in _Joint LM Training_. Then it
could be frozen or trained furthermore in Reinforcement Learning stages.

### Memory Cross-Attention
As in original encoder-decoder Transformer, decoder in Reactive Transformer has additional cross-attention layers between
self-attention and feed forward net, but it has a little different function - it's called _**Memory Cross-Attention**_.

In original Transformer, cross-attention in decoder is used to access the data encoded by encoder layers. In **RxT** it's
similar, memory cross-attention is still accessing encoder hidden states, but combined (accumulated) with all states from previous
interactions (by memory-attention). It has one crucial implication - _positional encoding_ - memory shouldn't have positional
encoding and should be treated rather as a _set_ than a _sequence_, because it's all updated incrementally - in each step
the message will include some different information on each position, so it shouldn't be reflected in the STM memory. Then,
as the _**Rotary Positional Embedding (RoPE)**_ is currently industry standard, in RxT's memory cross-attention we are applying
RoPE rotation only to _queries_ (input sequence) and not to _keys_ (STM layer state).

> Legacy positional embedding solutions (absolute/relative) aren't considered for Reactive Transformer. They will add,
> explicitly (relative) or implicitly (absolute), positional encoding to Short-Term Memory layers, then it shouldn't be
> a good choice.

#### Encoder Memory Cross-Attention in original architecture design
Initially, we planned to use **Memory Cross-Attention** also in encoder, to include information about current memory state
in encoded data. It was originally designed for upcoming **Long-Term Memory (LTM)** architectures (as **Reactive Transformer** is
the simplified **Reactor**, not the opposite), where encoded data is used not only to update **Short-Term Memory**, but also
for search in **Long-Term Memory** - then it will be good to not look for information already included in **STM**. But for
Reactive Transformer that has no **LTM**, it's not needed, because current **STM** state is normally accessed in **Memory
Attention**.

After experiments with **Memory Reinforcement Learning** we decided to remove **Memory Cross-Attention** from encoders
in all architectures, because of the crucial vector space misalignment issue that it's causing. The problem is that encoded
data of each layer and corresponding STM layer are connected to vector space of feed-forward layer results, but memory
cross-attention is before feed-forward - it breaks the encoder results and all the training. As I stated, it's not required
for **Reactive Transformer** and for the later **LTM** architectures it will be replaced by new **Long-Term Gates**.

### Memory Attention & Attention-based Memory System
Memory cross-attention provides the access to **Short-Term Memory** layers and integration with Transformer architecture. But
how to update memory? The _**Attention-Based Memory System (ABMS)**_ assumes that attention mechanisms are also responsible
for updating the memory state, in a process that is opposite to the memory cross-attention. In **Memory Attention** STM
states becoming _queries_, while the encoded sequences (encoder hidden states) are treated as _keys/values_. Before passing
STM to attention layers, it's normalized by _RMS_ based **Memory Norm** and the attention layer's result is combined with current
STM layer's state in **Residual Gate** - this design should ensure incremental STM state building.

As memory shouldn't be positionally encoded, in **Memory Attention** RoPE only _keys_ are rotated.

#### Memory Attention variants
We have designed multiple Memory Attention Network variants, with different attention layers configuration
- simple variant with single Memory Attention layer, combining STM state with encoded data (encoder layer's results)
- Memory Self-Attention variant with additional self-attention layer before, that's using STM layer data as query/keys/values
- Interlayer Memory Attention, that's combining STM layer data with mean from all layers, instead of self-attention
- Gated Self/Interlayer Memory Attention - first attention layer's keys/values are selected from mean STM interlayer data
  and current layer data by Residual Gate

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-memory-attention.png" />

Memory Self-Attention variant has been introduced to balance the contribution of data from the current and new states (the
first simple variant may favor the new data too much). Interlayer variant should help reducing duplication of information
in STM layers, while the Gated Self/Interlayer variant combines both approaches.

> In first experiments with **Memory Reinforcement Learning**, **Interlayer Memory Attention** variant achieved the best
> results in early training stages, so we selected it as a default variant for **RxT-Alpha** models.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-interlayer.png" />

#### Residual Gate
**Residual Gate** is introducing additional control mechanism after the attention layers, and is made to improve the
balance of new and previous states in updates, and to improve the gradients flow in training. The gate decides how much
data from new updates state and previous state it should use for new memory state - it could be per layer or per memory
slot (recommended).

Additionally, using normal residual connections (or tanh gate activation) for STM, that is accumulated on each step leads
to exploding STM updates. Each next STM update results in about 2x bigger numbers. Regularization based on MSE between new
and old STM state (implemented in Implicit Memory Policy Optimization algorithm) could fix this problem, by ensuring correct
balance of negative and positive values, but it could be also achieved naturally with sigmoid gate activation - it's using
weighted mean, instead of sum/weighted sum. In our **RxT-Alpha** models we are using sigmoid gate activation with small
memory diff regularization.

The most simple static gate result is based only on trained weights, and could be used to explicitly enforce the flow
through memory-connected components in first training stages, by setting initial weights values.

Dynamic gates are more complex and expressive - they calculate gate results from updated memory states and trained weights,
either by elementwise multiplication or by linear layer.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/residual-gate.png" />

#### Memory Layers
**Short-Term Memory** is divided into layers - each Reactive Transformer component's layer has corresponding STM layer, with
saved data on different levels of abstraction. While each STM layer has almost the same shape as the input sequence, with only
the second dimension different (sequence length/STM size - in sequence it could be variable, but STM has always same, constant size),
it's not equivalent - it's not a sequence with conversation history, but rather dense set/matrix, that's keeping incrementally
updated abstract representation of model memories.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/stm-abms.png">

So, in the summary, memory fetch and update mechanism are handled by opposite attention mechanisms:
- Memory Cross-Attention is combining the processed sequence (query) with data from memory (key/values)
- Memory-Attention is combining memory layer's state (query) with an encoded message and response (key/values)

### Mixture-of-Experts for decoder/encoder size balance
Looking at the State-of-the-Art Transformer decoder-only and encoder-only models, it's noticeable that encoders are a
lot smaller (millions of params) than decoders (billions of params). It looks like encoders tasks are a lot simpler, than
autoregressive text generation and require less complexity. However, in reactive, memory-based architectures both encoder
and decoder should have the same number of layers and model/embedding dimensionality, which makes it harder to scale. We
also need a strong encoder, with the same knowledge as decoder, to correctly transform data to memory spaces.

We decided, to use _**Mixture-of-Experts**_ feed forward layers for decoder, while encoder has regular dense feed forward. With
this approach, decoder could be N times bigger than encoder, so finally Reactive Transformer will be only slightly bigger
than classic, decoder-only MoE Transformer.

### Sparse Query Attention for efficient attention in Reactive Transformer
In our additional side research, we discovered, that reducing used query heads count (like using 8 heads from 16), instead
furthermore reduction of key/value heads from GQA level, up to MQA, leads to a lot better computational efficiency, and
almost the same model quality (loss/accuracy/perplexity, etc.) - we called this solution _**Sparse Query Attention**_. All
the SQA variants are a lot faster than GQA/MQA (even 2-3x for 128k tokens), while they performance is always somewhere
between GQA and MQA. Symmetric variants of SQA, that's using exactly 50% of query and key/value heads (use full MHA
optimizations) are still a lot faster than GQA with much smaller number of key/value heads, while their performance is almost
exactly the same. Extreme SQA variants, when there are used less than 50% heads for both query and key/value, are still on
MQA's performance level with the best possible computational efficiency. However, after some level - about 25% of each heads,
further reduction doesn't improve training times and performance is noticeable worse. SQA has also the smallest number of
parameters, because of an additional output projection input dimensionality reduction.

In summary, SQA seems to be the most cost-effective variant of Grouped Query Attention and the fastest attention mechanism
for 0-128k tokens - after 128k spatially sparse attention (Flex Attention) is becoming faster. However, symmetric variant of
**Sparse Query Attention** could be easily combined with **Flex Attention**, to enable 4-8x greater sliding window lengths.
**SQA** is a default choice in our experimental **Reactive Transformer** models - **RxT-Alpha**.

More about [Sparse Query Attention](../sparse_query_attention.md)

#### Spatially sparse attention for memory
Spatially sparse attention mechanisms, based on sliding local windows, especially _**Flex Attention**_, are becoming more
popular, because of their ability to handle very long sequences with `O(N * log N)` complexity. They are excellent for very
long contexts extending a million tokens, but as I mentioned in the introduction, the usefulness of such a long context is
highly questionable due to the quadratic growth of inference costs - and it's rather the opposite direction of research.

In **Reactive Transformer** that lengths, like million or more tokens, for single messages, are rather against the real-time
processing concepts (reading a book in real-time mode will be rather partial, and it's natural - people rather don't read whole
books at once). We want to achieve those conversation lengths differently with memory - "million-token like" conversation
should be achieved with about 32k message length and 32-64k memory layer size. Then, **Reactive Transformer** may just not need
spatial sparsity for self-attention, except some rare cases with very long single interactions - it will be handled by the
combination of **Flex Attention** and memory attention/cross-attention (sliding windows for input sequences, that are attending
to memory states of constant size).

While spatial sparsity could be optionally used for self-attention, it rather isn't a good idea for memory. Spatial sparsity
is based on positional relations, but memory doesn't have positional encoding and could have completely different spatial
relations structure. In example, one local window could include information from different interactions (time steps), that
are strictly connected to items in other local window. Then, the structural sparsity provided by SQA seems to be a lot
better option - attention has access to all the elements, but using only partial information about relations, so it could
learn to select a part connected with particular interaction.

Additionally, STM has always full, constant size (opposite to input sequence, that's variable from 0 to max message length),
so memory cross-attention and memory attention computational cost will be almost always higher, than self-attention, so it
has to be as fast as possible, so SQA seems to be the best option.

> Before SQA discovery, we considered using GQA for self-attention, memory cross-attention and memory attention. They are
> of course still viable and available in our RxNN framework, but SQA just fits better to the overall architecture.

### Sampler
As **Reactive Transformer** algorithm includes accumulating generated tokens, it needs the **Sampler** as a core architecture
component - its arguments - temperature, top p or top k - are becoming model's hyperparams, used not only in inference, but
also in Reinforcement Learning stages. 

### Quasi Live Learning and Pseudo-Infinite Context
As the first event-driven reactive neural network, **Reactive Transformer** has no overall context limitation (only single
message/interaction length is limited) and the ability to learn from interactions, but it's limited by the memory capacity,
that according to its name is "short-term" and it's size is not extendable. Although the RxT memory is many thousands of
times larger than a typical RNN memory (in large-scale reactive models STM can reach gigabytes), it is still limited,
so even if the model learns new things that it stores in the memory, it will forget them after a certain number of subsequent
interactions. So even if the overall context is theoretically infinite, this does not guarantee a completely fluid conversation
with access to all the information from the past.

Considering the size of the memory, which should allow reactive models to have conversations many times longer than classical
LLMs with similar context size (similar to memory size), it is strange to call this "catastrophic forgetting" - the incremental
nature of memory updates should ensure a very fluid conversation and the storage of the most important details for a very
long time, but this is still a limitation.

However, this architecture also enables the use of new solutions, such as storing and sharing memory states learned about
specific tasks/knowledge, as an alternative to prompt engineering. We hope this will be developed by the community, while
we focus on solving the problem of forgetting by extending memory with an additional level - _**Long-Term Memory (LTM)**_,
in the upcoming **Preactor** architecture, which will enable true live learning and infinite overall context

### Reactive communication patterns (Real-Time Processing)
In the inference, full model flow is following:
0. Model has some initial STM state (random initialization or loaded pre-trained STM - it will extend the idea of system prompt from LLMs)
1. The input _query_ sequence is passed to the model and is being embedded
2. Embedded _query_ is processed by decoder, that's accessing current STM state with memory cross-attention and is autoregressively generating answer (with Sampler)
3. Generated answer is **streamed/returned** to the environment, but is also accumulated into the _answer_ sequence, that's being embedded
4. Embedded _query_ and _answer_ sequences are concatenated and processed by encoder
5. _Encoded data_ - hidden states from each encoder's layer, are combined with previous STM state by memory attention layers.
6. When memory is updated, model is ready for next interaction - so we are back in the step 0.

This approach require using asynchronous reactive communication patterns - when the decoder (with Sampler) finishes generating
the answer, it still has to update memory for the next interaction (in background, from user's perspective). So the model has to
send some additional communicates to the UI, to display/hide the information about memory update and/or enqueue new message/interaction,
when it's send before it finished memory update.

### Cloud-based Reactive Models
The stateful nature of reactive models may suggest that they are local-only models and aren't compatible with cloud-based
processing, which is the standard for actual models - memory from one user's conversation, cannot be used in other user's case.
It's partially true - event-driven models are made to handle single (or very small number) stream of interactions (while for
data-driven models each prompt is always single instance of data), but storing and loading STM states dedicated per user/conversation
is only the infrastructural challenge. There are two main options:
- store STM in database, indexed by user or dialog id, and load it on each user's request - recommended solution
- store STM on user machine, send it with request (both query and response) - according to possible STM sizes, could be
  used only for very small models, not recommended

It should allow dynamic STM handling in cloud environments and is limited only by environment.

#### Multiple interaction streams
Reactive models should be able to handle multiple conversations at once with shared STM state, but they will rather need
something to distinguish it in each interaction, in example adding "Chat N: " prefix to each query, should work. It could
enable asking about a things from the other dialog, etc. However, it's not the model's default mode and may need some
additional tuning.

### Message template
Conversational LLM models are using complex chat templates based on list of JSON objects for each message (that could have
a different type like "user", "assistant" or "system"), to include full chat history in every prompt. Reactive language
models are processing only a single interactions, don't have the "system" messages and could be imagined as the Question
Answering (QA) models with implicit access to previous questions and answers, so they require a lot simpler template. Basically,
it's just: `[BOS] [Q] User's query... [A] Model's answer... [EOS]` - it also make the supervised fine-tuning datasets much
simpler and readable.

> For reasoning template is extended to: `[BOS] [Q] User's query... [R] Model's reasoning... [A] Model's answer... [EOS]`

## Training
As the **Reactive Transformer** is a connected ensemble of cooperating models, its training process is more complicated,
then standard LLM training. It requires more careful pre-training, fine-tuning and additional **Memory Reinforcement Learning (MRL)**
stage. Our progressive learning process is designed as an extension to existing **Transformer** learning algorithms.
It's divided into six separate stages:
1. Joint LM Pre-Training for encoder and decoder, on autoregressive language modeling and masked language modeling at once
2. Joint Components Interaction Supervised Fine-Tuning (SFT)
3. Memory Attention Self-Supervised Pre-Training
4. Supervised Memory-Aware Training
5. Memory Reinforcement Learning (MRL) for Short-Term Memory
6. Reinforcement Learning from Human Feedback for Reactive Models (RxRLHF)

In addition to standard language modelling training, first four stages are designed to connect vector spaces between components,
and provide great initial results in **Memory Reinforcement Learning**.

> ### Transfer learning from classic transformers
> Thanks to the residual connections, and the fact that with initial, very small random weights, the attention layer output
> is almost completely skipped by residuals, it should be possible to extend the pre-trained classic transformer (with
> corresponding QA fine-tuning) with memory cross-attention and use it in MRL. However, most current LLMs are decoder-only,
> and it's almost impossible to find the encoder with the same number of layers and same model dimensionality, so you will
> still have to pre-train the encoder using at least partially the same dataset. Additionally, it will require careful
> training to connect vector spaces and finally, may be harder than training new model from scratch - it's the topic for
> another research

## Inference Speed and Cost
As **Reactive Transformer** is processing single interactions instead of conversation history, each message processing have
similar speed and cost - it's increase with number of messages is linear. That's one of the biggest practical advantages
over LLMs or even Diffusion Language Models, that are also based on full conversation processing. For **Reactive Language
Models**, only the first interaction could be a little more expensive than for the same size LLM, because of the encoder,
memory attention and memory cross-attention overhead. Then, for next messages, it's about `N` times faster and cheaper, where
`N` is the number of messages.

The computational costs (and inference costs as well) growth, where `N` is the number of messages/interactions in conversation
and `T` is the number of tokens for single interaction:
- conversation cost:
  - for LLMs it's quadratic growth - `O(N * T + (N - 1) * T + (N - 2) * T + ... + T)` simplified to `O(N²T)`
  - for RxLMs it's linear growth - `O(T + T + T + ... + T)`, which is just `O(NT)`
- each single interaction cost
  - for LLMs it's the interaction and all previous ones cost - `O(NT)`
  - for RxLMs it's just the number of tokens in current interaction - `O(T)`
- it's proving that RxLMs are `N` times faster and cheaper than LLMs

> Encoder and Memory Attention overhead is almost unnoticeable, because it's used once for each interaction, when decoder
> is used once per generated token. So, decoder autoregressive processing is taking `N_tokens` times more time, than encoder
> and memory attention. On the other hand, Memory Cross-Attention overhead is noticeable, as it's used in decoder, but it's
> still nothing compared to full conversation processing, and it's optimized with full KV pre-caching (more details below)
> and fast **Sparse Query Attention** layers.

### Attention Key/Value Cache for autoregressive generation
LLM autoregressive generation is optimized by the attention key/value heads cache (KV cache) to reduce the number of duplicated
operations for each generated tokens - previous tokens results of key/value linear projections with RoPE encoding are stored
and re-used in next steps. In case of full conversation processing, it results in meaningful savings, but it's less useful
for real-time single interaction processing - just because it's almost always a lot shorter. However, duplicated operations
still have no sense, so KV cache is implemented in our models, just like in LLMs.

On the other hand, during the autoregressive answer generation, **Memory Cross-Attention** is always accessing the same
static STM state (that is updated between generating answers) as keys/values, so it all could be initially pre-cached
at once - then we completely skip all key/value linear projections in cross-attention. It results in about 20-25% faster
generation.

## Summary
**Reactive Transformer** is the first groundbreaking **Event-Driven AI** architecture that redefines the conversational NLP,
by introducing the **Real-Time Processing** to language models. It's a crucial milestone on the road to real awareness models.
Based on the state-of-the-art **Transformer** architecture it doesn't reinvent the wheel in language modelling, but extends the
LLM concepts with dedicated memory layers to keep all the context. It results in natural, human-like conversation mode, opposite
to simulated nature of LLM full conversation processing.

Models based on **Reactive Transformer** should be about `N` times cheaper and faster than the same size LLMs, where `N`
is the number of messages. That's only a guess, but that models could also achieve better task performance, just because
the isolation of processed interaction from the previous ones - that should be checked in further experiments.

After this introduction, in next articles we will describe all the training process of **Reactive Transformer**, starting
from the supervised stages.