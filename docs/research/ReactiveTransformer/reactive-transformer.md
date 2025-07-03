# Reactive Transformer: Real-time processing for language models
by Adam Filipek/Reactive AI

## Abstract
Language models based on Transformer architecture are industry standard in Natural Language Processing. They are great
in language modelling and generative tasks. However, Transformers are completely stateless, and they are only emulating Short-Term
Memory with their long contexts — every processed sequence is completely separate. Conversational models based on this concept,
are processing all the conversation history on each message/interaction. Despite it being extremely inefficient, it's not how humans
thinking and communicating. Awareness is a stateful and continuous process that requires _Real-time Processing_, which is completely
opposite to how transformers are working.

Previous approaches to memory in NLP, based on RNNs or even Neural Turing Machines, were concentrated on keeping state between
tokens in sequence, because it was the main problem for research in pre-transformer times. After transformer architecture release,
it no longer requires memory, thanks to attention layers. Then, it looks like the research community forgot about memory and
agreed that it's no longer necessary.

Someone even believed that transformers would achieve awareness only by scaling it furthermore to even bigger sizes. No, they
wouldn't, it's **impossible**. The key "feature" of awareness is that I rather know what I was doing or thinking 10 minutes
ago, without a need to read my whole-day history.

Awareness requires keeping state between sequences/interactions instead and processing only single messages in real-time,
with access to previous interactions by memory. It's only the first step to awareness, but this step is crucial.

Those features are partially covered by agentic frameworks, like LangChain, that are integrating LLM models with external
tools, including memory or databases, through _prompt engineering_. But the models, even if agents are called "reactive",
still processing whole histories, instead working in real time. Our new **Reactive Neural Networks** and **Event-driven AI**
paradigms are moving those agentic features into the models architectures.

In this research, we are introducing **Reactive Transformer** architecture and **Attention-based Memory System for Short-Term
Memory**, that's processing only single messages in real time and moving conversation history into separate memory layers,
accessed and updated by specialized attention layers.

## Reactive Neural Networks and Event-driven AI
_Reactive neural networks (RxNN)_ are event-driven, memory-aware neural networks with stateful real-time processing and infinite
inter-sequence recursion (while RNN recurrence is mainly intra-sequence). While our new reactive architectures are based
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

> **Strong Reactive Neural Networks (called also Proactive Neural Networks)**, like Reactor, have additional internal
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

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer.png" />

Models based on RxT architecture are called **Reactive Language Models (RxLM)**

### Shared Embedding
Both encoder and decoder have to share the same _embedding_ layer - they both have access to STM, that's connected to the
same _embedding_ space. It will reduce the parameter count in the final architecture and slightly improve a decoder quality (using
bidirectional MLM embeddings from encoder), but it's making the _tied embedding_ concept hard to implement (we're not using
it in our models). Embedding layer is pre-trained with encoder, then it's frozen for decoder pre-training. It's getting
unfrozen after initial steps of Memory Reinforcement Learning and is trained with all the other components.

### Memory Cross-Attention
Most important and noticeable difference between Reactive Transformer layers and classic Transformer layers are the additional
_**Memory Cross-Attention**_ layers between self-attention and feed forward net. They are included in both decoder and encoder,
as encoder should be also memory-aware, when it's encoding new data to update the memory.

In original Transformer, cross-attention in decoder is used to access the data encoded by encoder layers. In **RxT** it's
similar, memory cross-attention is still accessing encoder hidden states, but combined (accumulated) with all states from previous
interactions (by memory-attention). It has one crucial implication - _positional encoding_ - memory shouldn't have positional
encoding and should be treated rather as a _set_ than a _sequence_, because it's all updated incrementally - in each step
the message will include some different information on each position, so it shouldn't be reflected in the STM memory.

Then, as the _**Rotary Positional Embedding (RoPE)**_ is currently industry standard, in RxT's memory cross-attention we are
applying RoPE rotation only to _queries_ (input sequence) and not to _keys_ (STM layer state).

> Legacy positional embedding solutions (absolute/relative) aren't considered for Reactive Transformer. They will add,
> explicitly (relative) or implicitly (absolute), positional encoding to Short-Term Memory layers, then it shouldn't be
> a good choice.

### Memory Attention & Attention-based Memory System
Memory cross-attention provides the access to Short-Term Memory layers and integration with Transformer architecture. But
how to update memory? The _**Attention-based Memory System (ABMS)**_ assumes that attention mechanisms are also responsible
for updating the memory state, in a process that is opposite to the memory cross-attention. In **Memory Attention** STM
states becoming _queries_, while the encoded sequences (encoder hidden states) are treated as _keys/values_. Before passing
STM to attention layers, it's normalized by _RMS_ based **Memory Norm** and the attention layer's result is added to current
STM layer's state in residual connection - this design should ensure incremental STM state building.

As memory shouldn't be positionally encoded, in **Memory Attention** RoPE only _keys_ are rotated.

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
this approach, decoder could be N times bigger than encoder, and overally, Reactive Transformer will be only slightly bigger
than classic, decoder-only MoE Transformer.

<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/research/reactive-transformer-moe.png" />

### Sparse Query Attention for efficient attention in Reactive Transformer
In our additional side research, we discovered, that reducing used query heads count (like using 8 heads from 16), instead
furthermore reduction of key/value heads from GQA level, up to MQA, leads to a lot better computational efficiency, and
almost the same model quality (loss/accuracy/perplexity, etc.) - we called this solution _**Sparse Query Attention**_. All
the SQA variants are about ~5-15% faster than GQA/MQA, while they performance is always somewhere between GQA and MQA.
Symmetric variants of SQA, that's using exactly 50% of query and key/value heads (use full MHA optimizations) are still
a lot faster than GQA with much smaller number of key/value heads, while their performance is almost exactly the same.
Extreme SQA variants, when there are used less than 50% heads for both query and key/value, are still on MQA's performance
level with the best possible computational efficiency. However, after some level - about 25% of each heads, further reduction
doesn't improve training times and performance is noticeable worse. SQA has also the smallest number of parameters, because
of an additional output projection input dimensionality reduction. The time differences between SQA variants and GQA/MQA are
getting greater with longer sequences, while the performance is still on the same level, so SQA could be even better for very
long contexts, however, we didn't run detailed tests for long context models, because of the limited budget, so it's still
something to explore.

In summary, SQA seems to be the most cost-effective variant of Grouped Query Attention, reducing training time/cost by ~5-15%,
while staying on GQA/MQA performance level, and will be implemented in our **Reactive Transformer** models.

#### Spatially sparse attention for memory
Spatially sparse attention mechanisms, based on sliding local windows, especially _**Flex Attention**_, are becoming more
popular, because of their ability to handle very long sequences with `O(N * log N)` complexity. They are excellent for very
long contexts extending a million tokens, but like I said before - it's rather the opposite direction of research. I don't
know if "over 1M tokens" sequence length will be ever needed for single message processing (it's against the real-time
processing - reading a book in real-time mode will be rather partial, and it's natural - people rather don't read whole
books at once), because we want to achieve those conversation lengths differently with memory - "million-token like" conversation
should be achieved with about 32k message length and 32-64k memory layer size. Then, **Reactive Transformer** may just not need
spatial sparsity for self-attention, except some rare cases with very long single interactions.

While spatial sparsity could be optionally used for self-attention, it rather isn't a good idea for memory. Spatial sparsity
is based on positional relations, but memory doesn't have positional encoding and could have completely different spatial
relations structure. In example, one local window could include information from different interactions (time steps), that
are strictly connected to items in other local window. Then, the structural sparsity provided by SQA seems to be a lot
better option - attention has access to all the elements, but using only partial information about relations, so it could
learn to select a part connected with particular interaction.

Additionally, STM has always full, constant size (opposite to input sequence, that's variable from 0 to max message length),
so memory cross-attention and memory attention computational cost will be almost always higher, than self-attention, so it
has to be as fast as possible, so SQA seems to be the best option.

> Before SQA discovery, we considered using GQA for self-attention and MQA for memory cross-attention and memory attention. They
> should be of course still viable, but SQA just fits better to the overall architecture.

We consider using Symmetric Sparse Query Attention (sSQA) for model's self-attention and regular SQA (GQA with reduced active
query heads) for memory cross-attention and memory attention in **Reactive Transformer**.

### Sampler
As **Reactive Transformer** algorithm includes accumulating generated tokens, it needs the **Sampler** as a core architecture
component - its arguments - temperature, top p or top k - are becoming model's hyperparams, used not only in inference, but
also in Reinforcement Learning stages. 

### Quasi Live Learning and Pseudo-Infinite Context
As the first event-driven reactive neural network, **Reactive Transformer** has no overall context limitation (only single
message/interaction length is limited) and the ability to learn from interactions, but it's limited by the memory capacity,
which according to the name is "short-term" and it's size is not extendable. Although the RxT memory is many thousands of
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
3. Generated answer is **streamed/returned** to the environment (in separate thread), but is also accumulated into the _answer_ sequence, that's being embedded
4. Embedded _query_ and _answer_ sequences are concatenated and processed by encoder, with the same STM access through memory cross-attention as decoder
5. _Encoded data_ - hidden states from each encoder's layer, are combined with previous STM state by memory attention layers.
6. When memory is updated, model is ready for next interaction - so we are back in the step 0.

This approach require using asynchronous reactive communication patterns - when the decoder (with Sampler) finishes generating
the answer, it still has to update memory for the next interaction (in background, from user's perspective). So the model has to
send some additional communicates to the UI, to display/hide the information about memory update and/or enqueue new message/interaction,
when it's send before it finished memory update.

We are using solutions based on _Reactive Programming_ like signals or observables to handle asynchronous communication in
_Reactive Neural Networks_. It's handled on the low level (without moving data between devices and only minimal overhead) in a dedicated
**Rust** library - **Tensor Reactive Extensions (TRx)**, inspired on Reactive Extensions (like RxPy), but dedicated to work with tensors.

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

## Training
As the **Reactive Transformer** is a connected ensemble of cooperating models, its training process is more complicated,
then standard LLM training. It requires more careful pre-training and additional **Memory Reinforcement Learning (MRL)**
stage, but also simplifies some stages, like fine-tuning, because of single interaction processing. Our progressive learning
process is designed as an extension to existing **Transformer** learning algorithms. It's divided into four separate steps:
1. Base Models Pre-Training on autoregressive language modeling and masked language modeling
2. Components Supervised Fine-Tuning (SFT) on Instruct-QA and/or Chain-of-Thoughts-QA tasks
3. Memory Reinforcement Learning (MRL) for Short-Term Memory
4. Reinforcement Learning from Human Feedback for Reactive Models (RxRLHF)

In first two stages, memory cross-attention layers are frozen (and skipped by initial residual connections), to not disrupt
language modeling training. Memory attention layers are not even used - they are introduced from MRL stage.

> ### Transfer learning from classic transformers
> Thanks to the residual connections, and the fact that with initial, very small random weights, the attention layer output
> is almost completely skipped by residuals, it should be possible to extend the pre-trained classic transformer (with
> corresponding QA fine-tuning) with memory cross-attention and use it in MRL. However, most current LLMs are decoder-only,
> and it's almost impossible to find the encoder with the same number of layers and same model dimensionality, so you will
> still have to pre-train the encoder using at least partially the same dataset. It's still a cost reduction, but will require
> much more careful encoder pre-training and could be less stable

### Base Models Pre-Training
That stage is based on standard language modeling, and there isn't anything new, except shared embeddings and frozen, or
even skipped memory cross-attention. As encoder's bidirectional masked language modeling could result in better embedding
quality, the training is started from encoder:
1. Pre-train encoder on masked language modeling, with trainable embedding layer
2. Load pre-trained encoder's embedding layer and randomly initialized STM into decoder
3. Pre-train decoder on autoregressive language modeling, with frozen embedding layer

### Supervised Fine-Tuning (SFT)
In this stage, we have to fit the models (both decoder and encoder) to process the data in single interaction format - question
and answer. We have to provide the examples of interactions from the start of conversation and from the middle of conversation,
but it doesn't require any connections between consecutive interactions. Each example interaction could be independent, but there
should be examples that looks as taken from different stage of conversation, like follow-up questions or topic changes. The goal
is to tune the model to basic conversational interaction format and keep connections between questions and answers. The inter-sequence
memory dependencies are added in the next stage.

#### SFT Datasets
Dataset for Supervised Fine-Tuning for Reactive Transformer includes simple Question-Answer (Query-Response) interactions, from
different stages of dialog/conversation. They don't need to be connected, because they will be just shuffled in different epochs.

Single record - interaction - could be a single text sequence in format `[Q] User's question [A] Model's answer` or it could
be divided into two separate fields. First option is simpler, but the second is easier to customize, so it should be recommended.

More for [Reinforcement Learning Stages](./mrl.md)

[Memory Attention variants](./memory-attention.md)
