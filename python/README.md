<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/logo_rxai.webp" width="300" />
<img src="https://raw.githubusercontent.com/RxAI-dev/RxNN/refs/heads/main/assets/logo_rxnn.webp" width="300" />

# Reactive AI - RxNN
## Reactive Neural Networks Platform

RxNN is AI/DeepLearning development platform made for Reactive Neural Networks and Event-driven AI, introduced by Reactive AI.

## Reactive Neural Networks and Event-driven AI
Reactive neural networks (RxNN) are a new family of memory-augmented neural networks that combine classical deep learning
algorithms with reactive communication patterns. In Event-driven AI, input data (sequence) is treated as event, and memory
state has to be kept between events/interactions. Technically, it's a specific kind of RNN that's storing data between
processed sequences, instead of between sequence elements like in regular RNN. Then, their recurrence is on a higher level.
In the case of reactive communication patterns, RxRNNs are stateful reactive data sources that you have to connect before
you can send and receive messages.
While RxNNs are using some RNN concepts, they are rather made to extend Transformer language/multi-modal models. In our
opinion, the biggest downside of current LLMs is their stateless nature - conversational models have to process full chat
history on every interaction! That's not real-time processing, and it's not how human's awareness is working. In RxNN based
transformers, model is processing single messages, while all the previous interactions history should be saved and read
from memory. That features are required for **Weak** Reactive Neural Networks specification, and it will be the first major
step in transition from language models to awareness models - in Reactive AI ecosystem, it will be introduced in Reactive
Transformer architecture.

Additionally, to achieve awareness, **Strong** Reactive Neural Networks are working in reactive infinite reasoning loop,
that's generating Infinite Chain-of-Thoughts and is communicating in push-based mode (model decides if and when return output).

Reactive communication patterns in RxNN models are adapted to handle asynchronous nature of model - after it finish generating
sequence, it has to process it and save it in memory, but it could be done in background.

## Release plan
We are working on three new reactive architectures, that progressively advance from language models to awareness models:
- Reactive Transformer: Reactive Language Model (RLM) with Short-Term Memory
- Preactor: extending Reactive Transformer with additional Long-Term Memory, providing theoretically infinite context (only
  single message length is limited) and the ability to learn from interactions (Live Learning)
- Reactor: AGI awareness model & Strong Reactive Neural Network, that's working in infinite reasoning loop and doesn't require explicit human commands

Each new architecture is based on the previous one and adding new features/abilities. They will be progressively
released with next versions of **RxNN** framework:
- 0.1.x: Reactive Transformer base models, Base Model Learning (pre-training/fine-tuning) & Transformers extensions (MoE Attention, Short-Term Memory, etc.)
- 0.2.x: Memory Reinforcement Learning (MRL) for Short-Term Memory & Reactive Transformer, Attention-based Memory System details
- 0.3.x: Reinforcement Learning from Human Feedback for Reactive models (RxRLHF), basic Tensor Reactive
  Extensions (TRX/Rust) for full Reactive Transformer, RxT-Alpha release (+following models - RxT-Beta, etc.)
- 0.4.x: Preactor base models, Tensor Database (TDB/Rust) for Long-Term Memory, mxRAG/revRAG subsystems
- 0.5.x: MRL for Long-Term Memory & Preactor, Live Learning for Preactor, PRx-Alpha release (+following models - PRx-Beta, etc.)
- 0.6.x: Reactor base models, TRX full implementation, Receptors & Effectors Reactive RNNs
- 0.7.x: Behavioral Reinforcement Learning (BRL) for Reactor's Infinite Chain-of-Thoughts, Continuous Live Learning for Reactor
- 0.8.x: Rx-Alpha release
- 0.9.x: Rx-Beta release
- 1.0.0: Reactor AGI official release (Expert, Assistant & Utility class models)
- 1.x.x: Multimodal reactive models (could be released earlier, depending on progress)
- 2.0.0: Real-Time Vision Reactor - Worker class models
- x.x.x: ...and more!