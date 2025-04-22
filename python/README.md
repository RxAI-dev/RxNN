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
