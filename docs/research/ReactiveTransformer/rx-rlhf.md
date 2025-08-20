### Reinforcement Learning from Human Feedback for Reactive Models (RxRLHF)
In the last training stage, after the model is able to correctly use its Short-Term Memory, it should be finally trained on
human preferences' alignment. During **RxRLHF** model is improving the dialog quality, just like in regular **RLHF**. The only
difference is in real-time processing, so it needs specialized datasets and different environment handling, but it should be
simply - just not use full history as inputs, but only single interactions. According to that difference, we have to extend
the HuggingFace `trl` implementation in our **RxNN** framework.

#### Combining MRL and RxRLHF
As the MRL is very similar to RLHF and the most noticeable difference is the rewarding scheme, it could be possible to combine
these stages, by adding the conversation quality reward to MRL. In this mode, each interaction will get 2 rewards - for memory
retention and for response/answer quality. It should save a time and resources, but the training may be less stable. We have
to check separate steps and if we achieve success, then we could check the combined approach