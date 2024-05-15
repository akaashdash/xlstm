# xLSTM

A pure pytorch implementation of the [XLSTM paper](https://arxiv.org/abs/2405.04517).

## TODO
- Ensure correct with paper
- Create some usage examples
- Implement paralleization
- CUDA?
- Allow for different initializations according to: https://pytorch.org/docs/stable/nn.init.html
- Allow for flattening of x for greater shape conformity
- Allow for batching
- Add tests
    - https://github.com/catid/audio_prediction/tree/master
    - other classic RNN/LSTM tasks

## References
- https://arxiv.org/abs/2405.04517
- https://discuss.pytorch.org/t/causal-convolution/3456/3
- https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html