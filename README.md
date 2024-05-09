# xLSTM

A pure pytorch implementation of the [XLSTM paper](https://arxiv.org/abs/2405.04517).

## Notes
- I just made this after a pass or two of reading the paper so there may be inconsistencies with the paper or mistakes, I will work on fixing these as I notice them and as they are pointed out
- I have not yet looked into any of the efficiencies or parallelization techniques discussed in the paper.

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