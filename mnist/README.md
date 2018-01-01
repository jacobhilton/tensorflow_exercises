Python script to train a feed-forward neural network to recognise handwritten digits. The script finds the optimal "[temperature](https://blog.janestreet.com/does-batch-size-matter/)" (i.e. learning rate / mini-batch size) using cross-validation, trains the neural network over 50 epochs and then tests the accuracy of the resulting network. The training and test sets are both from [MNIST](http://yann.lecun.com/exdb/mnist/).

There are 4 models to choose from:

- Softmax regression: 84.0% test-set accuracy
- Dense neural network with 1 hidden layer of 100 units: 95.7% test-set accuracy
- Dense neural network with 1 hidden layer of 800 units: 97.5% test-set accuracy
- Convolutional neural network with the following hidden layers: 99.0% test-set accuracy
  - 5x5 convolutional layer with 40 channels, a stride of 1, zero padding and relu activation
  - 2x2 max pooling layer
  - 3x3 convolutional layer with 80 channels, a stride of 1, zero padding and relu activation
  - 2x2 max pooling layer
  - dense layer with 1000 units and sigmoid activation