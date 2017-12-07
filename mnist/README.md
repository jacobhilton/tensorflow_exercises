Script to train and test a neural network on the MNIST dataset. Finds the optimal [temperature](https://blog.janestreet.com/does-batch-size-matter/) (i.e. learning rate / mini-batch size) using cross-validation, trains the neural network over 50 epochs and then tests the result.

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