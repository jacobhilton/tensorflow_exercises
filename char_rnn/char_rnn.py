import tensorflow as tf
import random

sequence_length = 180
batch_size = 128
lstm_size = 256
lstm_layers = 2
epochs = 100 #100000
learning_rate = 0.004

def read_corpus(filename):
    with open(filename) as file:
        corpus = file.read()
    dictionary = list(set(corpus))
    dictionary.sort()
    characters = [dictionary.index(letter) for letter in corpus]
    return dictionary, characters

def network(dictionary_length):
    sequences_placeholder = tf.placeholder(tf.int64, [batch_size, sequence_length])
    inputs = tf.slice(tf.one_hot(sequences_placeholder, dictionary_length), [0, 0, 0], [-1, sequence_length - 1, -1])
    targets = tf.slice(sequences_placeholder, [0, 1], [-1, -1])
    def basic_lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm_cell() for _ in range(lstm_layers)])
    multi_lstm_outputs, _ = tf.contrib.rnn.static_rnn(multi_lstm_cell, tf.unstack(inputs, num=sequence_length - 1, axis=1), dtype=tf.float32)
    logits = tf.layers.dense(tf.stack(multi_lstm_outputs, axis=1), dictionary_length)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, sequence_length - 1], dtype=tf.float32))
    return sequences_placeholder, logits, loss

def train(filename):
    dictionary, characters = read_corpus(filename)
    number_of_sequences = len(characters) // sequence_length
    sequences = [characters[i:i + sequence_length] for i in range(0, number_of_sequences * sequence_length, sequence_length)]
    number_of_batches = len(sequences) // batch_size
    batches = [sequences[i:i + batch_size] for i in range(0, number_of_batches * batch_size, batch_size)]
    sequences_placeholder, _, loss = network(len(dictionary))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_loss = 0
        for batch in batches:
            sess.run(training_step, {sequences_placeholder: batch})
            total_loss += sess.run(loss, {sequences_placeholder: batch})
            print(total_loss)
        print("Loss after {0} epoch(s): {1}".format(epoch + 1, total_loss / number_of_batches))

train("input.txt")
