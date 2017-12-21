import tensorflow as tf
import random

training_sequence_length = 180
batch_size = 128
lstm_size = 256
lstm_layers = 2
learning_rate = 0.004
epochs = 100

def network(alphabet_length, sequence_length):
    sequences_placeholder = tf.placeholder(tf.int64, [batch_size, sequence_length])
    inputs = tf.slice(tf.one_hot(sequences_placeholder, alphabet_length), [0, 0, 0], [-1, sequence_length - 1, -1])
    targets = tf.slice(sequences_placeholder, [0, 1], [-1, -1])
    def basic_lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm_cell() for _ in range(lstm_layers)])
    multi_lstm_outputs, _ = tf.contrib.rnn.static_rnn(multi_lstm_cell, tf.unstack(inputs, num=sequence_length - 1, axis=1), dtype=tf.float32)
    logits = tf.layers.dense(tf.stack(multi_lstm_outputs, axis=1), alphabet_length)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, sequence_length - 1], dtype=tf.float32))
    return sequences_placeholder, logits, loss

def train(corpus_filename, alphabet_save_filename, ckpt_save_filename):
    with open(corpus_filename, "r") as file:
        corpus = file.read()
    alphabet = list(set(corpus))
    alphabet.sort()
    characters = [alphabet.find(letter) for letter in corpus]
    with open(alphabet_save_filename, "w") as file:
        file.write("".join(alphabet))
    number_of_sequences = len(characters) // training_sequence_length
    sequences = [characters[i:i + training_sequence_length] for i in range(0, number_of_sequences * training_sequence_length, training_sequence_length)]
    number_of_batches = len(sequences) // batch_size
    batches = [sequences[i:i + batch_size] for i in range(0, number_of_batches * batch_size, batch_size)]
    sequences_placeholder, _, loss = network(len(alphabet))
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_loss = 0
        for batch in batches:
            sess.run(training_step, {sequences_placeholder: batch})
            total_loss += sess.run(loss, {sequences_placeholder: batch})
            print(total_loss)
        print("Loss after {0} epoch(s): {1}".format(epoch + 1, total_loss / number_of_batches))
    tf.train.Saver().save(sess, ckpt_save_filename)

def generate(seed, alphabet_load_filename, ckpt_load_filename):
    with open(alphabet_load_filename, "r") as file:
        alphabet = file.read()
    sequence = [alphabet.find(letter) for letter in sequence]
    if -1 in sequence:
        raise RuntimeError("Seed contains a letter not found in the corpus")
    sequences_placeholder, logits, _ = network(len(alphabet), len(seed) + 1)
    sess = tf.Session()
    tf.train.Saver().restore(sess, ckpt_load_filename)
    sess.run(tf.argmax(logits, axis=1), {sequences_placeholder: seed + " "}
    #temperature
    #is last dense layer correct?
