import tensorflow as tf
import random
import os

sequence_length = 180
batch_size = 128
lstm_size = 256
lstm_layers = 2
learning_rate = 0.004
epochs = 100

def network(alphabet_length):
    sequences_placeholder = tf.placeholder(tf.int64, [None, None])
    inputs = tf.one_hot(sequences_placeholder[:, :-1], alphabet_length)
    targets = sequences_placeholder[:, 1:]
    def basic_lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm_cell() for _ in range(lstm_layers)])
    multi_lstm_outputs, _ = tf.contrib.rnn.static_rnn(multi_lstm_cell, tf.unstack(inputs, num=sequence_length-1, axis=1), dtype=tf.float32)
    logits = tf.layers.dense(tf.stack(multi_lstm_outputs, axis=1), alphabet_length)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones(tf.shape(targets), dtype=tf.float32))
    return sequences_placeholder, logits, loss

def train(corpus_filename, alphabet_save_filename, ckpt_save_filename):
    with open(corpus_filename, "r") as file:
        corpus = file.read()
    alphabet = list(set(corpus))
    alphabet.sort()
    characters = [alphabet.index(letter) for letter in corpus]
    with open(alphabet_save_filename, "w") as file:
        file.write("".join(alphabet))
    number_of_sequences = len(characters) // sequence_length
    sequences = [characters[i:i + sequence_length] for i in range(0, number_of_sequences * sequence_length, sequence_length)]
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
        print("Loss after {0} epoch(s): {1}".format(epoch + 1, total_loss / number_of_batches))
    tf.train.Saver().save(sess, os.getcwd() + "/" + ckpt_save_filename)

def generate(seed, output_length, alphabet_load_filename, ckpt_load_filename):
    with open(alphabet_load_filename, "r") as file:
        alphabet = file.read()
    sequence = [alphabet.find(letter) for letter in seed]
    if -1 in sequence:
        raise RuntimeError("Seed contains a letter not found in the corpus")
    sequences_placeholder, logits, _ = network(len(alphabet))
    sess = tf.Session()
    tf.train.Saver().restore(sess, os.getcwd() + "/" + ckpt_load_filename)
    while len(sequence) < output_length:
        truncated_sequence = sequence[-sequence_length-1:]
        input_sequence = truncated_sequence + [0 for _ in range(sequence_length - len(truncated_sequence))]
        sequence.append(sess.run(tf.squeeze(tf.argmax(logits[:, len(truncated_sequence) - 1, :], axis=1)), {sequences_placeholder: [input_sequence]}))
        #temperature
    return "".join([alphabet[character] for character in sequence])

#train("test.txt", "alphabet.txt", "test.ckpt")
#print(generate("KING", 500, "alphabet.txt", "test.ckpt"))
