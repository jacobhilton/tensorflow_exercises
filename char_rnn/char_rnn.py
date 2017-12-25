import tensorflow as tf
import random
import os

training_sequence_length = 180
batch_size = 128
lstm_size = 256
lstm_layers = 2
learning_rate = 0.004
epochs = 100

maximum_seed_length = 180
generating_sequence_length = 900
temperature = 0.5

def basic_lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm_cell() for _ in range(lstm_layers)])

def network(inputs, multi_lstm_state, alphabet_length, sequence_length):
    multi_lstm_inputs = tf.unstack(tf.one_hot(inputs, alphabet_length), num=sequence_length, axis=1)
    multi_lstm_outputs, multi_lstm_state = tf.contrib.rnn.static_rnn(multi_lstm_cell, multi_lstm_inputs, initial_state=multi_lstm_state, dtype=tf.float32)
    return tf.layers.dense(tf.stack(multi_lstm_outputs, axis=1), alphabet_length, reuse=tf.AUTO_REUSE), multi_lstm_state

def train(corpus_filename, alphabet_save_filename, ckpt_save_filename):
    with open(corpus_filename, "r") as file:
        corpus = file.read()
    alphabet = list(set(corpus))
    alphabet.sort()
    characters = [alphabet.index(letter) for letter in corpus]
    with open(alphabet_save_filename, "w") as file:
        file.write("".join(alphabet))
    number_of_sequences = len(characters) // training_sequence_length
    sequences = [characters[i:i + training_sequence_length] for i in range(0, number_of_sequences * training_sequence_length, training_sequence_length)]
    number_of_batches = len(sequences) // batch_size
    batches = [sequences[i:i + batch_size] for i in range(0, number_of_batches * batch_size, batch_size)]
    sequences_placeholder = tf.placeholder(tf.int64, [None, None])
    inputs = sequences_placeholder[:, :-1]
    logits, _ = network(inputs, None, len(alphabet), training_sequence_length - 1)
    targets = sequences_placeholder[:, 1:]
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones(tf.shape(targets), dtype=tf.float32))
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_loss = 0
        for batch in batches:
            sess.run(training_step, {sequences_placeholder: batch})
            total_loss += sess.run(loss, {sequences_placeholder: batch})
        print("Loss after {0} epoch(s): {1}".format(epoch + 1, total_loss / number_of_batches), flush=True)
    tf.train.Saver().save(sess, os.getcwd() + "/" + ckpt_save_filename)

def decode(sequence, alphabet):
    return "".join([alphabet[character] for character in sequence])

def generate(seed, total_length, alphabet_load_filename, ckpt_load_filename):
    if len(seed) == 0:
        raise RuntimeError("Seed is empty")
    with open(alphabet_load_filename, "r") as file:
        alphabet = file.read()
    seed_characters = [alphabet.find(letter) for letter in seed]
    if -1 in seed_characters[-maximum_seed_length:]:
        raise RuntimeError("Seed contains a letter not found in the corpus")
    sess = tf.Session()
    print(seed, end="", flush=True)
    restore = True
    total_sequence = seed_characters
    while len(total_sequence) < total_length:
        truncated_seed = total_sequence[-maximum_seed_length:]
        seed_placeholder = tf.placeholder(tf.int64, [None])
        if len(truncated_seed) == 1:
            state = None
        else:
            _, state = network(tf.expand_dims(seed_placeholder[:-1], 0), None, len(alphabet), len(truncated_seed) - 1)
        input = tf.expand_dims(seed_placeholder[-1:], 0)
        outputs = []
        for _ in range(min(total_length - len(total_sequence), generating_sequence_length)):
            logits, state = network(input, state, len(alphabet), 1)
            if temperature == 0:
                output = tf.argmax(logits, axis=2)
            else:
                probabilities = tf.nn.softmax(tf.scalar_mul(1 / temperature, logits))
                cumulative_probabilities = tf.cumsum(probabilities, axis=2, exclusive=True)
                indices = tf.cumsum(tf.ones(tf.shape(probabilities), dtype=tf.float32), axis=2)
                output = tf.argmax(tf.multiply(indices, tf.cast(tf.less(cumulative_probabilities, tf.random_uniform([1], minval=0, maxval=1)), dtype=tf.float32)), axis=2)
            outputs.append(tf.squeeze(output))
            input = output
        if restore:
            restore = False
            tf.train.Saver().restore(sess, os.getcwd() + "/" + ckpt_load_filename)
        new_characters = list(sess.run(tf.stack(outputs, axis=0), {seed_placeholder: truncated_seed}))
        new_letters = decode(new_characters, alphabet)
        print(new_letters, end="", flush=True)
        total_sequence += new_characters
    print("", flush=True)
    return decode(total_sequence, alphabet)
