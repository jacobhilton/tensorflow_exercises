import tensorflow as tf
import random

def read_corpus(filename):
    with open(filename) as file:
        corpus = file.read()
    dictionary = list(set(corpus))
    dictionary.sort()
    characters = [dictionary.index(letter) for letter in corpus]
    return dictionary, characters

def training_set(characters, sequence_length=64):
    [characters[i:i + sequence_length] for i in range(len(characters) - sequence_length)]
    [characters[i + sequence_length] for i in range(len(characters) - sequence_length)]
    

inputs = tf.placeholder(tf.float32, [time_steps, batch_size, len(dictionary)])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)



dictionary, characters = read_corpus("input.txt")
training_set = 
print(characters[2])
print(dictionary[characters[2]])
print(dictionary)

#epochs = 100000
#size_c = 256
#seq_len = 180
#batch_size = 128

