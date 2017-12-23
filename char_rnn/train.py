import char_rnn
import sys

corpus_filename = sys.argv[1]
char_rnn.train(corpus_filename, corpus_filename + ".alphabet.txt", corpus_filename + ".ckpt")
