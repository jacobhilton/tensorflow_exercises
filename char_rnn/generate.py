import char_rnn
import sys

corpus_filename = sys.argv[1]
char_rnn.generate(sys.argv[2], sys.argv[3], corpus_filename + ".alphabet.txt", corpus_filename + ".ckpt")
