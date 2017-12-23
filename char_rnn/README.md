Python scripts to train a recurrent neural network on a corpus of text and generate new text based on the result. It uses using a character-level model and a stacked LSTM, with hyperparameters borrowed from [this](https://github.com/LaurentMazare/tensorflow-ocaml/tree/master/examples/char_rnn) implementation.

It can be trained on `corpus.txt` using:

```
python3 train.py corpus.txt
```

This will create `corpus.txt.alphabet.txt` and some `corpus.txt.ckpt*` files.

A 10KB (say) piece of text beginning with `$SEED` can then be generated using:

```
python3 generate.py corpus.txt "$SEED" 10000
```

The default "temperature" parameter, which specifies the randomness of the generated text, is 0.5.