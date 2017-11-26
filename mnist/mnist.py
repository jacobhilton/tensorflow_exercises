import tensorflow as tf
import random

def image_to_vector(image):
    return [pixel for row in image for pixel in row]

def label_to_vector(label):
    return [int(label == label_number) for label_number in range(10)]

def read_image_vectors(filename, number=None, start=0):
    with open(filename, "rb") as file:
        file.seek(16 + start * 28 * 28)
        if number is None:
            bytes = file.read()
        else:
            bytes = file.read(number * 28 * 28)
    pixels = list(bytes)
    images = [[pixels[image_number + row_number * 28:image_number + row_number * 28 + 28] for row_number in range(28)] for image_number in range(0, len(pixels), 28 * 28)]
    return list(map(lambda image: image_to_vector(image), images))

def read_label_vectors(filename, number=None, start=0):
    with open(filename, "rb") as file:
        file.seek(8 + start)
        if number is None:
            bytes = file.read()
        else:
            bytes = file.read(number)
    labels = list(bytes)
    return list(map(lambda label: label_to_vector(label), labels))

def dense_fnn(inputs, hidden_layer_sizes):
    hidden_layer_sizes.append(10)
    prev_layer = inputs
    prev_layer_size = 28 * 28
    for layer_number in range(len(hidden_layer_sizes)):
        layer_size = hidden_layer_sizes[layer_number]
        bias_variable = tf.get_variable("bias_" + str(layer_number), [1, layer_size], dtype=tf.float32)
        weight_variable = tf.get_variable("weight_" + str(layer_number), [prev_layer_size, layer_size], dtype=tf.float32)
        logits = tf.matmul(prev_layer, weight_variable) + bias_variable
        prev_layer = tf.sigmoid(logits)
        prev_layer_size = layer_size
    return logits

def train(inputs, logits, x_train_and_cv, y_train_and_cv, temperature, mini_batch_size=1000, cross_validation_set_size=10000, epochs=100):
    learning_rate = temperature * mini_batch_size
    x_train = x_train_and_cv[:-cross_validation_set_size]
    y_train = y_train_and_cv[:-cross_validation_set_size]
    x_cv = x_train_and_cv[-cross_validation_set_size:]
    y_cv = y_train_and_cv[-cross_validation_set_size:]
    labels = tf.placeholder(tf.float32, shape=(None, 10))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_data = list(zip(x_train, y_train))
    for epoch in range(epochs):
        random.shuffle(train_data)
        x_train, y_train = zip(*train_data)
        for mini_batch_start in range(0, len(x_train), mini_batch_size):
            sess.run(training_step, {inputs: x_train[mini_batch_start:mini_batch_start + mini_batch_size], labels: y_train[mini_batch_start:mini_batch_start + mini_batch_size]})
        if cross_validation_set_size > 0:
            accuracy_score = sess.run(accuracy, {inputs: x_cv, labels: y_cv})
            print("Cross-validation set accuracy after {0} epoch(s): {1}".format(epoch + 1, accuracy_score))
    return (lambda x_test, y_test : sess.run(accuracy, {inputs: x_test, labels: y_test})), accuracy_score

def optimize_temperature(inputs, logits, x_train_and_cv, y_train_and_cv):
    temperatures=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002, 0.000001, 0.0000005, 0.0000002, 0.0000001]
    def accuracy_score_of_temperature(temperature):
        print("Training with temperature {0}.".format(temperature))
        _, accuracy_score = train(inputs, logits, x_train_and_cv, y_train_and_cv, temperature, epochs=20)
        return accuracy_score
    return max(temperatures, key=accuracy_score_of_temperature)

inputs = tf.placeholder(tf.float32, shape=(None, 28 * 28))
logits = dense_fnn(inputs, hidden_layer_sizes=[100])
x_train_and_cv = read_image_vectors("train-images-idx3-ubyte")
y_train_and_cv = read_label_vectors("train-labels-idx1-ubyte")
temperature = optimize_temperature(inputs, logits, x_train_and_cv, y_train_and_cv)
print("Optimal temperature found: {0}.".format(temperature))
accuracy_evaluator, _ = train(inputs, logits, x_train_and_cv, y_train_and_cv, temperature, cross_validation_set_size=0)
x_test = read_image_vectors("t10k-images-idx3-ubyte")
y_test = read_label_vectors("t10k-labels-idx1-ubyte")
test_accuracy_score = accuracy_evaluator(x_test, y_test)
print("Test set accuracy: {0}.".format(test_accuracy_score))
