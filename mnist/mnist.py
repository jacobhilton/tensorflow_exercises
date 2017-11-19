import tensorflow as tf

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

def train(hidden_layer_sizes=[], temperature=0.0001, batch_size=100, cross_validation_set_size=10000):
    learning_rate = temperature * batch_size
    training_set_size = 60000 - cross_validation_set_size
    inputs = tf.placeholder(tf.float32, shape=(None, 28 * 28))
    logits = dense_fnn(inputs, hidden_layer_sizes)
    labels = tf.placeholder(tf.float32, shape=(None, 10))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, training_set_size, batch_size):
        x_train = read_image_vectors("train-images-idx3-ubyte", number=batch_size, start=i)
        y_train = read_label_vectors("train-labels-idx1-ubyte", number=batch_size, start=i)
        sess.run(training_step, {inputs: x_train, labels: y_train})
    x_cv = read_image_vectors("train-images-idx3-ubyte", number=cross_validation_set_size, start=(60000 - cross_validation_set_size))
    y_cv = read_label_vectors("train-labels-idx1-ubyte", number=cross_validation_set_size, start=(60000 - cross_validation_set_size))
    return sess.run(accuracy, {inputs: x_cv, labels: y_cv})
