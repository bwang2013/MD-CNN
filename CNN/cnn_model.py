import tensorflow as tf


class TCNNConfig(object):
    num_classes = 4439
    num_filters = 4
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 30000
    print_per_batch = 10
    save_per_batch = 1000
    W = 100

class TextCNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [5, self.config.num_classes], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [1, self.config.num_classes], name='input_y')

        self.cnn()

    def cnn(self):
        with tf.name_scope("cnn"):
            conv1_1 = tf.layers.conv2d(self.input_x, self.config.num_filters, (5, 1), name='conv1_1')
            conv1_2 = tf.layers.conv2d(self.input_x, self.config.num_filters, (4, 1), name='conv1_2')
            conv1_3 = tf.layers.conv2d(self.input_x, self.config.num_filters, (3, 1), name='conv1_3')
            conv1_4 = tf.layers.conv2d(self.input_x, self.config.num_filters, (2, 1), name='conv1_4')

            maxpool1_1 = tf.nn.max_pool(conv1_1, [1, 1, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_1')
            maxpool1_2 = tf.nn.max_pool(conv1_2, [1, 2, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_2')
            maxpool1_3 = tf.nn.max_pool(conv1_3, [1, 3, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_3')
            maxpool1_4 = tf.nn.max_pool(conv1_4, [1, 4, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_4')

            concat = tf.concat([maxpool1_1, maxpool1_2, maxpool1_3, maxpool1_4], 1)
            fc1 = tf.layers.dense(concat, 1, name='fc1')
            conv2 = tf.layers.conv2d(fc1, self.config.num_filters, (2, 1), name='conv2')
            maxpool2 = tf.nn.max_pool(conv2, [1, 3, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_4')

        with tf.name_scope("score"):
            fc = tf.layers.dense(maxpool2, 1, name='fc2')
            self.y_pred = tf.sigmoid(fc)

        with tf.name_scope("optimize"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.input_y * tf.log(self.y_pred) * self.config.W + (1 - self.input_y) * tf.log(1 - self.y_pred)))
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))