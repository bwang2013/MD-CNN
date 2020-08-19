import tensorflow as tf
import DNN.inference as inference
import os
import DNN.input_data as input_data

BATCH_SIZE = 32
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
EPOCHS = 3000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "dnn_model"

train_list_side, train_list_tag, _, _ = input_data.load_data_label('')
TRAIN_NUM_EXAMPLES = DATASET_SIZE = len(train_list_side)

def train():
    x = tf.placeholder(tf.float32, [inference.INPUT_NODE, inference.NUM_CLASS], name='x-input')
    y_ = tf.placeholder(tf.float32, [1, inference.NUM_CLASS], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, regularizer)
    # y = inference.inference_nlayer(x,regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)
    W = 100
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)*W+(1-y_)* tf.log(1-y)))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATASET_SIZE / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    min_loss = float("inf")
    best_epoch = None
    best_model = None
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(EPOCHS):
            # print(train_list_side.shape)
            # print(train_list_tag.shape)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: train_list_side,
                                                      y_: train_list_tag})
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # # early stopping
                # if loss_value < min_loss:
                #     min_loss = loss_value
                #     best_epoch = i
                #     best_model = sess
                # else:
                #     print("Early Stop")
                #     saver.save(best_model, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=best_epoch)
                #     break
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    train()
if __name__ == '__main__':
    tf.app.run()