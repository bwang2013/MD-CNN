# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import DNN.inference as inference
import numpy as np
import DNN.input_data as input_data

MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "dnn_model"
print(MODEL_SAVE_PATH)
EVAL_INTERVAL_SECS = 2

_, _, test_list_side, test_list_tag = input_data.load_data_label('')

def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.int64, name='y-input')
        validate_feed = {x: test_list_side, y_: test_list_tag}

        y = inference.inference(x, None)
        # y = character_inference.inference_nlayer(x, None)

        # correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    # accuracy_score = get_acc(sess,true_y, pred_y)
                    # print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))

                    # print("the input data are \n%s" % test_list_side)
                    # print("the truly answer are \n%s" % test_list_tag)
                    eval_aws = sess.run(y, feed_dict=validate_feed)
                    print("the evaluate answer are \n%s" % eval_aws)

                    # accuracy_score, acc_list = get_acc(sess, text_list_tag, eval_aws)
                    # print("After %s training step(s), all test accuracy = %g" % (global_step, accuracy_score))

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    evaluate()
if __name__ == '__main__':
    tf.app.run()