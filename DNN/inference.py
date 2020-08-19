import tensorflow as tf
import math

NUM_CLASS = 4439
INPUT_NODE = 5
OUTPUT_NODE = 1
LAYER1_NODE = 7
LAYER2_NODE = 9
LAYER3_NODE = 11
# expr = 0.43 * INPUT_NODE * 5 + 0.12 * 5 * 5 + 2.54 * INPUT_NODE + 0.77 * 5 + 0.35
# LAYER1_NODE = int(math.sqrt(expr) + 0.51)

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope('layer3'):
        weights = get_weight_variable([LAYER2_NODE, LAYER3_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER3_NODE], initializer=tf.constant_initializer(0.1))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    with tf.variable_scope('layer4'):
        weights = get_weight_variable([LAYER3_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer4 = tf.sigmoid(tf.matmul(layer3, weights) + biases)

    return layer4


def get_weight(shape, regularizer):
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(var))
    return var


# def inference_nlayer(input_tensor, regularizer):
#     layer_dimension = [INPUT_NODE, 100, 100, 100, OUTPUT_NODE]
#     n_layers = len(layer_dimension)
#
#     cur_layer = input_tensor
#     in_dimension = layer_dimension[0]
#
#     for i in range(1, n_layers):
#         out_dimension = layer_dimension[i]
#         weight = get_weight([in_dimension, out_dimension], regularizer)
#         bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
#
#         cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
#         in_dimension = layer_dimension[i]
#     return cur_layer