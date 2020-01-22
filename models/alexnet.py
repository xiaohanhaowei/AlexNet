
import tensorflow as tf

K_BIAS = 2
N_DEPTH_RADIUS = 5
ALPHA = 1e-4
BETA = 0.75

dataset_dict = {
    'image_size': 224,
    'channels': 3,
    'lables': 1000
}

Conv_Filter_Shape = {
    "conv1": [11, 11, 3, 96],
    "conv2": [5, 5, 96, 256],
    "conv3": [3, 3, 256, 384],
    "conv4": [3, 3, 384, 384],
    "conv5": [3, 3, 384, 256]
}

Conv_Filter_Shape_origin = {
    "conv1": [11, 11, 3, 48],
    "conv2": [5, 5, 48, 128],
    "conv3": [3, 3, 128, 192],
    "conv4": [3, 3, 192, 192],
    "conv5": [3, 3, 192, 128]
}

Fc_Filter_Shape = {
    'fc1_shape': [13 * 13 * 256, 4096],
    'fc2_shape': [4096, 4096],
    'fc3_shape': [4096, dataset_dict['lables']]
}
Fc_Filter_Shape_origin = {
    'fc1_shape': [13 * 13 * 128, 2048],
    'fc2_shape': [2048, 2048],
    'fc3_shape': [2048, dataset_dict['lables']]
}

weights = {
    'cw1': tf.Variable(tf.random.truncated_normal(Conv_Filter_Shape['conv1']), name='c1_weights'),    
    'cw2': tf.Variable(tf.random.truncated_normal(Conv_Filter_Shape['conv2']), name='c2_weights'),    
    'cw3': tf.Variable(tf.random.truncated_normal(Conv_Filter_Shape['conv3']), name='c3_weights'),
    'cw4': tf.Variable(tf.random.truncated_normal(Conv_Filter_Shape['conv4']), name='c4_weights'),
    'cw5': tf.Variable(tf.random.truncated_normal(Conv_Filter_Shape['conv5']), name='c5_weights'),
    'fc1': tf.Variable(tf.random.truncated_normal(Fc_Filter_Shape['fc1_shape']), name='fc1_weights'),
    'fc2': tf.Variable(tf.random.truncated_normal(Fc_Filter_Shape['fc2_shape']), name='fc2_weights'),
    'fc3': tf.Variable(tf.random.truncated_normal(Fc_Filter_Shape['fc3_shape']), name='fc3_weights')
}

biases = {
    'c_b1': tf.Variable(tf.random.truncated_normal([Conv_Filter_Shape['conv1'][3]]), name='c_bias1'),
    'c_b2': tf.Variable(tf.random.truncated_normal([Conv_Filter_Shape['conv2'][3]]), name='c_bias2'),
    'c_b3': tf.Variable(tf.random.truncated_normal([Conv_Filter_Shape['conv3'][3]]), name='c_bias3'),
    'c_b4': tf.Variable(tf.random.truncated_normal([Conv_Filter_Shape['conv4'][3]]), name='c_bias4'),
    'c_b5': tf.Variable(tf.random.truncated_normal([Conv_Filter_Shape['conv5'][3]]), name='c_bias5'),
    'fc_b1': tf.Variable(tf.random.truncated_normal([Fc_Filter_Shape['fc1_shape'][1]]), name='fc_bias1'),
    'fc_b2': tf.Variable(tf.random.truncated_normal([Fc_Filter_Shape['fc2_shape'][1]]), name='fc_bias2'),
    'fc_b3': tf.Variable(tf.random.truncated_normal([Fc_Filter_Shape['fc3_shape'][1]]), name='fc_bias3'),
}


class AlexNet():
    def __init__(self, is_traing=False):
        self.input_images = tf.compat.v1.placeholder(dtype=tf.float32,
                                                shape=(None, 224, 224, 3),
                                                name='input_placeholder')
        self.lables = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, dataset_dict['lables']))
        self.output = None
        self._build_model()

    def _build_model(self):
        with tf.name_scope('conv1') as scope:
            conv1 = tf.nn.conv2d(self.input_images, weights['cw1'], strides=[1, 4, 4, 1], padding='SAME', name='conv1')
            conv1 = tf.nn.bias_add(conv1, biases['c_b1'])
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.lrn(conv1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.name_scope(name='conv2') as scope:
            conv2 = tf.nn.conv2d(conv1, weights['cw2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
            conv2 = tf.nn.bias_add(conv2, biases['c_b2'])
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.nn.lrn(conv2, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.name_scope('conv3') as scope:
            conv3 = tf.nn.conv2d(conv2, weights['cw3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
            conv3 = tf.nn.bias_add(conv3, biases['c_b3'])
            conv3 = tf.nn.relu(conv3)
        with tf.name_scope('conv4') as scope:
            conv4 = tf.nn.conv2d(conv3, weights['cw4'], strides=[1, 1, 1, 1], padding='SAME', name='conv4')
            conv4 = tf.nn.bias_add(conv4, biases['c_b4'])
            conv4 = tf.nn.relu(conv4)
        with tf.name_scope('conv5') as scope:
            conv5 = tf.nn.conv2d(conv4, weights['cw5'], strides=[1, 1, 1, 1], padding='SAME', name='conv5')
            # after_conv_shape_5 = tf.shape(conv5)
            # after_conv_shape_5_print = tf.Print(conv5,
                                                # [after_conv_shape_5],
                                                # 'after 5th conv')
            conv5 = tf.nn.bias_add(conv5, biases['c_b5'])
            conv5 = tf.nn.relu(conv5)
            conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        with tf.name_scope('fc1') as scope:
            shape = [-1, weights['fc1'].get_shape().as_list()[0]]
            feature_map = tf.reshape(conv5, shape)
            fclayer_1 = tf.matmul(feature_map, weights['fc1'])
            fclayer_1 = tf.nn.dropout(fclayer_1, keep_prob=0.5)
        with tf.name_scope('fc2') as scope:
            fclayer_2 = tf.matmul(fclayer_1, weights['fc2'])
            fclayer_2 = tf.nn.dropout(fclayer_2, keep_prob=0.5)
        with tf.name_scope('out') as scope:
            fclayer_3 = tf.matmul(fclayer_2, weights['fc3'])
            self.output = tf.nn.softmax(fclayer_3, name='output_node')

    def _build_loss(self):
        pass
