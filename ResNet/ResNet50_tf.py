
import math
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.slim.nets import inception

class ResNet50():
    def __init__(self, parameter_path=None):
        if parameter_path:
            self.parameter_dict = np.load(parameter_path, encoding="latin1").item()
        else:
            self.parameter_dict = {}
        self.is_training = True

    def set_training(self, is_training):
        self.is_training = is_training


    def bulid(self, image):
        RGB_MEAN = [103.939, 116.779, 123.68]
        # image_r, image_g, image_b = tf.split(value=image, num_or_size_splits=3, axis=3)
        # assert image_r.get_shape().as_list()[1:] == [224, 224, 1]
        # assert image_g.get_shape().as_list()[1:] == [224, 224, 1]
        # assert image_b.get_shape().as_list()[1:] == [224, 224, 1]
        with tf.variable_scope("preprocess"):
            mean = tf.constant(value=RGB_MEAN, dtype=tf.float32, shape=[1,1,1,3], name="preprocess_mean")
            image = image - mean

        self.conv1 = self._conv_layer(image, stride=2, filter_size=7, in_channels=3, out_channels=64, name="conv1")
        self.conv1_bn = self.batch_norm(self.conv1)
        self.conv1_relu = tf.nn.relu(self.conv1_bn)
        print("self.conv1_relu.shape={}".format(self.conv1_relu.get_shape()))

        self.pool1 = self._max_pool(self.conv1_relu, filter_size=3, stride=2)
        self.block1_1 = self._bottleneck(self.pool1, filters=(64, 64, 256), name="block1_1", channge_dimens=True)
        self.block1_2 = self._bottleneck(self.block1_1, filters=(64, 64, 256), name="block1_2", channge_dimens=False)
        self.block1_3 = self._bottleneck(self.block1_2, filters=(64, 64, 256), name="block1_3", channge_dimens=False)
        print("self.block1_3.shape={}".format(self.block1_3.get_shape()))

        self.block2_1 = self._bottleneck(self.block1_3, filters=(128, 128, 512), name="block2_1", channge_dimens=True, block_stride=2)
        self.block2_2 = self._bottleneck(self.block2_1, filters=(128, 128, 512), name="block2_2", channge_dimens=False)
        self.block2_3 = self._bottleneck(self.block2_2, filters=(128, 128, 512), name="block2_3", channge_dimens=False)
        self.block2_4 = self._bottleneck(self.block2_3, filters=(128, 128, 512), name="block2_4", channge_dimens=False)
        print("self.block2_4.shape={}".format(self.block2_4.get_shape()))

        self.block3_1 = self._bottleneck(self.block2_4, filters=(256, 256, 1024), name="block3_1", channge_dimens=True,
                                         block_stride=2)
        self.block3_2 = self._bottleneck(self.block3_1, filters=(256, 256, 1024), name="block3_2", channge_dimens=False)
        self.block3_3 = self._bottleneck(self.block3_2, filters=(256, 256, 1024), name="block3_3", channge_dimens=False)
        self.block3_4 = self._bottleneck(self.block3_3, filters=(256, 256, 1024), name="block3_4", channge_dimens=False)
        self.block3_5 = self._bottleneck(self.block3_4, filters=(256, 256, 1024), name="block3_5", channge_dimens=False)
        self.block3_6 = self._bottleneck(self.block3_5, filters=(256, 256, 1024), name="block3_6", channge_dimens=False)
        print("self.block3_6.shape={}".format(self.block3_6.get_shape()))

        self.block4_1 = self._bottleneck(self.block3_6, filters=(512, 512, 2048), name="block4_1", channge_dimens=True,
                                         block_stride=2)
        self.block4_2 = self._bottleneck(self.block4_1, filters=(512, 512, 2048), name="block4_2", channge_dimens=False)
        self.block4_3 = self._bottleneck(self.block4_2, filters=(512, 512, 2048), name="block4_3", channge_dimens=False)
        self.block4_4 = self._bottleneck(self.block4_3, filters=(512, 512, 2048), name="block4_4", channge_dimens=False)

        print("self.block4_4.shape={}".format(self.block4_4.get_shape()))
        self.pool2 = self._avg_pool(self.block4_4, filter_size=7, stride=1, )
        print("self.pool2.shape={}".format(self.pool2.get_shape()))
        self.fc = self._fc_layer(self.pool2, in_size=2048, out_size=1000, name="fc1200")

        return self.fc



    def _bottleneck(self, input, filters, name, channge_dimens, block_stride=1):
        filter1, filter2, filter3 = filters
        input_shortcut = input
        input_channel = input.get_shape().as_list()[-1]

        block_conv_1 = self._conv_layer(input, block_stride, 1, input_channel, filter1, name=name+"_Conv1")
        block_bn1 = self.batch_norm(block_conv_1)
        block_relu1 = tf.nn.relu(block_bn1)

        block_conv_2 = self._conv_layer(block_relu1, 1, 3, filter1, filter2, name=name + "_Conv2")
        block_bn2 = self.batch_norm(block_conv_2)
        block_relu2 = tf.nn.relu(block_bn2)

        block_conv_3 = self._conv_layer(block_relu2, 1, 1, filter2, filter3, name=name + "_Conv3")
        block_bn3 = self.batch_norm(block_conv_3)

        if channge_dimens:
            input_shortcut = self._conv_layer(input, block_stride, 1, input_channel, filter3, name=name+"_ShortcutConv")
            input_shortcut = self.batch_norm(input_shortcut)

        block_res = tf.nn.relu(tf.add(input_shortcut, block_bn3))

        return block_res



    def batch_norm(self, input):
        return tf.layers.batch_normalization(inputs=input, axis=3, momentum=0.99,
                                             epsilon=1e-12, center=True, scale=True,
                                             training=self.is_training)

    def _avg_pool(self, input, filter_size, stride, padding="VALID"):
        return tf.nn.avg_pool(input, ksize=[1, filter_size, filter_size, 1],
                              strides=[1, stride, stride, 1], padding=padding)

    def _max_pool(self, input, filter_size, stride, padding="SAME"):
        return tf.nn.max_pool(input, ksize=[1, filter_size, filter_size, 1],
                              strides=[1, stride, stride, 1], padding=padding)

    def _conv_layer(self, input, stride, filter_size, in_channels, out_channels, name, padding="SAME"):
        '''
        定义卷积层
        '''
        with tf.variable_scope(name):
            conv_filter, bias = self._get_conv_parameter(filter_size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(input, filter=conv_filter, strides=[1, stride, stride, 1], padding=padding)
            conv_bias = tf.nn.bias_add(conv, bias)
            return conv_bias

    def _fc_layer(self, input, in_size, out_size, name):
        '''
        定义全连接层
        '''
        with tf.variable_scope(name):
            input = tf.reshape(input, [-1, in_size])
            fc_weights, fc_bais = self._get_fc_parameter(in_size, out_size, name)
            fc = tf.nn.bias_add(tf.matmul(input, fc_weights), fc_bais)
            return fc


    def _get_conv_parameter(self, filter_size, in_channels, out_channels, name):
        '''
        用于获取卷积层参数
        :param filter_size:  卷积核大小
        :param in_channel:    卷积核channel
        :param out_channel:   卷积输出的channel,也就是卷积核个数
        :param name:         当前卷积层name
        :return: 返回对应卷积核 和 偏置
        '''
        if name in self.parameter_dict:
            conv_filter_initValue = self.parameter_dict[name][0];
            bias_initValue = self.parameter_dict[name][1]
        else:
            conv_filter_initValue = tf.truncated_normal(shape=[filter_size, filter_size, in_channels, out_channels],
                                            mean=0.0, stddev=1 / math.sqrt(float(filter_size * filter_size)))
            bias_initValue = tf.truncated_normal(shape=[out_channels], mean=0.0, stddev=1.0)

        conv_filter_value = tf.Variable(initial_value=conv_filter_initValue, name=name+"_weights")
        bias = tf.Variable(initial_value=bias_initValue, name=name+"_biases")

        return conv_filter_value, bias

    def _get_fc_parameter(self, in_size, out_size, name):
        '''
        用于获取全连接层参数
        :param in_size:
        :param out_size:
        :param name:
        :return:
        '''
        if name in self.parameter_dict:
            fc_weights_initValue = self.parameter_dict[name][0]
            fc_bias_initValue = self.parameter_dict[name][1]
        else:
            fc_weights_initValue = tf.truncated_normal(shape=[in_size, out_size], mean=0.0,
                                                       stddev=1.0 / math.sqrt(float(in_size)))
            fc_bias_initValue = tf.truncated_normal(shape=[out_size], mean=0.0, stddev=1.0)

        fc_weights = tf.Variable(initial_value=fc_weights_initValue, name=name+"_weights")
        fc_bias = tf.Variable(initial_value=fc_bias_initValue, name=name+"_biases")
        return fc_weights, fc_bias

    def save_npy(self, sess, npy_path="./model/Resnet-save.npy"):
        """
        Save this model into a npy file
        """
        assert isinstance(sess, tf.Session)

        self.data_dict = None
        data_dict = {}

        for (name, idx), var in list(self.parameter_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.parameter_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="input")
    resnet = ResNet50()
    out_put = resnet.bulid(input)
    print(out_put.get_shape())