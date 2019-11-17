import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

class SqueezeNet():
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
        with tf.variable_scope("preprocess"):
            mean = tf.constant(value=RGB_MEAN, dtype=tf.float32, shape=[1, 1, 1, 3], name="preprocess_mean")
            image = image - mean

        self.conv1 = self._conv_layer(image, stride=2, filter_size=7, in_channels=3, out_channels=96, name="conv1") #112
        self.conv1_relu = tf.nn.relu(self.conv1)
        self.maxpool1 = self._max_pool(self.conv1_relu, filter_size=3, stride=2)   #56

        self.Fire2 = self._Fire(self.maxpool1, 96, 16, 64, 64, name="Fire2_")
        self.Fire3 = self._Fire(self.Fire2, 128, 16, 64, 64, name="Fire3_")
        self.Fire4 = self._Fire(self.Fire3, 128, 32, 128, 128, name="Fire4_")

        self.maxpool2 = self._max_pool(self.Fire4, filter_size=3, stride=2, padding="VALID")  #27

        self.Fire5 = self._Fire(self.maxpool2, 256, 32, 128, 128, name="Fire5_")
        self.Fire6 = self._Fire(self.Fire5, 256, 48, 192, 192, name="Fire6_")
        self.Fire7 = self._Fire(self.Fire6, 384, 48, 192, 192, name="Fire7_")
        self.Fire8 = self._Fire(self.Fire7, 384, 64, 256, 256, name="Fire8_")

        self.maxpool3 = self._max_pool(self.Fire8, filter_size=3, stride=2, padding="VALID")  #13

        self.Fire9 = self._Fire(self.maxpool3, 512, 54, 256, 256, name="Fire9_")
        # self.droup = tf.nn.dropout(self.Fire9, keep_prob=0.5)
        self.conv10 = self._conv_layer(self.Fire9, stride=1, filter_size=1, in_channels=512, out_channels=10,
                                       name="conv10")

        print("self.conv10.get_shape()={}".format(self.conv10.get_shape()))
        self.avgpool = self._avg_pool(self.conv10, filter_size=13, stride=1)
        print("self.avgpool.get_shape()={}".format(self.avgpool.get_shape()))
        return tf.squeeze(self.avgpool, [1, 2])




    def _Fire(self, input, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, name):
        self.squeeze_conv = self._conv_layer(input, stride=1, filter_size=1,
                                             in_channels=in_channels, out_channels=squeeze_channels,
                                             name=name+"squeeze_conv")
        self.squeeze_conv_relu = tf.nn.relu(self.squeeze_conv)

        self.expand1x1_conv = self._conv_layer(self.squeeze_conv_relu, stride=1, filter_size=1,
                                               in_channels=squeeze_channels, out_channels=expand1x1_channels,
                                               name=name+"expand1x1_conv")
        self.expand1x1_conv_relu = tf.nn.relu(self.expand1x1_conv)

        self.expand3x3_conv = self._conv_layer(self.squeeze_conv_relu, stride=1, filter_size=3,
                                               in_channels=squeeze_channels, out_channels=expand3x3_channels,
                                               name=name + "expand3x3_conv")
        self.expand3x3_conv_relu = tf.nn.relu(self.expand3x3_conv)

        return tf.concat([self.expand1x1_conv_relu, self.expand3x3_conv_relu], axis=3)



    def _batch_norm(self, input):
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
            conv_filter_value = tf.Variable(initial_value=conv_filter_initValue, name=name + "_weights")
            bias = tf.Variable(initial_value=bias_initValue, name=name + "_biases")
        else:

            conv_filter_value = tf.get_variable(name=name+"_weights",
                                                shape=[filter_size, filter_size, in_channels, out_channels],
                                                initializer=tf.contrib.keras.initializers.he_normal())
            bias = tf.get_variable(name=name+"_biases", shape=[out_channels],
                                   initializer=tf.constant_initializer(0.1, dtype=tf.float32))


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
            fc_weights = tf.Variable(initial_value=fc_weights_initValue, name=name + "_weights")
            fc_bias = tf.Variable(initial_value=fc_bias_initValue, name=name + "_biases")
        else:
            fc_weights = tf.get_variable(name=name + "_weights",
                                                shape=[in_size, out_size],
                                                initializer=tf.contrib.keras.initializers.he_normal())
            fc_bias = tf.get_variable(name=name + "_biases", shape=[out_size],
                                   initializer=tf.constant_initializer(0.1, dtype=tf.float32))

        return fc_weights, fc_bias

if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="input")
    resnet = SqueezeNet()
    out_put = resnet.bulid(input)
    print(out_put.get_shape())