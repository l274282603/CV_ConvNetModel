import tensorflow as tf
import numpy as np
import time
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]
class vgg16:
    def __init__(self):
        self.parameters = np.load("vgg16.npy", encoding='latin1').item()


    def vgg_buid(self, img):
        start_time = time.time()
        with tf.variable_scope("preprocess") as scop:
            mean = tf.constant(VGG_MEAN, dtype=tf.float32, shape=(1,1,1,3),name="imag_mean")
            image = img - mean

        self.conv1_1 = self.conv_layer(image, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, name="pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, name="pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, name="pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, name="pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.pool_layer(self.conv5_3, name="pool5")

        self.fc6 = self.fc_layer(self.pool5, name="fc6")
        self.fc6_relu = tf.nn.relu(self.fc6)

        self.fc7= self.fc_layer(self.fc6_relu, name="fc7")
        self.fc7_relu = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.fc7_relu, name="fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        print(("build model finished: %ds" % (time.time() - start_time)))

    def conv_layer(self, input, name):
        with tf.variable_scope(name) as scop:
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return conv

    def pool_layer(self, input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def fc_layer(self, input, name):
        # shape = tf.shape(input).aslist()
        shape = input.get_shape().as_list()
        print(shape)
        dims = 1
        for i in shape[1:]:
            dims = dims * i
        fc_input = tf.reshape(input, shape=(-1, dims))
        filter = self.get_conv_filter(name)
        bias = self.get_bias(name)
        fc = tf.nn.bias_add(tf.matmul(fc_input, filter), bias)
        return fc



    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name][0], name = "filter")

    def get_bias(self, name):
        return tf.constant(self.parameters[name][1], name = "bias")



def predict(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    print(len(synset))

    prob = prob.reshape(-1)
    preb_index = np.argsort(prob)[::-1]
    print(preb_index.shape)

    top1 = synset[preb_index[0]]
    print("top1:",top1,"--",prob[preb_index[0]])

    top5 = [(synset[preb_index[i]], prob[preb_index[i]]) for i in range(5)]
    print(("Top5: ", top5))

    return top1

if __name__ == '__main__':
    image = cv2.imread("tiger.jpeg")
    image = cv2.resize(src=image, dsize=(224,224))
    image = image.reshape(1,224,224,3)
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name= "input")

    vgg = vgg16()
    vgg.vgg_buid(input)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prob_result = sess.run(vgg.prob, feed_dict={input: image})
        print("prob_result.shape:",prob_result.shape)

        predict(prob_result, "synset.txt")
