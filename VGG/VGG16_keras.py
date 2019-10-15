from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Model
import cv2
import numpy as np

def VGG16(input):
    # Block 1
    X = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
               padding="same", name="block1_conv1")(input)
    X = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
               padding="same", name="block1_conv2")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(X)

    # Block 2
    X = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
               padding="same", name="block2_conv1")(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
               padding="same", name="block2_conv2")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(X)

    # Block 3
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv1")(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv2")(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv3")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(X)

    # Block 4
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv1")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv3")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(X)

    # Block 5
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv1")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv3")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool")(X)
    print(X.shape)
    X = Flatten(name='flatten')(X)
    X = Dense(4096, activation='relu', name='fc1')(X)
    X = Dense(4096, activation='relu', name='fc2')(X)
    X = Dense(1000, activation='softmax', name='predictions')(X)
    return X

def predict(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    print(len(synset))

    preb_index = np.argsort(prob)[::-1]

    top1 = synset[preb_index[0]]
    print("top1:",top1,"--",prob[preb_index[0]])

    top5 = [(synset[preb_index[i]], prob[preb_index[i]]) for i in range(5)]
    print(("Top5: ", top5))

    return top1

if __name__ == '__main__':
    image = cv2.imread("car.jpg")
    image = cv2.resize(src=image, dsize=(224, 224))
    image = image.reshape(1, 224, 224, 3)

    input = Input(shape=(224, 224, 3))
    vgg_output = VGG16(input)
    model = Model(input, vgg_output)

    model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)

    prob = model.predict(image)
    print(prob, prob.shape)
    predict(prob[0], "synset.txt")