import os
import numpy as np
from keras.models import *
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Reshape,Dense,Flatten
from keras.optimizers import *
from keras.layers import merge,Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from data import *

import random

class gnet(object):

    def __init__(self, img_rows = 512, img_cols = 512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        data = dataProcess(self.img_rows, self.img_cols)
        imgs_train, img_mask_trian = data.load_train_data()
        imgs_test = data.load_test_data()
        return imgs_train,img_mask_trian,imgs_test

    def gnet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv11 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(inputs)
        conv12 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)


        conv21 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(pool1)
        conv22 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv21)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)


        conv31 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(pool2)
        conv32 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv31)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv32)


        conv41 = Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(pool3)
        conv42 = Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv41)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv42)
        drop4 = Dropout(0.5)(pool4)


        conv51 = Convolution2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(drop4)
        conv52 = Convolution2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv51)
        drop5 = Dropout(0.5)(conv52)


        up6 = UpSampling2D(size=(2, 2))(drop5)
        conv61 = Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(up6)
        conv62 = Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv61)


        up7 = concatenate([Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                                         activation='relu',padding='same')
                                        (UpSampling2D(size=(2, 2))(conv62)), conv32], axis=-1)
        conv71 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(up7)
        conv72 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv71)

        up8 = concatenate([Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                         activation='relu',padding='same')
                                        (UpSampling2D(size=(2, 2))(conv72)), conv22], axis=-1)
        conv81 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(up8)
        conv82 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv81)

        up9 = UpSampling2D(size=(2,2))(conv82)
        conv91 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(up9)
        conv92 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                               activation='relu', padding='same')(conv91)


        conv10 = Convolution2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
                               activation='sigmoid')(conv92)

        g_model = Model(outputs=conv10, inputs=inputs)

        g_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        g_model.summary()

        return g_model

    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.gnet()
        print("got gnet")
        ModelCheckpoint('gnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=1, verbose=1, validation_split=0.1, shuffle=True,
                  callbacks=[TensorBoard(log_dir='/home/704/code/mycode2/dlog')])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        results = os.path.join("results/")

        np.save(results + 'imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('results/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("results/%d.jpg" % (i))

class dnet(object):
    def __init__(self, img_rows = 40, img_cols = 40):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        data = dataProcess(self.img_rows, self.img_cols)
        imgs_nodules, imgs_results = data.load_nodule_data()
        return imgs_nodules, imgs_results

    def contrastive_loss(self,y_true, y_pred):

        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    def euclidean_distance(self,vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def compute_accuracy(y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)

    def create_pairs(self,x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(2)]) - 1
        for d in range(2):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 2)
                dn = (d + inc) % 2
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def dnet(self,input_shape):

        # inputs = Input(shape=input_shape)
        #
        # expand =  K.expand_dims(inputs,-1)
        #
        # conv1 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
        #                       activation='relu')(expand)
        #
        #
        # conv2 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1,1),
        #                        activation='relu')(conv1)
        #
        # conv3 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1,1),
        #                        activation='relu')(conv2)
        #
        # flatten = Flatten()(conv3)
        #
        # fc = Dense(2, activation='relu')(flatten)
        #
        # return Model(inputs, fc)
        input = Input(shape=input_shape)

        x = Flatten()(input)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)

        return Model(input, x)


    def train(self):
        print("loading data")

        (imgs_nodules, imgs_results) = self.load_data()
        print("loading data done")

        imgs_results = imgs_results.reshape(imgs_results.shape[0],imgs_results.shape[1],imgs_results.shape[2])
        imgs_nodules = imgs_nodules.reshape(imgs_nodules.shape[0],imgs_nodules.shape[1],imgs_nodules.shape[2])
        indices = [np.where(y == i)[0] for i in range(2)]
        tr_pairs, tr_y = self.create_pairs(imgs_nodules, indices)
        indices = [np.where(y == i)[0] for i in range(2)]
        te_pairs, te_y = self.create_pairs(imgs_results, indices)
        input_shape = imgs_results.shape[1:3]
        net = self.dnet(input_shape)

        input_shape = imgs_results.shape[1:3]

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        process_a = net(input_a)
        process_b = net(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([process_a, process_b])

        model = Model([input_a, input_b], distance)
        model.summary()
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss, optimizer=rms)
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=64,
                  epochs=10000,
                  callbacks=[TensorBoard(log_dir='/home/704/code/mycode2/glog')])

if __name__ == '__main__':


    # d_net = dnet()
    # d_net.train()

    g_net = gnet()
    g_net.train()
    g_net.save_img()

