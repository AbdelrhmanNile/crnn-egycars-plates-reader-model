import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import BatchNormalization, Reshape, Lambda
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import add, concatenate

from parameter import *


# VGG Blocks
class VggBlock1(keras.layers.Layer):
    def __init__(self, num_filters, do_maxpool: bool = True, **kwargs):
        super(VggBlock1, self).__init__(**kwargs)
        self.do_maxpool = do_maxpool
        self.conv = Conv2D(
            num_filters, (3, 3), padding="same", kernel_initializer="he_normal"
        )
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        if self.do_maxpool:
            self.pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        if self.do_maxpool:
            x = self.pool(x)
        return x


class VggBlock2(keras.layers.Layer):
    def __init__(self, num_filters, **kwargs):
        super(VggBlock2, self).__init__(**kwargs)

        self.conv1 = Conv2D(
            num_filters, (3, 3), padding="same", kernel_initializer="he_normal"
        )
        self.bn1 = BatchNormalization()
        self.relu1 = Activation("relu")
        self.conv2 = Conv2D(
            num_filters, (3, 3), padding="same", kernel_initializer="he_normal"
        )
        self.bn2 = BatchNormalization()
        self.relu2 = Activation("relu")
        self.pool = MaxPooling2D(pool_size=(1, 2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        return x


# RNN
class BI_LSTM_Block(keras.layers.Layer):
    def __init__(self, num_units, merge_mode, **kwargs):
        super(BI_LSTM_Block, self).__init__(**kwargs)

        self.bi_lstm = Bidirectional(
            LSTM(num_units, return_sequences=True, kernel_initializer="he_normal"),
        merge_mode=merge_mode)
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = self.bi_lstm(inputs)
        x = self.bn(x)
        return x


class CTCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


class OCR_Model(keras.Model):
    def __init__(self, **kwargs):
        super(OCR_Model, self).__init__(**kwargs)

        self.input_sahpe = (128, 128, 1)

        # self.input_layer = Input(shape=self.input_sahpe, name='input_layer', dtype='float32')

        # VGG (CNN)
        self.vgg_block1 = VggBlock1(64, name="VGG_Block1")
        self.vgg_block2 = VggBlock1(128, name="VGG_Block2")
        self.vgg_block3 = VggBlock2(256, name="VGG_Block3")
        self.vgg_block4 = VggBlock2(512, name="VGG_Block4")
        self.vgg_block5 = VggBlock1(512, do_maxpool=False, name="VGG_Block5")

        # CNN to RNN
        new_shape = ((self.input_sahpe[0] // 16), (self.input_sahpe[1] // 4) * 512)
        self.reshape = Reshape(target_shape=new_shape, name="reshape")
        self.dense1 = Dense(
            64, activation="relu", kernel_initializer="he_normal", name="dense1"
        )

        # RNN
        self.bi_lstm1 = BI_LSTM_Block(256, name="BI_LSTM_Block1")
        self.bi_lstm2 = BI_LSTM_Block(256, name="BI_LSTM_Block2")

        ## ------------ change the number of classes
        self.dense2 = Dense(
            30, activation="softmax", kernel_initializer="he_normal", name="dense2"
        )
        self.ctc_loss = CTCLayer(name="ctc_loss")

    def call(self, inputs):
        # x = self.input_layer(inputs)

        x = self.vgg_block1(inputs)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        x = self.vgg_block4(x)
        x = self.vgg_block5(x)

        x = self.reshape(x)
        x = self.dense1(x)

        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)

        x = self.dense2(x)

        labels = Input(name="the_labels", shape=(None,), dtype="float32")
        output = self.ctc_loss(labels, x)

        return output
    

def ctc_lamda_func(args):
    y_pred, labels, input_length, label_length = args
    
    y_pred = y_pred[:, 2:, :]
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(training):
    
    input_shape = (128, 64, 1)
    
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # VGG (CNN)
    x = VggBlock1(64, name="VGG_Block1")(inputs)
    x = VggBlock1(128, name="VGG_Block2")(x)
    x = VggBlock2(256, name="VGG_Block3")(x)
    x = VggBlock2(512, name="VGG_Block4")(x)
    x = VggBlock1(512, do_maxpool=False, name="VGG_Block5")(x)
    
    # CNN to RNN
    new_shape = (32, 2048)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal", name="dense1")(x)
    
    # RNN
    x = BI_LSTM_Block(256, merge_mode='sum', name="BI_LSTM_Block1")(x)
    x = BI_LSTM_Block(128, merge_mode='concat', name="BI_LSTM_Block2")(x)
    
    x = Dropout(0.25)(x)
    
    # RNN output to character activations
    x = Dense(num_classes, kernel_initializer="he_normal", name="dense2")(x)
    y_pred = Activation('softmax', name='softmax')(x)
    
    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lamda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    if training:
        return keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return keras.Model(inputs=[inputs], outputs=y_pred)

def CRNN_model(is_training=True):
    input_shape = (128, 64, 1)
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1)
    batchnorm_2 = BatchNormalization()(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(batchnorm_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
    batchnorm_4 = BatchNormalization()(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(batchnorm_4)

    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_4)
    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_5)
    batchnorm_6 = BatchNormalization()(conv_6)

    bn_shape = batchnorm_6.get_shape()  


    print(bn_shape) 

    x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(batchnorm_6)
    drop_reshape = Dropout(0.25)(x_reshape)
    fc_1 = Dense(256, activation='relu')(drop_reshape)  

    print(x_reshape.get_shape())  
    print(fc_1.get_shape()) 

    bi_LSTM_1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(fc_1)
    bi_LSTM_2 = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(bi_LSTM_1)

    drop_rnn = Dropout(0.3)(bi_LSTM_2)

    fc_2 = Dense(num_classes, kernel_initializer='he_normal', activation='softmax')(drop_rnn)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lamda_func, output_shape=(1,), name='ctc')([fc_2, labels, input_length, label_length])

    if is_training:
        return keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    else:
        return keras.Model(inputs=[inputs], outputs=fc_2)
