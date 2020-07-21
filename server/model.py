#!/usr/bin/python

import numpy as np
import math
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

class CustomModel(Model):
    def train_step(self, data):
        predictionProbability = lambda x, y: math.exp(self.predict(x)) / (math.exp(self.predict(x)) - math.exp(self.predict(y)))
        compute_loss = lambda data: -1 * sum([mu[0] * math.log(predictedProbability(x,y)) + mu[1] * math.log(predictedProbability(y,x)) for x, y, mu in data])

        with tf.GradientTape() as tape:
            loss = compute_loss(data)
        
        gradients = tape.gradients(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return {"loss": loss}

def createModel(input_shape):
    inputA = Input(shape=input_shape, name='inputA')
    inputB = Input(shape=input_shape, name='inputB')
    inputC = Input(shape=input_shape, name='inputC')

    x = Conv2D(32, 8, (4, 4), activation='relu')(inputA)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 4, (2, 2), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Model(inputs=inputA, outputs=x)

    y = Conv2D(32, 8, (4, 4), activation='relu')(inputB)
    y = MaxPooling2D()(y)
    y = Conv2D(64, 4, (2, 2), activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    y = Dense(256, activation='relu')(y)
    y = Model(inputs=inputB, outputs=y)

    z = Conv2D(32, 8, (4, 4), activation='relu')(inputC)
    z = MaxPooling2D()(z)
    z = Conv2D(64, 4, (2, 2), activation='relu')(z)
    z = MaxPooling2D()(z)
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)
    z = Model(inputs=inputC, outputs=z)

    combined = Concatenate()([x.output, y.output, z.output])
    final = Dense(128, activation='relu')(combined)
    final = Dense(32, activation='relu')(final)
    final = Dense(16, activation='relu')(final)
    final = Dense(1, activation='linear')(final)
    model = Model(inputs=[x.input, y.input, z.input], outputs=[final])
    return model
