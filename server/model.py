from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import *


class ConvNet:
    def __init__(self, input_shape, action_space):
        inputA = Input(shape=input_shape, name='inputA')
        inputB = Input(shape=input_shape, name='inputB')
        inputC = Input(shape=input_shape, name='inputC')

        x = TimeDistributed(Conv2D(32, 8, (4, 4), activation='relu'))(inputA)
        x = TimeDistributed(MaxPooling2D())(x)
        x = TimeDistributed(Conv2D(64, 1, (2, 2), activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D())(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(128)(x)
        x = models.Model(inputs=inputA, outputs=x)

        y = TimeDistributed(Conv2D(32, 8, (4, 4), activation='relu'))(inputB)
        y = TimeDistributed(MaxPooling2D())(y)
        y = TimeDistributed(Conv2D(64, 1, (2, 2), activation='relu'))(y)
        y = TimeDistributed(MaxPooling2D())(y)
        y = TimeDistributed(Flatten())(y)
        y = LSTM(128)(y)
        y = models.Model(inputs=inputB, outputs=y)

        z = TimeDistributed(Conv2D(32, 8, (4, 4), activation='relu'))(inputC)
        z = TimeDistributed(MaxPooling2D())(z)
        z = TimeDistributed(Conv2D(64, 1, (2, 2), activation='relu'))(z)
        z = TimeDistributed(MaxPooling2D())(z)
        z = TimeDistributed(Flatten())(z)
        z = LSTM(128)(z)
        z = models.Model(inputs=inputC, outputs=z)

        combined = Concatenate()([x.output, y.output, z.output])
        combined_input = [x.input, y.input, z.input]

        keya = Dense(action_space[0])(combined)
        mousea = Dense(action_space[1])(combined)
        key_a = models.Model(combined_input, keya, name="key_a")
        mouse_a = models.Model(combined_input, mousea, name="mouse_a")
        key_a_values = key_a(combined_input)
        mouse_a_values = mouse_a(combined_input)

        keyb = Dense(action_space[0])(combined)
        mouseb = Dense(action_space[1])(combined)
        key_b = models.Model(combined_input, keyb, name="key_b")
        mouse_b = models.Model(combined_input, mouseb, name="mouse_b")
        key_b_values = key_b(combined_input)
        mouse_b_values = mouse_b(combined_input)

        keyc = Dense(action_space[0])(combined)
        mousec = Dense(action_space[1])(combined)
        key_c = models.Model(combined_input, keyc, name="key_c")
        mouse_c = models.Model(combined_input, mousec, name="mouse_c")
        key_c_values = key_c(combined_input)
        mouse_c_values = mouse_c(combined_input)

        self.model = models.Model(inputs=combined_input, outputs=[[key_a_values, mouse_a_values], [key_b_values, mouse_b_values], [key_c_values, mouse_c_values]], name="full_conv_net")
        self.model.compile(loss="mean_squared_error",
                           optimizer=optimizers.RMSprop(lr=0.00025,
                                                        rho=0.95,
                                                        epsilon=0.01),
                           metrics=["accuracy"])

        self.model.summary()

