from tensorflow.keras import layers, models, optimizers


class ConvNet:
    def __init__(self, input_shape, action_space):
        conv_input = layers.Input(shape=input_shape)
        x = layers.TimeDistributed(layers.Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape))(conv_input)
        x = layers.TimeDistributed(layers.Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(128)(x)
        x = layers.Dense(512, activation="relu")(x)
        key_conv_output = layers.Dense(action_space[0])(x)
        mouse_conv_output = layers.Dense(action_space[1])(x)
        key_conv_net = models.Model(conv_input, key_conv_output, name="key_conv_net")
        mouse_conv_net = models.Model(conv_input, mouse_conv_output, name="mouse_conv_net")
        key_q_values = key_conv_net(conv_input)
        mouse_q_values = mouse_conv_net(conv_input)
        self.model = models.Model(inputs=conv_input, outputs=[key_q_values, mouse_q_values], name="full_conv_net")
        self.model.compile(loss="mean_squared_error",
                           optimizer=optimizers.RMSprop(lr=0.00025,
                                                        rho=0.95,
                                                        epsilon=0.01),
                           metrics=["accuracy"])

        self.model.summary()

