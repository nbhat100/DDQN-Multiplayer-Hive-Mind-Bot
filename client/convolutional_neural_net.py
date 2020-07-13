from tensorflow.keras import layers, models, optimizers


class ConvNet:
    def __init__(self, input_shape, action_space):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32,
                                     8,
                                     strides=(4, 4),
                                     padding="valid",
                                     activation="relu",
                                     input_shape=input_shape,
                                     data_format="channels_first"))
        self.model.add(layers.Conv2D(64,
                                     4,
                                     strides=(2, 2),
                                     padding="valid",
                                     activation="relu",
                                     input_shape=input_shape,
                                     data_format="channels_first"))
        self.model.add(layers.Conv2D(64,
                                     3,
                                     strides=(1, 1),
                                     padding="valid",
                                     activation="relu",
                                     input_shape=input_shape,
                                     data_format="channels_first"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=optimizers.RMSprop(lr=0.00025,
                                                        rho=0.95,
                                                        epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()
