from tensorflow.keras import layers, models, optimizers

def build_lenet(input_shape=(256, 922, 1), num_classes=5):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(6, (5, 5), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
