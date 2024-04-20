import os
import time
from tensorflow import keras
from keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt


def get_model():
    model = keras.Sequential([
        Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    if os.path.exists("model.weights.h5"):
        model.load_weights('model.weights.h5')
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train_cat = keras.utils.to_categorical(y_train)

        x_train = x_train / 255

        # 4 8 16 16 - 0.44 (100 эпох)
        # 8 16 16 16 - 0.54 (100 эпох)
        # 8 16 32 32 - 0.52 (100 эпох)
        # 16 32 16 16 - 0.40 (100 эпох)
        # 16 32 32 32 - 0.57 (100 эпох)
        # 32 64 32 32 - 0.60 (100 эпох)
        # 32 64 64 64 - 0.62 (100 эпох)
        # 64 128 64 64 - 0.66 (50 эпох)
        # 32 64 64 64 64 - 0.61 (50 эпох)
        # 32 64 128 128 - 0.65 (50 эпох)
        # 32 64 256 256 - 0.68 (50 эпох)
        # 32 64 256 256 - 0.69 (50 эпх батч = 1000)
        # 32 64 512 512 - 0.69 (50 эпох)
        # 64 128 512 512 - 0.71 (50 эпох)
        # 64 128 1024 1024 - 0.71 (50 эпох батч = 5000)
        # 128 256 256 256 - 0.71 (50 эпх батч = 1000)

        # ИТОГ: 64 128 256 256

        start_time = time.time()
        history = model.fit(x_train, y_train_cat, batch_size=10000, epochs=10, validation_split=0.2)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Нейросеть обучилась за ", execution_time, "секунд.")

        model.save_weights('model.weights.h5')

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.grid(True)
        plt.show()

    return model
