import tensorflow as tf
import numpy as np
import sys
import time

def main():
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<num threads>")
        sys.exit(0)
    tf.config.threading.set_intra_op_parallelism_threads(int(sys.argv[1]))
    tf.config.threading.set_inter_op_parallelism_threads(min(int(sys.argv[1]), 4))
    train_x = np.load("train_images.npy").astype(np.single)
    train_y = np.load("train_labels.npy")
    N = 6000
    H = 28
    W = 28
    train_x = train_x[:N, :]
    train_x = train_x.reshape((N, H, W, 1))
    train_y = train_y[:N].ravel()

    x = tf.convert_to_tensor(train_x)
    y = tf.convert_to_tensor(train_y)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(3, 3, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(8, 3, strides=(2, 2), data_format='channels_last'),
        tf.keras.layers.Flatten('channels_last'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])
    start = time.time()
    model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    end = time.time()
    model.fit(x, y, batch_size=N, epochs=100, shuffle=False, verbose=2)
    print("Compile Time: ", str(end-start) + "s")
if __name__ == "__main__":
    main()