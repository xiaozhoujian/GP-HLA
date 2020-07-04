from jax import random
import numpy as np


def mini_batch(x_train, y_train, batch_size, train_epochs):
    # epoch = 0
    start = 0
    key = random.PRNGKey(0)

    while True:
        end = start + batch_size

        if end > x_train.shape[0]:
            key, split = random.split(key)
            permutation = random.shuffle(split, np.arange(x_train.shape[0], dtype=np.int64))
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            # epoch += 1
            start = 0
            # print(epoch)
            continue
        yield x_train[start:end], y_train[start:end]
        start = start + batch_size
