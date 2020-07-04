from neural_tangents import stax
import tensorflow_datasets as tfds
import neural_tangents as nt
import numpy as np
import time


def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = np.reshape(x, (x.shape[0], -1))
    return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(name, n_train=None, n_test=None, permute_train=False,
                do_flatten_and_normalize=True):

    """Download, parse and process a dataset to unit scale and one-hot labels."""
    ds_builder = tfds.builder(name)

    ds_train, ds_test = tfds.as_numpy(
        tfds.load(
            name + ':3.*.*',
            split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                   'test' + ('[:%d]' % n_test if n_test is not None else '')],
            batch_size=-1,
            as_dataset_kwargs={'shuffle_files': False}))

    train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                            ds_train['label'],
                                                            ds_test['image'],
                                                            ds_test['label'])

    if do_flatten_and_normalize:
        train_images = _partial_flatten_and_normalize(train_images)
        test_images = _partial_flatten_and_normalize(test_images)

    num_classes = ds_builder.info.features['label'].num_classes
    train_labels = _one_hot(train_labels, num_classes)
    test_labels = _one_hot(test_labels, num_classes)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def _accuracy(y, y_hat):
    """Compute the accuracy of the predictions with respect to one-hot labels."""
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def print_summary(name, labels, net_p, lin_p, loss):
    """Print summary information comparing a network with its linearization."""
    print('\nEvaluating Network on {} data.'.format(name))
    print('---------------------------------------')
    print('Network Accuracy = {}'.format(_accuracy(net_p, labels)))
    print('Network Loss = {}'.format(loss(net_p, labels)))
    if lin_p is not None:
        print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
        print('Linearization Loss = {}'.format(loss(lin_p, labels)))
        print('RMSE of predictions: {}'.format(np.sqrt(np.mean((net_p - lin_p) ** 2))))
    print('---------------------------------------')


def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum())


def WideResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
      blocks += [WideResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def WideResnet(block_size, k, num_classes):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        WideResnetGroup(block_size, int(16 * k)),
        WideResnetGroup(block_size, int(32 * k), (2, 2)),
        WideResnetGroup(block_size, int(64 * k), (2, 2)),
        stax.AvgPool((8, 8)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.))


def main():
    train_size = 1000
    test_size = 1000
    batch_size = 0

    init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=1, num_classes=10)
    x_train, y_train, x_test, y_test = get_dataset('cifar10', train_size, test_size, do_flatten_and_normalize=False)
    kernel_fn = nt.batch(kernel_fn,
                         device_count=0,
                         batch_size=batch_size)

    start = time.time()
    # Bayesian and infinite-time gradient descent inference with infinite network.
    fx_test_nngp, fx_test_ntk = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, x_test,
                                                        get=('nngp', 'ntk'), diag_reg=1e-3)
    fx_test_nngp.block_until_ready()
    fx_test_ntk.block_until_ready()
    duration = time.time() - start
    print('Kernel construction and inference done in %s seconds.' % duration)

    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
    print_summary('NTK test', y_test, fx_test_ntk, None, loss)


if __name__ == '__main__':
    main()
