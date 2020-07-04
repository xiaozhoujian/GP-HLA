from utils import read_data_set
from gensim.models import Word2Vec
import numpy as np
from utils import AA_IDX
from models import Embedding
import torch
from models import wide_resnet
from neural_tangents import stax
import neural_tangents as nt
import time
from examples import util
from jax import random
from jax.api import jit
from jax.experimental import optimizers
from jax.experimental.stax import logsoftmax
from jax.api import grad
import datasets
import jax.experimental.stax as ostax

import utils

def infinite_fcn(train_embedding, test_embedding, data_set):
    _, _, kernel_fn = stax.serial(
        stax.Dense(64, 2., 0.05),
        stax.Relu(),
        stax.Dense(32, 2., 0.05),
        stax.Relu(),
        stax.Dense(4, 2., 0.05),
        stax.Relu(),
    )
    # 0 for no batching, whole batch
    kernel_fn = nt.batch(kernel_fn,
                         device_count=0,
                         batch_size=0)
    start = time.time()
    data_set['Y_train']= np.argmax(data_set['Y_train'],-1)[:,np.newaxis]
    data_set['Y_test']= np.argmax(data_set['Y_test'],-1)[:,np.newaxis]
    # Bayesian and infinite-time gradient descent inference with infinite network.
    #for i in range(10):
    predict_fn = \
            nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_embedding, data_set['Y_train'],
                                                     diag_reg_absolute_scale=True, learning_rate=1, diag_reg=1e-3) #1e0 1e-3


    nngp_mean, nngp_covariance = predict_fn(x_test=test_embedding, get='nngp',
                                            compute_cov=True)

    #fx_test_nngp.block_until_ready()
    #fx_test_ntk.block_until_ready()

    duration = time.time() - start
    print('Kernel construction and inference done in %s seconds.' % duration)

    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    utils.print_summary('NNGP test', data_set['Y_test'], nngp_mean, None, loss,nngp_covariance)
    #util.print_summary('NTK test', data_set['Y_test'], ntk_mean, None, loss)


def weight_space(train_embedding, test_embedding, data_set):
    init_fn, f, _ = stax.serial(
        stax.Dense(512, 1., 0.05),
        stax.Erf(),
        # 2 denotes 2 type of classes
        stax.Dense(2, 1., 0.05))

    key = random.PRNGKey(0)
    # (-1, 135),  135 denotes the feature length, here is 9 * 15 = 135
    _, params = init_fn(key, (-1, 135))

    # Linearize the network about its initial parameters.
    f_lin = nt.linearize(f, params)

    # Create and initialize an optimizer for both f and f_lin.
    opt_init, opt_apply, get_params = optimizers.momentum(1.0, 0.9)
    opt_apply = jit(opt_apply)

    state = opt_init(params)
    state_lin = opt_init(params)

    # Create a cross-entropy loss function.
    loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)

    # Specialize the loss function to compute gradients for both linearized and
    # full networks.
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # Train the network.
    print('Training.')
    print('Epoch\tLoss\tLinearized Loss')
    print('------------------------------------------')

    epoch = 0
    # Use whole batch
    batch_size = 64
    train_epochs = 10
    steps_per_epoch = 100

    for i, (x, y) in enumerate(datasets.mini_batch(train_embedding, data_set['Y_train'], batch_size, train_epochs)):
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, x, y), state)

        params_lin = get_params(state_lin)
        state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)

        if i % steps_per_epoch == 0:
            print('{}\t{:.4f}\t{:.4f}'.format(
                epoch, loss(f(params, x), y), loss(f_lin(params_lin, x), y)))
            epoch += 1
        if i / steps_per_epoch == train_epochs:
            break

    # Print out summary data comparing the linear / nonlinear model.
    x, y = train_embedding[:10000], data_set['Y_train'][:10000]
    util.print_summary('train', y, f(params, x), f_lin(params_lin, x), loss)
    util.print_summary(
        'test', data_set['Y_test'], f(params, test_embedding), f_lin(params_lin, test_embedding), loss)


def infinite_resnet(train_embedding, test_embedding, data_set):
    _, _, kernel_fn = wide_resnet(block_size=4, k=1, num_classes=2)
    kernel_fn = nt.batch(kernel_fn, device_count=0, batch_size=0)
    fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn, train_embedding, data_set['Y_train'], test_embedding,
                                                        get=('nngp', 'ntk'), diag_reg=1e-3)
    fx_test_nngp.block_until_ready()
    fx_test_ntk.block_until_ready()

    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    util.print_summary('NNGP test', data_set['Y_test'], fx_test_nngp, None, loss)
    util.print_summary('NTK test', data_set['Y_test'], fx_test_ntk, None, loss)


def train(params, files):
    data_set, peptide_n_mer = read_data_set(files, test_size=0.05)
    print(data_set['X_train'].shape)
    print(data_set['X_test'].shape)
    # variable batch size depending on number of data points
    batch_size = int(np.ceil(len(data_set['X_train']) / 100.0))
    epochs = int(params['epochs'])
    nb_filter = int(params['filter_size'])
    filter_length = int(params['filter_length'])
    dropout = float(params['dropout'])
    lr = float(params['lr'])

    # manual drop last
    for name in data_set.keys():
        if data_set[name].shape[0] % batch_size != 0:
            data_set[name] = data_set[name][:-(data_set[name].shape[0] % batch_size)]

    # load in learned distributed representation HLA-Vec
    hla_vec_obj = Word2Vec.load(files['vector_embedding'])
    hla_vec_embed = hla_vec_obj.wv
    embed_shape = hla_vec_embed.syn0.shape
    embedding_weights = np.random.rand(embed_shape[0] + 1, embed_shape[1])
    for key in AA_IDX.keys():
        embedding_weights[AA_IDX[key], :] = hla_vec_embed[key]
        embedded_dim = embed_shape[1]

    embedding = Embedding(embedded_dim, embedding_weights)
    train_embedding = embedding(torch.from_numpy(data_set['X_train'])).numpy()
    train_embedding = train_embedding.reshape((train_embedding.shape[0], -1))
    test_embedding = embedding(torch.from_numpy(data_set['X_test'])).numpy()
    test_embedding = test_embedding.reshape((test_embedding.shape[0], -1))
    # weight_space(train_embedding, test_embedding, data_set)
    infinite_fcn(train_embedding, test_embedding, data_set)
    #infinite_resnet(train_embedding, test_embedding, data_set)

