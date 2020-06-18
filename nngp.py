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


def train(params, files):
    data_set, peptide_n_mer = read_data_set(files)

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
    # manual drop last
    # if train_embedding.shape[0] % batch_size != 0:
    #     train_embedding = train_embedding[:-(train_embedding.shape[0] % batch_size)]
    test_embedding = embedding(torch.from_numpy(data_set['X_test'])).numpy()
    test_embedding = test_embedding.reshape((test_embedding.shape[0], -1))
    # if test_embedding.shape[0] % batch_size != 0:
    #     test_embedding = test_embedding[:-(test_embedding.shape[0] % batch_size)]
    _, _, kernel_fn = stax.serial(
        stax.Dense(1, 2., 0.05),
        stax.Relu(),
        stax.Dense(1, 2., 0.05)
    )
    kernel_fn = nt.batch(kernel_fn,
                         device_count=0,
                         batch_size=846)
    start = time.time()
    # Bayesian and infinite-time gradient descent inference with infinite network.
    fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn, train_embedding, data_set['Y_train'], test_embedding,
                                                        get=('nngp', 'ntk'), diag_reg=1e-3)
    fx_test_nngp.block_until_ready()
    fx_test_ntk.block_until_ready()

    duration = time.time() - start
    print('Kernel construction and inference done in %s seconds.' % duration)

    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    util.print_summary('NNGP test', data_set['Y_test'], fx_test_nngp, None, loss)
    util.print_summary('NTK test', data_set['Y_test'], fx_test_ntk, None, loss)
    # init_fn, apply_fn, kernel_fn = wide_resnet(block_size=4, k=1, num_classes=1)
    # key1, key2 = random.split(random.PRNGKey(1))
    # x1 = random.normal(key1, (10, 100))
    # x2 = random.normal(key2, (20, 100))
    #
    # kernel = kernel_fn(x1, x2, 'nngp')

