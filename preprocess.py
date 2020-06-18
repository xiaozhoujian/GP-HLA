from gensim.models import Word2Vec
import os
import random
from utils import str2bool


def sequence2vector(params, files):
    sentences = []
    train_set = files['train_set']
    with open(train_set) as f:
        f.readline()
        for line in f:
            sentences.append(list(line.strip().split('\t')[0]))
    random.shuffle(sentences)

    print("There are {} peptides sequences in the {}".format(len(sentences), train_set))

    model = Word2Vec(sentences, min_count=int(params['min_count']), size=int(params['vec_dim']),
                     window=int(params['window_size']), sg=str2bool(params['sg_model']),
                     iter=int(params['iter']), batch_words=int(params['batch_words']))

    model.save(files['vector_embedding'])

