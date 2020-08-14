from gensim.models import Word2Vec
import os
import random
from utils import str2bool
import pandas as pd


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


def filter_source(source_path, target_path):
    all_data = pd.read_csv(source_path, header=[0, 1], low_memory=False)
    # header = all_data.columns.values.tolist()
    # cols = tuple(zip(header, all_data.iloc[0]))
    #
    # new_header = pd.MultiIndex.from_tuples(cols, names=['Lvl_1', 'Lvl_2'])
    # all_data.drop([0], inplace=True)
    # all_data.columns = new_header

    # all_data.head()
    # print(all_data)
    # filtered_data = all_data.filter(items=[])
    filtered_data = all_data.loc[(all_data['Epitope']['Object Type'] == "Linear peptide") &
                                 (all_data['Assay']['Units'] == "nM") &
                                 (all_data['MHC']['MHC allele class'] == 'I')]
    filtered_data = filtered_data[[('Epitope', 'Description'),
                                   ('MHC', 'Allele Name'),
                                   ('Assay', 'Quantitative measurement')]]
    filtered_data.columns = ['peptide', 'HLA', 'IC50']
    # print(filtered_data)
    filtered_data.to_csv(target_path, index=False, sep='\t')


if __name__ == '__main__':
    source_path = "/Users/jojen/Downloads/mhc_ligand_full.csv"
    target_path = "/Users/jojen/Downloads/mhc_ligand_full_multi_file/mhc_class1.csv"
    filter_source(source_path, target_path)
