import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict


AA_IDX = OrderedDict([
    ('A', 1),
    ('C', 2),
    ('E', 3),
    ('D', 4),
    ('G', 5),
    ('F', 6),
    ('I', 7),
    ('H', 8),
    ('K', 9),
    ('M', 10),
    ('L', 11),
    ('N', 12),
    ('Q', 13),
    ('P', 14),
    ('S', 15),
    ('R', 16),
    ('T', 17),
    ('W', 18),
    ('V', 19),
    ('Y', 20)
])


def str2bool(word):
    if word.lower() == 'true':
        return True
    else:
        return False


def read_data_set(files, test_size=0.5):
    test_peptides, test_targets, peptide_length, allele = get_test_matrix(files['test_set'])
    seq_matrix, target_matrix = get_train_matrix(files['train_set'], allele, peptide_length)
    train_peptides, test_peptides, train_targets, test_targets = train_test_split(seq_matrix, target_matrix,
                                                                                  test_size=0.05, random_state=1)

    # map the training peptide sequences to their integer index
    feature_matrix = np.empty((0, peptide_length), dtype=int)
    for index in range(len(train_peptides)):
        feature_matrix = np.append(feature_matrix, [sequence2int(train_peptides.iloc[index])], axis=0)

    # map the test peptide sequences to their integer index
    test_matrix = np.empty((0, peptide_length), int)
    for num in range(len(test_peptides)):
        test_matrix = np.append(test_matrix, [sequence2int(test_peptides.iloc[num])], axis=0)

    # create training and test data_set
    data_set = dict()
    data_set['X_train'] = feature_matrix
    data_set['Y_train'] = train_targets
    data_set['X_test'] = test_matrix
    data_set['Y_test'] = test_targets
    return data_set, peptide_length


def get_test_matrix(test_file):
    test_data = pd.read_csv(test_file, delim_whitespace=True)
    allele = test_data['Allele'][0]
    peptide_length = len(test_data['Peptide_seq'][0])
    measurement_type = test_data['Measurement_type'][0]
    # the first dimension is 1 represent IC50 < 500
    # test_category = np.zeros((len(test_data), 2))

    if measurement_type.lower() == 'binary':
        test_data['Measurement_value'] = np.where(test_data.Measurement_value == 1.0, 1, 0)
    else:
        test_data['Measurement_value'] = np.where(test_data.Measurement_value < 500.0, 1, 0)
    test_peptide = test_data.Peptide_seq
    test_target = test_data.Measurement_value
    test_target = test_target.as_matrix()
    # for i in range(test_target.shape[0]):
    #     if test_target[i] == 1:
    #         test_category[i][0] = 1
    #     else:
    #         test_category[i][1] = 0

    return test_peptide, test_target, peptide_length, allele


def get_train_matrix(train_file, allele, peptide_length):
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=0)
    train_data.columns = ['sequence', 'HLA', 'target']

    # build training matrix
    peptide_data = train_data[train_data.HLA == allele]
    peptide_data = peptide_data[peptide_data['sequence'].map(len) == peptide_length]

    # remove any peptide with  unknown variables
    filtered_x_data = peptide_data[~peptide_data.sequence.str.contains('X')]
    filtered_xb_data = filtered_x_data[~filtered_x_data.sequence.str.contains('B')]

    # remap target values to 1's and 0's
    filtered_xb_data['target'] = np.where(filtered_xb_data.target == 1, 1, 0)

    seq_matrix = filtered_xb_data.sequence
    target_matrix = filtered_xb_data.target
    target_matrix = target_matrix.as_matrix()

    category_matrix = np.zeros((target_matrix.shape[0], 2))
    for i in range(target_matrix.shape[0]):
        if target_matrix[i] == 1:
            category_matrix[i][0] = 1
        else:
            category_matrix[i][1] = 1
    return seq_matrix, category_matrix


def sequence2int(peptide_sequence):
    """
    Convert the amino acid sequence to int number
    :param peptide_sequence: string, the peptide sequence with a fixed length of amino acid
    :return: np.array, peptide array with int to represent the amino acid
    """
    peptide_array = []
    for amino_acid in peptide_sequence:
        peptide_array.append(AA_IDX[amino_acid])
    return np.asarray(peptide_array)
