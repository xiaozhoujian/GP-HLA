import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats


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
                                                                                  test_size=0.05, random_state=2)

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
    test_target = test_target.to_numpy()
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
    target_matrix = target_matrix.to_numpy()

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




# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A set of utility operations for running examples.
"""

thread = 0.5


def _accuracy(y, y_hat,variance):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  binary_y = (y-variance > thread)[:, 0].astype(int)
  equal = (binary_y == y_hat)
  all_zero = np.mean(1 == y_hat)
  print('zero_variance: ' + str(round(np.mean(equal*variance), 10)))
  print('one_variance: ' + str(round(np.mean((1-equal)*variance), 10)))
  return np.mean(binary_y== y_hat)

def aucs(y, y_hat, variance):
    binary_y = np.clip(y ,0,1)
    mean_fpr, mean_tpr, mean_thresholds = roc_curve(y_hat, binary_y, pos_label=1)
    mean_auc = auc(mean_fpr, mean_tpr)
    rho, pValue = stats.spearmanr(y_hat, binary_y)
    print('SRCC: ' + str(round(rho, 3)))
    print('AUC: ' + str(round(mean_auc,3)))



def print_summary(name, labels, net_p, lin_p, loss,variance):
  """Print summary information comparing a network with its linearization."""
  net_p = np.array(net_p)
  variance = np.diag(np.array(variance))
  #lin_p = np.array(lin_p)

  print('\nEvaluating Network on {} data.'.format(name))
  print('---------------------------------------')
  print('Network Accuracy = {}'.format(_accuracy(net_p, labels,variance)))
  aucs(net_p, labels,variance)
  print('Network Loss = {}'.format(loss(net_p, labels)))
  # if lin_p is not None:
  #   print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
  #   print('Linearization Loss = {}'.format(loss(lin_p, labels)))
  #   print('RMSE of predictions: {}'.format(
  #       np.sqrt(np.mean((net_p - lin_p) ** 2))))
  # print('---------------------------------------')