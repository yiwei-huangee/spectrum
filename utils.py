from scipy.io import savemat, loadmat
import numpy as np
import os
from args import args
# from tensorflow.keras.utils import to_categorical

# %% Data loading (using true arrival time)
def load_train_data(data_directory='/root/WorkSpace/project/spectrum_two_stage/database/'):
    """Load data from .mat file
    return: X_train, Y_train, sample_length
    """
    # init 3D matrix
    # X_train_all = np.zeros((train_sample,args.N,1024))
    # Y_train_all = np.zeros((train_sample,1638))
    
    mat = loadmat(data_directory + 'train_1.mat')
    # mat = loadmat(data_directory + 'train_len_512.mat')
    # X_train = mat['X_train_all'] / 100. -0.5 # normalization
    X_train = mat['X_train_all']
    
    # X_train = X_train / 7472 #min-max normalization
    # z-score normalization
    # for i in range(X_train.shape[0]):
    #     mu = np.mean(X_train[i,:])
    #     sigma = np.std(X_train[i,:])
    #     for j in range(X_train.shape[1]):
    #         X_train[i,j] = (X_train[i,j] - mu) / sigma
    # Y_train = to_categorical(mat['Y_train_all'], num_classes=num_classes)
    Y_train = mat['energy_label_all']
    sample_length = X_train.shape[-1]  # frame length
    return X_train, Y_train,sample_length


def load_test_data(j: int,k: int, data_directory='/root/WorkSpace/project/spectrum_two_stage/database/datatest/', num_classes=6):
    """Load data from .mat file
    k: 10, 20, 30, 40, 50, 60
    returns: X_test, Y_test, Y_test_i
    """
    mat = loadmat(data_directory + str(j)+ '_data_test_' + str(k) +'.mat')
    # mat = loadmat(data_directory + 'data_test_' + str(k)+'_len_512' +'.mat')
    X_test = mat['X_test'] / 100. -0.5
    Y_test = mat['Y_test']
    # Y_test = to_categorical(Y_test_i, num_classes=num_classes)
    # Y_test = mat['Y_test'].astype(bool).astype(np.uint8)
    return X_test, Y_test


def save_test_data(k: int, Y_pred, data_directory='/root/WorkSpace/project/activity/database/'):
    """Save test data to .mat file"""
    file_name = os.getcwd() + '\\data_pred_' + str(k) + '.mat'
    savemat(file_name, {'Y_pred': Y_pred})

