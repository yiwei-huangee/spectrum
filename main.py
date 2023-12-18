import numpy as np
import pandas as pd
import random
import tensorflow as tf
from utils import load_train_data
from args import args
from model import model_test
from MultiResUnet import UNet
from DenseInceptionUnet import Dense_Inception_UNet
# %% Loss functions
def MSE_KL_loss(y_true, y_pred,weight=1.0):
    """MSE + KL loss"""
    # MSE loss
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    # KL loss
    kl_loss = tf.keras.losses.KLDivergence()(y_true, y_pred)
    return mse_loss + weight*kl_loss

def getrandomIndex(n,x):
    train_idx = random.sample(range(n),x)
    val_idx = list(set(range(n))-set(train_idx))
    return train_idx,val_idx

# %% Main function
def main():
    if args.run_mode == 'train':
        print('Start training')
        X_train, Y_train, length = load_train_data()
        train_idx, val_idx = getrandomIndex(X_train.shape[0],int(X_train.shape[0] * 0.8))
        Xtrain = X_train[train_idx]
        Xval = X_train[val_idx]
        ytrain= Y_train[train_idx]
        yval = Y_train[val_idx]
        train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
        train_dataset = train_dataset.shuffle(buffer_size=123).batch(128)
        val_dataset = tf.data.Dataset.from_tensor_slices((Xval, yval))
        val_dataset = val_dataset.shuffle(buffer_size=123).batch(128)



        model_name = 'MultiResUNet'  # UNet or UNetPP
        model_depth = 5  # Number of Level in the CNN Model
        model_width = 64  # Width of the Initial Layer, subsequent layers start from here
        kernel_size = 3  # Size of the Kernels/Filter
        num_channel = 1  # Number of Channels in the Model
        D_S = 0  # Turn on Deep Supervision
        A_E = 1  # Turn on AutoEncoder Mode for Feature Extraction
        A_G = 1  # Turn on for Guided Attention
        LSTM = 1  # Turn on for LSTM, Implemented for UNet and MultiResUNet only
        num_dense_loop = 2
        problem_type = 'Regression'
        output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
        is_transconv = True # True: Transposed Convolution, False: UpSampling
        '''Only required if the AutoEncoder Mode is turned on'''
        feature_number = 1024  # Number of Features to be Extracted
        alpha = 1  # Model Width Expansion Parameter, for MultiResUNet only
        LR = 0.0005
        #

        tf.keras.backend.clear_session()
        # Model = UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
        #              ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, alpha=alpha, is_transconv=is_transconv).MultiResUNet()
        

        Model = Dense_Inception_UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type,
                                 output_nums=output_nums, num_dense_loop =num_dense_loop, ds=D_S, ae=A_E, ag=A_G).Dense_Inception_UNet()

        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.MeanAbsoluteError())
        # callback_stp = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                             min_delta = 1,
        #                                             patience=20,
        #                                             verbose=1,
        #                                             restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)

        Model.fit(train_dataset, epochs=300, validation_data=val_dataset, callbacks=[callback_reduce_lr])#, callbacks=[callback_stp, callback_reduce_lr]
        Model.save_weights('/root/WorkSpace/project/spectrum_two_stage/results/model.h5')
        # Model.summary()

    if args.run_mode == 'test':
        print('Start testing')
        
if __name__ == '__main__':
    main()