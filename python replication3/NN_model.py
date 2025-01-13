import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import time
from tensorflow.keras import layers, regularizers, optimizers, callbacks, losses
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

print(f"tensorflow version:{tf.__version__}")

def fun_data_process(NNmodel, shuffle=0):
    """
    This function loads and processes data.\n
    :param NNmodel: A string chosen from ('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n','Linear1', 'Linear5', 'Quad5', 'Cubic5', 'NN5', 'Linear72').\n
    :param shuffle: An indicator scaler. Value=0 (default) if data are splitted by temporal order, value=1 if data are splitted by random shuffle.\n
    :return: A list of data sample to fit the model, as ["Y_train", "X_train", "Y_valid", "X_valid", "Y_test", "X_test"].
    """
    if NNmodel == 'Linear1':
        Data_IL = pd.read_csv('./Data/Data_Linear1.csv')
    elif NNmodel == 'Linear5':
        Data_IL = pd.read_csv('./Data/Data_Linear5.csv')
    elif NNmodel == 'NN5':
        Data_IL = pd.read_csv('./Data/Data_Linear5.csv')
    elif NNmodel == 'Quad5':
        Data_IL = pd.read_csv('./Data/Data_quad5.csv')
    elif NNmodel == 'Cubic5':
        Data_IL = pd.read_csv('./Data/Data_cubic5.csv')
    else:
        Data_IL = pd.read_csv('./Data/Data_weathersample.csv')

    "Split data"
    if shuffle == 0:
        data_train = Data_IL[Data_IL['year'] >= 1925][Data_IL['year'] <= 1991]
        data_valid = Data_IL[Data_IL['year'] > 1991][Data_IL['year'] <= 2003]
        data_test = Data_IL[Data_IL['year'] > 2003]
    else:
        Data_IL = Data_IL.sample(frac = 1, random_state = 99999)
        data_train, data_test, data_valid = Data_IL[0:700], Data_IL[700:850], Data_IL[850:1000]
    
    "Normalize training data"

    scalar_train = StandardScaler().fit(data_train.iloc[:, 2:])
    X_train = scalar_train.transform(data_train.iloc[:, 2:])
    X_valid = scalar_train.transform(data_valid.iloc[:, 2:])
    X_test = scalar_train.transform(data_test.iloc[:, 2:])

    Y_train = data_train['Loss']
    Y_valid = data_valid['Loss']
    Y_test = data_test['Loss']

    output = {"Y_train": Y_train, "X_train": X_train,
                    "Y_valid": Y_valid, "X_valid": X_valid,
                    "Y_test": Y_test, "X_test": X_test}
    return output


class NNmodel(keras.Model):
    """
    This class defines the NNmodel in the paper.
    :param x_model: A string chosen from ('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n','Linear1', 'Linear5', 'Quad5', 'Cubic5', 'NN5', 'Linear72').\n
    :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
    :param prem_upper: pper bound of insurance accepted.\n
    :param prem_lower: Lower bound of insurance accepted.\n
    :param risk_load: Risk loading of insurance pricing.\n
    :param sub_scoef: Percentage of insurance premium to be paid by policyholder.\n
    :param co_alpha: Coefficient as in exponential utility function.\n
    :param fit_method: Method to solve the problem, penalty or genlagrange.
    """
    def __init__(self, x_model, x_objective, prem_upper, prem_lower, risk_load, subs_coef, co_alpha, fit_method='penalty', optimizer='rmsprop'):
        """
        First create a NNmodel instance. 
        :param x_model: A string chosen from ('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n','Linear1', 'Linear5', 'Quad5', 'Cubic5', 'NN5', 'Linear72').\n
        :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
        :param premupper: pper bound of insurance accepted.\n
        :param prem_lower: Lower bound of insurance accepted.\n
        :param risk_load: Risk loading of insurance pricing.\n
        :param sub_scoef: Percentage of insurance premium to be paid by policyholder.\n
        :param co_alpha: Coefficient as in exponential utility function.\n
        :param fit_method: Method to solve the problem, penalty or genlagrange.\n
        :param optimizer:adam or rmsprop
        """
        super(NNmodel, self).__init__()
        self.x_model = x_model
        self.x_objective = x_objective
        init_weal_0 = 388.6 # Wealth at the begining of period
        farm_cost = 504 # Farming cost for one period of time
        revenue_max = 861 # Maximum revenue to possibly gain
        self.init_weal = init_weal_0 - farm_cost + revenue_max 
        self.prem_upper = prem_upper
        self.prem_buff = prem_upper - prem_lower
        self.risk_load = risk_load
        self.penalty_coef = 10**(-6) # Starting value of penalty coefficient. This value may need adjustment under different applications
        self.subs_coef = subs_coef #percentage paid by policyholders
        self.co_alpha = co_alpha
        self.fit_method = fit_method
        self.optimizer = optimizer
        if self.fit_method == 'penalty':
            self.penalty_coef = 10**(-6) # Starting value of penalty coefficient. This value may need adjustment under different applications
        elif self.fit_method == 'genlagrange':
            # Initialize the param in generalized Lagrange method.
            self.lambda1 = 2
            self.lambda2 = 2
            self.sigma = 0.001
        

    def _fun_NNstructure(self, input_layer):
        """
        This function specifies the output layer by its model structure.\n
        :param input_layer: A generic input layer.\n
        :param x_model: A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
        'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.
        """
        x_model = self.x_model
        if x_model in ['Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72']:
            output_layer = layers.Dense(1, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.001))(input_layer)
        elif x_model == 'NN1L2n':
            output_layer = layers.Dense(2, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN1L8n':
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN1L64n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN2L16n8n':
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN2L64n8n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN2L64n16n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN3L64n16n8n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN3L64n16n16n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model in ['NN3L64n64n16n', 'NN1', 'NN5']:
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN4L64n16n8n8n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN4L64n32n16n8n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(32, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(8, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        elif x_model == 'NN4L64n64n16n16n':
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(input_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(64, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(16, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
            output_layer = layers.Dropout(0.01)(output_layer)
            output_layer = layers.Dense(1, activation='relu', 
                                    kernel_regularizer=regularizers.l2(0.001))(output_layer)
        return output_layer    

    def call(self):
        """
        This function builds a NN model by its model structure.
        """
        # First construct the inputlayer.
        if self.x_model == 'Linear1': 
            input_layer = keras.Input(shape=(1,))
        elif self.x_model in ['Linear5', 'Quad5', 'Cubic5']:
            input_layer = keras.Input(shape=(5,))
        else:
            input_layer = keras.Input(shape=(72,))


        # Second define the outputlayer.
        output_layer = self._fun_NNstructure(input_layer)
        # Third define the model.
        model = keras.Model(inputs=input_layer, outputs = output_layer)
        #model = output_layer
        
        if self.fit_method == 'penalty':
            # Finally compile the model.
            if self.x_objective == 'utility':
                model.compile(
                    loss=partial(utilityloss_penalty, risk_load=self.risk_load, init_weal=self.init_weal, 
                                            subs_coef=self.subs_coef, co_alpha=self.co_alpha, 
                                            prem_upper=self.prem_upper, prem_buff=self.prem_buff, 
                                            penalty_coef=self.penalty_coef),
                    optimizer=self.optimizer,
                    metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                            self.subs_coef, self.co_alpha),
                            Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                            Metric_premium(self.risk_load)]
                )
                '''
            model.compile(
                loss=tilted_loss(self.risk_load, self.init_weal, 
                                 self.subs_coef, self.co_alpha, 
                                 self.prem_upper, self.prem_buff,
                                 self.penalty_coef),
                optimizer='rmsprop',
                metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                        self.subs_coef, self.co_alpha),
                        Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                        Metric_premium(self.risk_load)]
            )
            '''
            elif self.x_objective == 'VaR':
                model.compile(
                    loss=partial(varloss_penalty, var_level=0.05, risk_load=self.risk_load, 
                                init_weal=self.init_weal, subs_coef=self.subs_coef, 
                                prem_upper=self.prem_upper, prem_buff=self.prem_buff,
                                penalty_coef=self.penalty_coef),
                    optimizer=self.optimizer,
                    metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                            self.subs_coef, self.co_alpha),
                            Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                            Metric_premium(self.risk_load)]
                )
                '''
            model.compile(
                loss=tilted_loss_2(var_level=0.05, risk_load=self.risk_load,
                                   init_weal=self.init_weal, subs_coef=self.subs_coef,
                                   prem_upper=self.prem_upper, prem_buff=self.prem_buff,
                                   penalty_coef=self.penalty_coef),
                optimizer='rmsprop',
                metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                        self.subs_coef, self.co_alpha),
                        Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                        Metric_premium(self.risk_load)]
            )
            '''
            else: raise ValueError("x_objective must be 'utility' or 'VaR'.")
        elif self.fit_method == 'genlagrange':
            # Finally compile the model.
            if self.x_objective == 'utility':
                model.compile(
                    loss=partial(utilityloss_genlagrange, risk_load=self.risk_load, init_weal=self.init_weal, 
                                            subs_coef=self.subs_coef, co_alpha=self.co_alpha, 
                                            prem_upper=self.prem_upper, prem_buff=self.prem_buff, 
                                            lambda1 = self.lambda1, lambda2 = self.lambda2,
                                            sigma = self.sigma),
                    optimizer=self.optimizer,
                    metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                            self.subs_coef, self.co_alpha),
                            Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                            Metric_premium(self.risk_load)]
                )
            elif self.x_objective == 'VaR':
                model.compile(
                    loss=partial(varloss_genlagrange, var_level=0.05, risk_load=self.risk_load, 
                                init_weal=self.init_weal, subs_coef=self.subs_coef, 
                                prem_upper=self.prem_upper, prem_buff=self.prem_buff,
                                lambda1 = self.lambda1, lambda2 = self.lambda2,
                                sigma = self.sigma),
                    optimizer=self.optimizer,
                    metrics=[Metric_utility(self.risk_load, self.init_weal, 
                                            self.subs_coef, self.co_alpha),
                            Metric_penalty(self.risk_load, self.prem_upper, self.prem_buff),
                            Metric_premium(self.risk_load)]
                )
            else: raise ValueError("x_objective must be 'utility' or 'VaR'.")
        return model

def utilityloss_penalty(y_true, y_pred, risk_load, init_weal, subs_coef, co_alpha, prem_upper, prem_buff, penalty_coef):
    temp2 = risk_load * tf.reduce_mean(y_pred)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    Uw = -tf.exp(-co_alpha * w) / co_alpha
    temp1 = -tf.reduce_mean(Uw)
    temp3 = tf.maximum(temp2 - prem_upper, 0)
    temp4 = tf.square(temp3)
    temp5 = tf.minimum(temp2 - prem_upper + prem_buff, 0)
    temp6 = tf.square(temp5)
    temp7 = temp1 + penalty_coef * temp4 + penalty_coef * temp6
    return temp7

def utilityloss_genlagrange(y_true, y_pred, risk_load, init_weal, subs_coef, co_alpha, prem_upper, prem_buff, lambda1, lambda2, sigma):
    """
    增广Lagrange法的损失函数
    """
    # 效用部分
    temp2 = risk_load * tf.reduce_mean(y_pred)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    Uw = -tf.exp(-co_alpha * w) / co_alpha
    temp1 = -tf.reduce_mean(Uw)
    # 增广lagrange部分
    temp3 = tf.where(temp2 - prem_upper + prem_buff >= lambda1/sigma,
                    -lambda1**2 / (2 * sigma),
                    0.5 * sigma * (- lambda1**2 / sigma**2 + tf.square(temp2 - prem_upper + prem_buff - lambda1/sigma)))
    temp4 = tf.where(prem_upper - temp2 >= lambda2/sigma,
                    -lambda2**2 / (2 * sigma),
                    0.5 * sigma * (- lambda2**2 / sigma**2 + tf.square(prem_upper - temp2 -lambda2/sigma)))
    
    return temp1 + temp3 + temp4


def varloss_penalty(y_true, y_pred, var_level, risk_load, init_weal, subs_coef, prem_upper, prem_buff, penalty_coef):
    phi_inverse = stats.norm.pdf(var_level)
    temp2 = risk_load * tf.reduce_mean(y_pred)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    temp10 = tf.reduce_mean(w)
    temp11 = tf.reduce_mean(tf.pow(w - temp10, 3)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 3/2)
    temp12 = tf.reduce_mean(tf.pow(w - temp10, 4)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 2) - 3
    temp13 = phi_inverse + (phi_inverse ** 2 - 1) * temp11 / 6\
              + (phi_inverse ** 3 - 3 * phi_inverse) * temp12 / 24\
             - (2 * phi_inverse ** 3 - 5 * phi_inverse) * tf.pow(temp11, 2) / 36
    temp22 = temp10 + temp13 * tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 1/2)
    temp1 = -temp22
    temp14 = tf.reduce_mean(tf.maximum(y_pred - y_true - 100, 0)) * 1e3  # To ensure indemnity cannot exceed loss
    temp3 = tf.maximum(temp2 - prem_upper, 0)
    temp4 = tf.square(temp3)
    temp5 = tf.minimum(temp2 - prem_upper + prem_buff, 0)
    temp6 = tf.square(temp5)
    temp7 = temp1 + penalty_coef * temp4 + penalty_coef * temp6 + penalty_coef * temp14
    return temp7

def varloss_genlagrange(y_true, y_pred, var_level, risk_load, init_weal, subs_coef, prem_upper, prem_buff, lambda1, lambda2, sigma):
    #先计算VaR
    phi_inverse = stats.norm.pdf(var_level)
    temp2 = risk_load * tf.reduce_mean(y_pred)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    temp10 = tf.reduce_mean(w)
    temp11 = tf.reduce_mean(tf.pow(w - temp10, 3)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 3/2)
    temp12 = tf.reduce_mean(tf.pow(w - temp10, 4)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 2) - 3
    temp13 = phi_inverse + (phi_inverse ** 2 - 1) * temp11 / 6\
              + (phi_inverse ** 3 - 3 * phi_inverse) * temp12 / 24\
             - (2 * phi_inverse ** 3 - 5 * phi_inverse) * tf.pow(temp11, 2) / 36
    temp22 = temp10 + temp13 * tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 1/2)
    temp1 = -temp22
    # 增广lagrange部分
    temp3 = tf.where(temp2 - prem_upper + prem_buff >= lambda1/sigma,
                    -lambda1**2 / (2 * sigma),
                    0.5 * sigma * (- lambda1**2 / sigma**2 + tf.square(temp2 - prem_upper + prem_buff - lambda1/sigma)))
    temp4 = tf.where(prem_upper - temp2 >= lambda2/sigma,
                    -lambda2**2 / (2 * sigma),
                    0.5 * sigma * (- lambda2**2 / sigma**2 + tf.square(prem_upper - temp2 - lambda2/sigma)))
    
    return temp1 + temp3 + temp4

class tilted_loss(losses.Loss):
    """
    This class define a customized loss based on exponential utility function, with absolute risk aversion of "co_alpha".
    """
    def __init__(self, risk_load, init_weal, subs_coef, co_alpha, prem_upper, prem_buff, penalty_coef, name='tilted_loss'):
        super().__init__(name=name)
        self.risk_load = risk_load  
        self.init_weal = init_weal
        self.subs_coef = subs_coef
        self.co_alpha = co_alpha
        self.prem_upper = prem_upper
        self.prem_buff = prem_buff
        self.penalty_coef = penalty_coef
    
    def call(self, y_true, y_pred):
        """
        This function will conduct once a tilted_loss instance is built.\n
        And this function gives the tilted_loss function.
        """
        temp2 = self.risk_load * tf.reduce_mean(y_pred)
        
        w = self.init_weal - y_true + y_pred - self.subs_coef * temp2
        Uw = -tf.exp(-self.co_alpha * w) / self.co_alpha
        temp1 = -tf.reduce_mean(Uw)
        temp3 = tf.maximum(temp2 - self.prem_upper, 0)
        temp4 = tf.square(temp3)
        temp5 = tf.minimum(temp2 - self.prem_upper + self.prem_buff, 0)
        temp6 = tf.square(temp5)
        temp7 = temp1 + self.penalty_coef * temp4 + self.penalty_coef * temp6
        return temp7
    

class tilted_loss_2(losses.Loss):
    """
    This function defines a customized loss function based on Value at Risk 95%.
    """
    def __init__(self, var_level, risk_load, init_weal, subs_coef, prem_upper, prem_buff, penalty_coef, name='tilted_loss2'):
        super().__init__(name=name)
        self.var_level = var_level
        self.risk_load = risk_load
        self.init_weal = init_weal
        self.subs_coef = subs_coef
        self.prem_upper = prem_upper
        self.prem_buff = prem_buff
        self.penalty_coef = penalty_coef

    def call(self, y_true, y_pred):
        """
        This function will conduct once a tilted_loss instance is built.\n
        And this function gives the tilted_loss2 function.
        """
        phi_inverse = stats.norm.ppf(self.var_level)
    
        temp2 = self.risk_load * tf.reduce_mean(y_pred)
        w = self.init_weal - y_true + y_pred - self.subs_coef * temp2
        temp10 = tf.reduce_mean(w)
        temp11 = tf.reduce_mean(tf.pow(w - temp10, 3)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 3/2)
        temp12 = tf.reduce_mean(tf.pow(w - temp10, 4)) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 2) - 3
        temp13 = phi_inverse + (phi_inverse ** 2 - 1) * temp11 / 6\
              + (phi_inverse ** 3 - 3 * phi_inverse) * temp12 / 24\
             - (2 * phi_inverse ** 3 - 5 * phi_inverse) * tf.pow(temp11, 2) / 36
        temp22 = temp10 + temp13 * tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2)), 1/2)
        temp1 = -temp22
        temp14 = tf.reduce_mean(tf.maximum(y_pred - y_true - 100, 0)) * 1e3  # To ensure indemnity cannot exceed loss
        temp3 = tf.maximum(temp2 - self.prem_upper, 0)
        temp4 = tf.square(temp3)
        temp5 = tf.minimum(temp2 - self.prem_upper + self.prem_buff, 0)
        temp6 = tf.square(temp5)
        temp7 = temp1 + self.penalty_coef * temp4 + self.penalty_coef * temp6 + self.penalty_coef * temp14
        return temp7 
            

class Metric_utility(keras.metrics.Metric):
    """
    This class defines a customized metric class, based on exponential utility function,
    with absolute risk aversion of co_alpha.
    """
    def __init__(self, risk_load, init_weal, subs_coef, co_alpha, name='metric_utility',  **kwargs):
        super(Metric_utility, self).__init__(name=name, **kwargs)
        self.risk_load = risk_load
        self.init_weal = init_weal
        self.subs_coef = subs_coef
        self.co_alpha = co_alpha
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        temp2 = tf.reduce_mean(self.risk_load * y_pred)
        w = self.init_weal - y_true + y_pred - self.subs_coef * temp2
        Uw = -tf.exp(-self.co_alpha * w) / self.co_alpha
        temp1 = -tf.reduce_mean(Uw)
        self.total.assign_add(tf.reduce_mean(temp1))
    
    def result(self):
        return self.total
    
    def reset_states(self):
        self.total.assign(0)


class Metric_penalty(keras.metrics.Metric):
    """
    This class measures the penalty to a certain NN model, if its corresponding insurance
    premium is out of the accepted range of policyholder.
    """
    def __init__(self, risk_load, prem_upper, prem_buff, name='metric_penalty', **kwargs):
        super(Metric_penalty, self).__init__(name=name, **kwargs)
        self.risk_load = risk_load
        self.prem_upper = prem_upper
        self.prem_buff = prem_buff

        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        temp0 = tf.reduce_mean(y_pred)
        self.total.assign_add(tf.reduce_mean(temp0))
    
    def result(self):
        temp1 = self.risk_load * self.total
        temp2 = tf.maximum(temp1 - self.prem_upper, 0)
        temp3 = tf.square(temp2)
        temp4 = tf.minimum(temp1 - self.prem_upper + self.prem_buff, 0)
        temp5 = tf.square(temp4)
        temp6 = temp3 + temp5 # Penalty term
        return temp6
    
    def reset_states(self):
        self.total.assign(0)
    

class Metric_premium(keras.metrics.Metric):
    """
    This class estimates the corresponding insurance premium of a certain NN model.
    """
    def __init__(self, risk_load, name='metric_premium', **kwargs):
        super(Metric_premium, self).__init__(name=name, **kwargs)
        self.risk_load = risk_load
        
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        temp1 = tf.reduce_mean(y_pred)
        self.total.assign_add(temp1)

    def result(self):
        return self.risk_load * self.total
    
    def reset_states(self):
        self.total.assign(0)

def fit_NN_model(model, X, y, if_price_bound, method='penalty'):
    """
    This function is to fit NN model with data.\n
    :param model: A tensorflow object which is to be fitted.\n
    :param X:the training data X_train.\n
    :param y:the training data y_train.\n
    :param if_price_bound: 1 if policyholder has strict bounds for price, otherwise 0.\n
    \t If \(if_price_bound=1\), then penalty method will be used.\n
    :param method: The method used to solve the restricted optimization problem. penalty means using the penalty method, genlagrange means using the generalized Lagragne method.
    """
    print(f"The optimizer is {model.optimizer}.")
    if method == 'penalty':
        return fit_NN_model_penalty(model, X, y, if_price_bound)
    elif method == 'genlagrange':
        return fit_NN_model_genlagrange(model, X, y, if_price_bound)


def fit_NN_model_penalty(model, X, y, if_price_bound):
    """
    This function is to fit NN model with data using penalty method.\n
    :param model: A tensorflow object which is to be fitted.\n
    :param X:the training data X_train.\n
    :param y:the training data y_train.\n
    :param if_price_bound: 1 if policyholder has strict bounds for price, otherwise 0.\n
    \t If \(if_price_bound=1\), then penalty method will be used.
    """
    print("We are using the penalty method.")
    # define callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print_dot_callback = callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, log: print("\n.") if epoch%80==0 else None
    )
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Initialize lists to store the values
    metric_utility_values = []  # Store metric_utility values
    metric_premium_values = []  # Store metric_premium values

    # train the model
    X_train, Y_train = tf.convert_to_tensor(X), tf.convert_to_tensor(y.values)
    
    modelfit = model.call()
    modelfit.fit(X_train, Y_train, epochs=100, validation_split=0.2, batch_size=X_train.shape[0], 
                verbose=0, callbacks=[early_stop, print_dot_callback], shuffle=True)
    modelfit.save_weights('./temp_model.weights.h5')
    if if_price_bound == 0: None
    elif if_price_bound == 1:
        # 使用罚函数法训练模型
        
        (_, metric_utility, metric_penalty, metric_premium) =\
             modelfit.evaluate(X_train, Y_train, batch_size=X_train.shape[0], verbose=0)
        
        metric_utility_values.append(-metric_utility)
        metric_premium_values.append(metric_premium)


        # When corresponding premium is out of acceptance, apply the penalty method using the following while loop
        # Ensure while loop can start
        metric_premium_last = metric_premium - 1
        prem_upper = model.prem_upper
        prem_buff = model.prem_buff
        print(f"Penalty coefficient:{model.penalty_coef:.2e}")
        print(f"Mean utility:{metric_utility:.2f}")
        print(f"Premium penalty:{metric_penalty:.2f}")
        print(f"Premium amount:{metric_premium:.2f}")
        print(f"The mean of y_pred:{tf.reduce_mean(modelfit.predict(X_train, verbose=0))}")
        
        while ((metric_premium > prem_upper) | (metric_premium < prem_upper-prem_buff) | (abs((metric_premium-metric_premium_last)/(metric_premium_last+1e-10))>0.01)) & (model.penalty_coef < 0.1):
            metric_premium_last = metric_premium
            
            #model.penalty_coef *= 5 #更新罚函数系数
            model.penalty_coef *= 1.2 #更新罚函数系数
            modelfit = model.call() #re compile the model with a new penalty_coef
            
            modelfit.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=X_train.shape[0],
                        verbose=0, callbacks=[early_stop, print_dot_callback])
            modelfit.load_weights('./temp_model.weights.h5')
            modelfit.fit(X_train, Y_train, epochs=100, validation_split=0.2,
                        verbose=0, callbacks=[early_stop, print_dot_callback])
            modelfit.save_weights('./temp_model.weights.h5')
            (_, metric_utility, metric_penalty, metric_premium) =\
             modelfit.evaluate(X_train, Y_train, batch_size=X_train.shape[0], verbose=0)
            
            metric_utility_values.append(-metric_utility)
            metric_premium_values.append(metric_premium)# 记录负效用函数

            time.sleep(0.1)
            print(f"Penalty coefficient:{model.penalty_coef:.2e}")
            print(f"Mean utility:{metric_utility:.2f}")
            print(f"Premium penalty:{metric_penalty:.2f}")
            print(f"Premium amount:{metric_premium:.2f}")
            print(f"The mean of y_pred:{tf.reduce_mean(modelfit.predict(X_train, verbose=0))}")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot utility values on the left y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Utility', color='r')
        ax1.plot(range(len(metric_utility_values)), metric_utility_values, label='Utility (with penalty)', color='r', marker='o')
        ax1.tick_params(axis='y', labelcolor='r')

        # Create a second y-axis for premium values
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Premium', color='gray')  
        ax2.plot(range(len(metric_premium_values)), metric_premium_values, label='Premium', color='gray', linestyle = '--')
        # ax2.axhline(prem_upper, color='b', linestyle='--', label=f'Upper Bound {prem_upper}')
        # ax2.axhline(prem_upper-prem_buff, color='b', linestyle='--', label=f'Lower Bound {prem_upper-prem_buff}')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Title and grid
        plt.title(f'Utility and Premium Changes During Training with Penalty using {model.optimizer}')
        ax1.grid(True)

        # Display the plot
        fig.tight_layout()

        # Combine the legends from both axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')
        plt.savefig(f'penalty{model.optimizer}.png')
    return modelfit


def fit_NN_model_genlagrange(model, X, y, if_price_bound):
    """
    This function is to fit NN model with data using generalized Lagrange method.\n
    :param model: A tensorflow object which is to be fitted.\n
    :param X:the training data X_train.\n
    :param y:the training data y_train.\n
    :param if_price_bound: 1 if policyholder has strict bounds for price, otherwise 0.\n
    \t If \(if_price_bound=1\), then penalty method will be used.
    """
    print("We are using generalized Lagrange")
    # define callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print_dot_callback = callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, log: print("\n.") if epoch%80==0 else None
    )
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # train the model
    X_train, Y_train = tf.convert_to_tensor(X), tf.convert_to_tensor(y.values)
    
    modelfit = model.call()
    modelfit.fit(X_train, Y_train, epochs=100, validation_split=0.2, batch_size=X_train.shape[0], 
                verbose=0, callbacks=[early_stop, print_dot_callback], shuffle=True)
    modelfit.save_weights('./temp_model.weights.h5')
    if if_price_bound == 0: None
    elif if_price_bound == 1:
        utilities, premiums = [], []
        # 使用增广Lagange法训练模型
    
        (_, metric_utility, metric_penalty, metric_premium) =\
             modelfit.evaluate(X_train, Y_train, batch_size=X_train.shape[0], verbose=0)
        # When corresponding premium is out of acceptance, apply the penalty method using the following while loop
        # Ensure while loop can start
        metric_premium_last = metric_premium - 1
        prem_upper = model.prem_upper
        prem_buff = model.prem_buff
        #print(f"Panelty coefficient:{model.penalty_coef:.2e}")
        print(f"Mean utility:{metric_utility:.2f}")
        print(f"Premium penalty:{metric_penalty:.2f}")
        print(f"Premium amount:{metric_premium:.2f}")
        print(f"The mean of y_pred:{tf.reduce_mean(modelfit.predict(X_train, verbose=0))}")
        epsilon = 10e-8
        while (tf.square(tf.minimum(metric_premium - prem_upper + prem_buff, model.lambda1/model.sigma))\
             + tf.square(tf.minimum(prem_upper - metric_premium, model.lambda2/model.sigma))  > epsilon**2):
            '''
            & ((metric_premium > prem_upper) | (metric_premium < prem_upper-prem_buff) |\
             (abs((metric_premium-metric_premium_last)/(metric_premium_last+1e-10))>0.01))
            '''
            
            #更新增广Lagrange法的参数
            model.lambda1 = - model.sigma * tf.minimum(metric_premium - prem_upper + prem_buff - model.lambda1/model.sigma, 0)
            model.lambda2 = - model.sigma * tf.minimum(prem_upper - metric_premium - model.lambda2/model.sigma, 0)
            model.sigma *= 1.2

            metric_premium_last = metric_premium
            modelfit = model.call() #re compile the model with a new penalty_coef
            
            modelfit.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=X_train.shape[0],
                        verbose=0, callbacks=[early_stop, print_dot_callback])
            modelfit.load_weights('./temp_model.weights.h5')
            modelfit.fit(X_train, Y_train, epochs=100, validation_split=0.2,
                        verbose=0, callbacks=[early_stop, print_dot_callback])
            modelfit.save_weights('./temp_model.weights.h5')
            (_, metric_utility, metric_penalty, metric_premium) =\
             modelfit.evaluate(X_train, Y_train, batch_size=X_train.shape[0], verbose=0)
            time.sleep(0.1)

            # Record metrics for plotting
            utilities.append(-metric_utility)
            premiums.append(metric_premium)

            #print(f"Panelty coefficient:{model.penalty_coef:.2e}")
            print(f"Mean utility:{metric_utility:.2f}")
            print(f"Premium penalty:{metric_penalty:.2f}")
            print(f"Premium amount:{metric_premium:.2f}")
            print(f"The mean of y_pred:{tf.reduce_mean(modelfit.predict(X_train, verbose=0))}")


        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot utility values on the left y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Utility', color='r')
        ax1.plot(range(len(utilities)), utilities, label='Utility (with genlagrange)', color='r', marker='o')
        ax1.tick_params(axis='y', labelcolor='r')

        # Create a second y-axis for premium values
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Premium', color='gray')  
        ax2.plot(range(len(premiums)), premiums, label='Premium', color='gray', linestyle = '--')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Title and grid
        plt.title(f'Utility and Premium Changes During Training with GenLagrange using {model.optimizer}')
        ax1.grid(True)

        # Display the plot
        fig.tight_layout()

        # Combine the legends from both axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')
        plt.savefig(f'genL{model.optimizer}.png')

    return modelfit


def plot_predictions(x_model,y_train, train_predictions, y_valid, valid_predictions, y_test, test_predictions):


    plt.figure(figsize=(8, 6))  
    sns.scatterplot(x=y_train, y=train_predictions, color='red', alpha=0.6, label='train')

    sns.scatterplot(x=y_valid, y=valid_predictions, color='green', alpha=0.6, label='valid')

    sns.scatterplot(x=y_test, y=test_predictions, color='blue', alpha=0.6, label='test')

    plt.xlabel("Loss")
    plt.ylabel("Indemnity Payoff")
    plt.title(f"A {x_model} contract")
    plt.grid(True)
    plt.legend(title="Groups")
    plt.savefig(f'prediction{x_model}.png')


def fun_result_summary(model, x_model, init_weal, risk_load, subs_coef, co_alpha):
    """
    This function is to evaluate performance of an NN model.\n
    :param model: A tensorflow object whose performance to be evaluated.\n
    :param x_model:  A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
    'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.\n 
    """
    data = fun_data_process(x_model, shuffle=0)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_valid, Y_valid = data['X_valid'], data['Y_valid']
    X_test, Y_test = data['X_test'], data['Y_test']
    X_train, Y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train.values)
    X_valid, Y_valid = tf.convert_to_tensor(X_valid), tf.convert_to_tensor(Y_valid.values)
    X_test, Y_test = tf.convert_to_tensor(X_test), tf.convert_to_tensor(Y_test.values)
    # Training set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_train, Y_train, batch_size=X_train.shape[0], verbose=0)
    train_utility = -metric_utility
    train_utility_wo = np.mean((-np.exp(-co_alpha*(init_weal-Y_train))/co_alpha))
    train_improve = -(train_utility - train_utility_wo) / train_utility_wo
    train_cew = -np.log(-co_alpha * train_utility) / co_alpha
    train_cew_wo = -np.log(-co_alpha * train_utility_wo) / co_alpha
    train_cew_improve = train_cew - train_cew_wo
    train_cew_improve_per = train_cew_improve/train_cew_wo
    
    train_premium = metric_premium
    train_coverage = train_premium/risk_load
    train_profit = train_premium - train_coverage

    train_predictions = model.predict(X_train)
    #train_payments = pd.DataFrame([])
    w_w_ins_train = init_weal - Y_train + train_predictions.reshape(-1) - subs_coef * metric_premium
    w_wo_ins_train = init_weal - Y_train
    sd_w_ins_train = np.std(w_w_ins_train)
    sd_wo_ins_train = np.std(w_wo_ins_train)
    sd_reduction_train = 1 - sd_w_ins_train/sd_wo_ins_train

    var5_w_ins_train = np.quantile(w_w_ins_train, 0.05)
    var5_wo_ins_train = np.quantile(w_wo_ins_train, 0.05)
    var5_improve_train = var5_w_ins_train - var5_wo_ins_train

    # Validation set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_valid, Y_valid, batch_size=X_valid.shape[0], verbose=0)
    valid_utility = -metric_utility
    valid_utility_wo = np.mean((-np.exp(-co_alpha*(init_weal-Y_valid))/co_alpha))
    valid_improve = -(valid_utility - valid_utility_wo) / valid_utility_wo
    valid_cew = -np.log(-co_alpha * valid_utility) / co_alpha
    valid_cew_wo = -np.log(-co_alpha * valid_utility_wo) / co_alpha
    valid_cew_improve = valid_cew - valid_cew_wo
    valid_cew_improve_per = valid_cew_improve/valid_cew_wo
    
    valid_premium = metric_premium
    valid_coverage = valid_premium/risk_load
    valid_profit = valid_premium - valid_coverage

    valid_predictions = model.predict(X_valid)
    #valid_payments = pd.DataFrame([])
    w_w_ins_valid = init_weal - Y_valid + valid_predictions.reshape(-1) - subs_coef * metric_premium
    w_wo_ins_valid = init_weal - Y_valid
    sd_w_ins_valid = np.std(w_w_ins_valid)
    sd_wo_ins_valid = np.std(w_wo_ins_valid)
    sd_reduction_valid = 1 - sd_w_ins_valid/sd_wo_ins_valid

    var5_w_ins_valid = np.quantile(w_w_ins_valid, 0.05)
    var5_wo_ins_valid = np.quantile(w_wo_ins_valid, 0.05)
    var5_improve_valid = var5_w_ins_valid - var5_wo_ins_valid

    # Test set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_test, Y_test, batch_size=X_test.shape[0], verbose=0)
    test_utility = -metric_utility
    test_utility_wo = np.mean((-np.exp(-co_alpha*(init_weal-Y_test))/co_alpha))
    test_improve = -(test_utility - test_utility_wo) / test_utility_wo
    test_cew = -np.log(-co_alpha * test_utility) / co_alpha
    test_cew_wo = -np.log(-co_alpha * test_utility_wo) / co_alpha
    test_cew_improve = test_cew - test_cew_wo
    test_cew_improve_per = test_cew_improve/test_cew_wo
    
    test_premium = metric_premium
    test_coverage = test_premium/risk_load
    test_profit = test_premium - test_coverage

    test_predictions = model.predict(X_test)
    #test_payments = pd.DataFrame([])
    w_w_ins_test = init_weal - Y_test + test_predictions.reshape(-1) - subs_coef * metric_premium
    w_wo_ins_test = init_weal - Y_test
    sd_w_ins_test = np.std(w_w_ins_test)
    sd_wo_ins_test = np.std(w_wo_ins_test)
    sd_reduction_test = 1 - sd_w_ins_test/sd_wo_ins_test

    var5_w_ins_test = np.quantile(w_w_ins_test, 0.05)
    var5_wo_ins_test = np.quantile(w_wo_ins_test, 0.05)
    var5_improve_test = var5_w_ins_test - var5_wo_ins_test


    result_metrics = {
        "U with insurance": [train_utility, valid_utility, test_utility], 
        "U w/o insurance": [train_utility_wo, valid_utility_wo, test_utility_wo],
        "U improvement": [train_improve, valid_improve, test_improve],
        "CEW with insurance":[train_cew, valid_cew, test_cew],
        "CEW w/o insurance":[train_cew_wo, valid_cew_wo, test_cew_wo],
        "CEW improvement": [train_cew_improve, valid_cew_improve, test_cew_improve],
        "CEW improvement (%)":[train_cew_improve_per, valid_cew_improve_per, test_cew_improve_per],
        "Premium": [train_premium, valid_premium, test_premium],
        "Coverage":[train_coverage, valid_coverage, test_coverage],
        "Insurer Profit": [train_profit, valid_profit, test_profit],
        "Std": [sd_w_ins_train, sd_w_ins_valid, sd_w_ins_test],
        "Std w/o insurance": [sd_wo_ins_train, sd_wo_ins_valid, sd_wo_ins_test],
        "Std reduction":[sd_reduction_train, sd_reduction_valid, sd_reduction_test],
        "VaR5%" :[var5_w_ins_train, var5_w_ins_valid, var5_w_ins_test],
        "VaR5% w/o insurance": [var5_wo_ins_train, var5_wo_ins_valid, var5_wo_ins_test],
        "VaR5% improvement":[var5_improve_train, var5_improve_valid, var5_improve_test]
    }

    result_metrics = pd.DataFrame(result_metrics, index=['train', 'valid', 'test']).T

    return result_metrics


if __name__ == '__main__':
    x_model = 'NN3L64n16n16n'
    x_objective = 'utility'
    init_weal = 388.6-504+861
    prem_upper = 500
    prem_lower = 5
    risk_load = 1.2
    subs_coef = 1
    co_alpha = 0.008
    model = NNmodel(x_model, x_objective, prem_upper, 
                    prem_lower, risk_load, subs_coef, co_alpha)
    print(type(model))
    print(model.init_weal)
    model.penalty_coef = 10
    print(model.penalty_coef)
    

    