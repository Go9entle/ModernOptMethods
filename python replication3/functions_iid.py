import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import time
from keras import layers, regularizers, optimizers, callbacks


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
                    "Y_valid": Y_valid, "X_train": X_valid,
                    "Y_test": Y_test, "X_test": X_test}
    return output


'''
if __name__=="__main__":
    data = fun_data_process('NN3L64n16n16n', shuffle=0)
    print(data.keys())
'''


def fun_para_values(premupper, premlower, riskload, subscoef=1, coalpha=0.888):
    """
    This function loads parameter values (for model fitting).\n
    :param premupper: pper bound of insurance accepted.\n
    :param premlower: Lower bound of insurance accepted.\n
    :param riskload: Risk loading of insurance pricing.\n
    :param subscoef: Percentage of insurance premium to be paid by policyholder.\n
    :param coalpha: Coefficient as in exponential utility function.\n
    :return: A list of exogenous parameters used to fit the model. 
    """
    init_weal_0 = 388.6 # Wealth at the begining of period
    farm_cost = 504 # Farming cost for one period of time
    revenue_max = 861 # Maximum revenue to possibly gain
    global init_weal
    init_weal = init_weal_0 - farm_cost + revenue_max 
    global prem_upper
    prem_upper = premupper
    global prem_buff
    prem_buff = premupper-premlower
    global risk_load
    risk_load = riskload
    global penalty_coef
    penalty_coef = 10**(-6) # Starting value of penalty coefficient. This value may need adjustment under different applications
    global subs_coef
    subs_coef = subscoef  # Percentage paid by policyholder
    global co_alpha
    co_alpha = coalpha
    return {
        "init_weal": init_weal,
        "prem": prem_upper,
        "prem_buff": prem_buff,
        "risk_load": risk_load,
        "penalty_coef": penalty_coef,
        "subs_coef": subs_coef,
        "co_alpha": co_alpha
    }
    

def tilted_loss(y_true, y_pred):
    """
    This function defines a customized loss function, based on exponential utility function, with absolute risk aversion of "co_alpha".\n
    :param y_true: True observations of response variable.\n
    :param y_pred: Predicted values of response from an NN model.
    """
    temp2 = risk_load * tf.reduce_mean(y_pred, axis=1)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    Uw = -tf.exp(-co_alpha * w) / co_alpha
    temp1 = -tf.reduce_mean(Uw, axis=1)
    temp3 = tf.maximum(temp2 - prem_upper, 0)
    temp4 = tf.square(temp3)
    temp5 = tf.minimum(temp2 - prem_buff, 0)
    temp6 = tf.square(temp5)
    temp7 = temp1 + penalty_coef * temp4 + penalty_coef * temp6
    return temp7


def tilted_loss_2(y_true, y_pred):
    """
    This function defines a customized loss function, based on Value at Risk 95%.\n
    :param y_true: True observations of response variable.\n
    :param y_pred: Predicted values of response from an NN model.
    """
    var_level = 0.05
    phi_inverse = stats.norm.ppf(var_level)
    
    temp2 = risk_load * tf.reduce_mean(y_pred, axis=1)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    temp10 = tf.reduce_mean(w, axis=1)
    temp11 = tf.reduce_mean(tf.pow(w - temp10, 3), axis=1) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2), axis=1), 3/2)
    temp12 = tf.reduce_mean(tf.pow(w - temp10, 4), axis=1) / tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2), axis=1), 2) - 3
    temp13 = phi_inverse + (phi_inverse ** 2 - 1) * temp11 / 6 + (phi_inverse ** 3 - 3 * phi_inverse) * temp12 / 24 - (2 * phi_inverse ** 3 - 5 * phi_inverse) * tf.pow(temp11, 2) / 36
    temp22 = temp10 + temp13 * tf.pow(tf.reduce_mean(tf.pow(w - temp10, 2), axis=1), 1/2)
    temp1 = -temp22
    temp14 = tf.reduce_mean(tf.maximum(y_pred - y_true - 100, 0), axis=1) * 1e3  # To ensure indemnity cannot exceed loss
    temp3 = tf.maximum(temp2 - prem_upper, 0)
    temp4 = tf.square(temp3)
    temp5 = tf.minimum(temp2 - prem_buff, 0)
    temp6 = tf.square(temp5)
    temp7 = temp1 + penalty_coef * temp4 + penalty_coef * temp6 + penalty_coef * temp14
    return temp7


@tf.function
def metric_utility(y_true, y_pred):
    """
    This function defines a customized metric function, based on exponential utility function, with absolute risk aversion of "co_alpha".\n
    :param y_true: True observations of response variable.\n
    :param y_pred: Predicted values of response from an NN model.
    """
    temp2 = risk_load * tf.reduce_mean(y_pred, axis=1)
    w = init_weal - y_true + y_pred - subs_coef * temp2
    Uw = -tf.exp(-co_alpha * w) / co_alpha
    temp1 = -tf.reduce_mean(Uw, axis=1)
    return temp1


@tf.function
def metric_penalty(y_true, y_pred):
    """
    This function measures the penalty to a certain NN model, if its corresponding insurance premium is out of the accepted range of policyholder.\n
    :param y_true: True observations of response variable.\n
    :param y_pred: Predicted values of response from an NN model.
    """
    temp1 = risk_load * tf.reduce_mean(y_pred, axis=1)
    temp2 = tf.maximum(temp1 - prem_upper, 0)
    temp3 = tf.square(temp2)
    temp4 = tf.minimum(temp1 - prem_buff, 0)
    temp5 = tf.square(temp4)
    temp6 = temp3 + temp5  # Penalty term
    return temp6


@tf.function
def metric_premium(y_true, y_pred):
    """
    This function estimates the corresponding insurance premium of a certain NN model.\n
    :param y_true: True observations of response variable.\n
    :param y_pred: Predicted values of response from an NN model.
    """
    temp1 = risk_load * tf.reduce_mean(y_pred, axis=1)
    return temp1


def fun_NNstructure(input_layer, x_model):
    """
    This function specifies the output layer by its model structure.\n
    :param input_layer: A generic input layer.\n
    :param x_model: A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
    'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.
    """
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


def build_model(x_model,x_objective, ncol):
    """
    This function builds an NN model by its model structure.\n
    :param x_model:  A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
    'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.\n
    :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
    :param ncol: How many columns the covariates have. 
    """
    input_layer = layers.Input(shape=(ncol,))

    output_layer = fun_NNstructure(input_layer, x_model)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    if x_objective == 'utility':
        model.compile(
            loss=tilted_loss,  
            optimizer=optimizers.RMSprop(),
            metrics=[metric_utility, metric_penalty, metric_premium]  
        )
    elif x_objective == 'VaR':
        model.compile(
            loss=tilted_loss_2,  
            optimizer=optimizers.RMSprop(),
            metrics=[metric_utility, metric_penalty, metric_premium]  
        )

    return model


def prep_fitting(x_model, x_objective, ncol):
    """
    This function prepares for fitting the model.\n
    :param x_model:  A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
    'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.\n
    :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
    :param ncol: How many columns the covariates have.
    """
    model = build_model(x_model, x_objective, ncol)
    callbacks.ModelCheckpoint(filepath='./temp_model_callback.h5', monitor='val_loss',
                                verbose=0, save_best_only=True)
    return model


def fit_NN_model(if_price_bound, x_model, x_objective, model, X, y):
    """
    This function is to fit NN model with data.\n
    :param if_price_bound: An indicator. Value = 1 if policyholder has strict bounds for price, otherwise value=0. If value = 1, then penalty method will be used.\n
    :param x_model: A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.\n
    :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
    :param model: A tensorflow object which is to be fitted.\n
    :param X:X_train\n
    :param y:y_train.
    """
    # define callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20)
    print_dot_callback = callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, log: print("\n.") if epoch%80==0 else None
    )
    # train the model
    X_train, Y_train = X, y
    model.fit(X_train, Y_train, epochs=100, validation_split=0.2, 
                verbose=0, callbacks=[early_stop, print_dot_callback])
    model.save_weights('./temp_model.weights.h5')
    if if_price_bound == 0: None
    elif if_price_bound == 1:
        # 使用罚函数法训练模型
        (loss, metric_utility, metric_penalty, metric_premium) =\
             model.evaluate(X_train, Y_train, batchsize=X_train.shape[0], verbose=0)
        # When corresponding premium is out of acceptance, apply the penalty method using the following while loop
        # Ensure while loop can start
        metric_premium_last = metric_premium - 1
        while ((metric_premium > prem_upper) | (metric_premium < prem_upper-prem_buff) | (abs((metric_premium-metric_premium_last)/metric_premium_last)>0.01)) & (penalty_coef < 10^(-1)):
            metric_premium_last = metric_premium
            penalty_coef *= 5
            model = build_model(x_model, x_objective, ncol=X_train.shape[1])
            model.load_weights('./temp_model.weights.h5')
            model.fit(X_train, Y_train, epochs=100, validation_split=0.2,
                        verbose=0, callbacks=[early_stop, print_dot_callback])
            model.save_weights('./temp_model.weights.h5')
            (loss, metric_utility, metric_penalty, metric_premium) =\
             model.evaluate(X_train, Y_train, batchsize=X_train.shape[0], verbose=0)
            time.sleep(0.1)
            print(f"Panelty coefficient:{penalty_coef:.2f}")
            print(f"Mean utility:{metric_utility:.2f}")
            print(f"Premium penalty:{metric_penalty:.2f}")
            print(f"Premium amount:{metric_premium:.2f}")
    return model


def  fun_result_summary(model, x_model, data):
    """
    This function is to evaluate performance of an NN model.\n
    :param model: A tensorflow object whose performance to be evaluated.\n
    :param x_model:  A string chosen from c('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 
    'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n', 'Linear1', 'Linear5', 'Quad5', 'Cubic5', 'Linear72'). It specifies the number of layers and the numbers of neurons in each layer in the model.\n
    :param data: The data used to fit before and its validation and training set.  
    """
    data = fun_data_process(x_model, shuffle=0)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_valid, Y_valid = data['X_valid'], data['Y_valid']
    X_test, Y_test = data['X_test'], data['Y_test']

    # Training set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_train, Y_train, batchsize=X_train.shape[0], verbose=0)
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
    w_w_ins_train = init_weal - Y_train + train_predictions - subs_coef * metric_premium
    w_wo_ins_train = init_weal - Y_train
    sd_w_ins_train = np.std(w_w_ins_train)
    sd_wo_ins_train = np.std(w_wo_ins_train)
    sd_reduction_train = 1 - sd_w_ins_train/sd_wo_ins_train

    var5_w_ins_train = np.quantile(w_w_ins_train, 0.05)
    var5_wo_ins_train = np.quantile(w_wo_ins_train, 0.05)
    var5_improve_train = var5_w_ins_train - var5_wo_ins_train

    # Validation set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_valid, Y_valid, batchsize=X_valid.shape[0], verbose=0)
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
    w_w_ins_valid = init_weal - Y_valid + valid_predictions - subs_coef * metric_premium
    w_wo_ins_valid = init_weal - Y_valid
    sd_w_ins_valid = np.std(w_w_ins_valid)
    sd_wo_ins_valid = np.std(w_wo_ins_valid)
    sd_reduction_valid = 1 - sd_w_ins_valid/sd_wo_ins_valid

    var5_w_ins_valid = np.quantile(w_w_ins_valid, 0.05)
    var5_wo_ins_valid = np.quantile(w_wo_ins_valid, 0.05)
    var5_improve_valid = var5_w_ins_valid - var5_wo_ins_valid

    # Test set
    (_ , metric_utility, metric_penalty, metric_premium) = \
        model.evaluate(X_test, Y_test, batchsize=X_test.shape[0], verbose=0)
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
    w_w_ins_test = init_weal - Y_test + test_predictions - subs_coef * metric_premium
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