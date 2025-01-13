import time
from NN_model import *
from keras.utils import plot_model


# This file shows how to train a customized NN models.

def exec_NN_fit(x_model, x_objective, prem_upper, prem_lower, risk_load,
                 subs_coef, co_alpha, shuffle=0, if_price_bound=1, output_type=0, method='penalty', optimizer='rmsprop'):
    """
    This function shows how to train a customized NN models illustrated in the paper.\n
    :param x_model: A string chosen from ('NN1L2n', 'NN1L8n', 'NN1L64n', 'NN2L16n8n', 'NN2L64n8n', 'NN2L64n16n', 'NN3L64n16n8n', 'NN3L64n16n16n', 'NN3L64n64n16n', 'NN4L64n16n8n8n', 'NN4L64n32n16n8n', 'NN4L64n64n16n16n','Linear1', 'Linear5', 'Quad5', 'Cubic5', 'NN5', 'Linear72').\n
    :param x_objective: A string chosen from 'utility' and 'VaR'.x_objective='utility' is a utility maximization model. x_objective='VaR' is a tail risk minimization model.\n
    :param prem_upper: pper bound of insurance accepted.\n
    :param prem_lower: Lower bound of insurance accepted.\n
    :param risk_load: Risk loading of insurance pricing.\n
    :param sub_scoef: Percentage of insurance premium to be paid by policyholder.\n
    :param co_alpha: Coefficient as in exponential utility function.\n
    :param if_price_bound: 1(default) if policyholder has strict bounds for price, otherwise 0. If \(if_price_bound=1\), then penalty method will be used.
    :param output_type: 0(default) if to output performance metrics, 1 if to output predictions.\n
    :param shuffle: 0(default) if data are splitted by temporal order, 1 if data are splitted by random.\n
    :param optimizer:adam or rmsprop\n
    :return: return a performance metrics if output_type=0, and return the predictions when output_type=1.
    """
    data = fun_data_process(x_model, shuffle)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_valid, Y_valid = data['X_valid'], data['Y_valid']
    X_test, Y_test = data['X_test'], data['Y_test']

    model = NNmodel(x_model=x_model, x_objective=x_objective, 
                    prem_upper=prem_upper, prem_lower=prem_lower,
                    risk_load=risk_load, subs_coef=subs_coef,
                    co_alpha=co_alpha, fit_method=method, optimizer=optimizer)
    init_weal = model.init_weal
    model = fit_NN_model(model, X_train, Y_train, if_price_bound, method=method)
    model.save_weights('./fitted_model.weights.h5')
    # model.save('./fitted_model.keras')
    #plot_model(model, to_file='model.png', show_shapes=True)

    if output_type == 0:
        result_metrics = fun_result_summary(model, x_model, init_weal, risk_load, subs_coef, co_alpha)
        time.sleep(0.1)
        return result_metrics
    elif output_type == 1:
        train_predictions = model.predict(X_train, verbose=0)
        valid_predictions = model.predict(X_valid, verbose=0)
        test_predictions = model.predict(X_test, verbose=0)

        plot_predictions(x_model,Y_train.to_numpy(),train_predictions.reshape(-1),Y_valid.to_numpy(), valid_predictions.reshape(-1), Y_test.to_numpy(), test_predictions.reshape(-1))
        

        return {'training':train_predictions, 'validation':valid_predictions, 'test':test_predictions}

    
if __name__ == "__main__":
    method = 'genlagrange'# penalty,genlagrange
    optimizer = 'adam'# rmsprop,adam
    result_metric = exec_NN_fit('Linear72', 'utility', prem_upper=500, prem_lower=5,
                risk_load=1.2133, subs_coef=1, co_alpha=0.008, output_type=1, method=method, optimizer=optimizer)
    print(result_metric)
    # NN3L64n16n16n(lambda=1.2428) Linear72，lambda=1.2133(Linear72)
    if method == 'genlagrange':
        result_metric.to_csv(f"result_genl{optimizer}.csv",index=True)
    elif method == 'penalty':
        result_metric.to_csv(f"result_penalty{optimizer}.csv",index=True)