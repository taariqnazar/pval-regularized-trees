import numpy as np 

def computed_mse(true_y, predicte_y):
    return np.mean((true_y - predicte_y)**2)

def in_and_out_sample_mse(X, y, predictor, split=0.8):
    n = len(y)
    n_train = int(split * n)
    train_X, train_y = X[:n_train], y[:n_train]
    test_X, test_y = X[n_train:], y[n_train:]

    predictor.fit(train_X, train_y)
    return computed_mse(train_X, train_y, predictor), computed_mse(test_X, test_y, predictor)


