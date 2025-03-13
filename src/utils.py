import numpy as np

def sample_neufeld_data(N, p=10, a=0.5, b=5, sig=5):
    # Data generation
    sig = 5
    X = np.random.normal(0, 1, (N, p))
    eps = np.random.normal(0, sig, N)

    mu = b * (X[:, 0] <= 0) * \
        (1 + a * (X[:, 1] > 0) + ((X[:, 1] * X[:, 2]) > 0))

    y = mu + eps

    return X, y
