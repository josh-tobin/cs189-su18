import matplotlib.pyplot as plt
import numpy as np

NUM_DIMENSIONS = 4

def polynomial(values, coeffs):
    # Coeffs are assumed to be in order 0, 1, ..., n-1
    expanded = np.column_stack([coeffs[i] * (values ** i) for i in range(0, len(coeffs))])
    return np.sum(expanded, axis=-1)

def plot_polynomial(coeffs, x_range=[-1, 1], color='red', label='polynomial'):
    values = np.linspace(x_range[0], x_range[1], 1000)
    poly = polynomial(values, coeffs)
    plt.plot(values, poly, color=color, linewidth=2, label=label)

def polynomial_data(coeffs, n_data=100, x_range=[-1, 1], eps=0.1):
    x = np.random.uniform(x_range[0], x_range[1], n_data)
    poly = polynomial(x, coeffs)
    return x.reshape([-1, 1]), np.reshape(poly + eps * np.random.randn(n_data), [-1, 1])

def plot_linear(coeffs, x_range=[-1, 1], color='red'):
    assert list(coeffs.shape) == [1, 1]
    x = np.linspace(x_range[0], x_range[1], 1000)
    plt.plot(x, coeffs[0, 0] * x, color=color, linewidth=2)

def polynomial_features(x, degree):
    return np.column_stack([x ** i for i in range(degree + 1)])

def least_squares(x, y):
    w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    err = np.mean((x.dot(w) - y)**2)
    return w, err

def plot_errors(errors, label='train err', color='b', figsize=(8, 4)):
    plt.plot(errors, color=color, label=label)
    plt.semilogy()
    plt.legend()
    plt.title(f"Minimum error achieved at hyperparam value {np.argmin(errors)}")
    plt.xticks(np.arange(len(errors)))
