import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp

from numpy import pi


# TODO: c ) Test your code with θ = 0.3, 0.5, 1, N = 2l − 1 and M = 4l with l = {2, 3, 4, 5, 6}. As before,
# study if those numerical schemes converge and report the convergence rates if they converge. Com-
# ment on your results


def basis(x, i, N):
    """
    Basis function
    :param x: point to evaluate
    :param i: index
    :param N: Discretization steps
    :return: b_i(x)
    """
    h = 1 / (N + 1)
    y = np.where(((i - 1) * h < x) & (x <= i * h), (x - (i - 1) * h) / h, 0)
    y = np.where((i * h < x) & (x < (i + 1) * h), 1 - (x - i * h) / h, y)
    return y


def bPrime(x, i, N):
    """
    Basis function derivative
    :param x: point to evaluate
    :param i: index
    :param N: Discretization steps
    :return: b_i(x)
    """
    h = 1 / (N + 1)
    y = np.where(((i - 1) * h < x) & (x <= i * h), 1 / h, 0)
    y = np.where((i * h < x) & (x < (i + 1) * h), -1 / h, y)
    return y


def simpson(a, b, F):
    return (b - a) / 6 * (F(a) + 4 * F((a + b) / 2) + F(b))


def integral(a, b, idx_i, idx_j, N, alpha_loc, beta_loc, gamma_loc):

    def F(x):
        sum_1 = alpha_loc(x) * bPrime(x, idx_i, N) * bPrime(x, idx_j, N)
        sum_2 = beta_loc(x) * bPrime(x, idx_i, N) * basis(x, idx_j, N)
        sum_3 = gamma_loc(x) * basis(x, idx_i, N) * basis(x, idx_j, N)
        return sum_1 + sum_2 + sum_3

    res = simpson(a, b, F)
    return res


def alpha(x):
    return 1 + x ** 2


def beta(x):
    return 2 * x


def gamma(x):
    return (np.pi**2) * (x**2)


def build_massMatrix(N):
    a = 1 / 6 * np.ones(N - 1)
    M = sp.diags(a, -1) + sp.diags(a, 1) + 2 / 3 * np.eye(N)
    return 1 / (N + 1) * M


def build_rigidityMatrix(N, alpha, beta, gamma):
    print("build_rigidityMatrix")
    h = 1 / (N + 1)
    A = np.zeros((N, N))

    for i in range(N + 1):
        # Define start and end point of interval K_i. Start points here are shifted as the matrix index is shifted up
        a, b = i * h, (i + 1) * h
        if i not in [0, N]:
            j = i - 1  # The corresponding matrix index, since Python starts indexing at 0
            A[j, j + 1] += integral(a, b, i, i + 1, h, alpha, beta, gamma)
            A[j + 1, j] += integral(a, b, i + 1, i, h, alpha, beta, gamma)
            A[j, j] += integral(a, b, i, i, h, alpha, beta, gamma)
            A[j + 1, j + 1] += integral(a, b, i + 1, i + 1, h, alpha, beta, gamma)
        elif i == 0:
            A[0, 0] = integral(a, b, 1, 1, h, alpha, beta, gamma)
        elif i == N:
            A[N - 1, N - 1] = integral(a, b, N, N, h, alpha, beta, gamma)
    return A


# TODO: Adjust f(t,x) to the solution in a): Done
def f(t, x):
    return np.exp(-t) * ((1 + pi ** 2 * (2 * x ** 2 + 1)) * np.sin(pi * x) - 2 * pi * x * np.cos(pi * x))


def initial_value(x):
    return np.sin(pi * x)


def exact_solution_at_1(x):
    return np.exp(-1) * x * np.sin(np.pi * x)


def build_F(t, N):
    h = 1 / (N + 1)
    a = np.zeros((N, 1))
    for i in range(N):
        a[i] = h * 1 / 3 * (f(t, h * (i + 0.5)) + f(t, h * (i + 1)) + f(t, h * (i + 1.5)))
    return a


def FEM_theta(N, M, theta):
    k = 1 / M
    grid = (1 / (N + 1)) * (np.arange(N) + 1)
    u_sol = initial_value(grid).reshape(N, 1)
    MatrixM = build_massMatrix(N)
    MatrixA = build_rigidityMatrix(N, alpha, beta, gamma)
    B_theta = MatrixM + k * theta * MatrixA
    C_theta = MatrixM - k * (1 - theta) * MatrixA
    for i in range(M):
        F_theta = k * theta * build_F(k * (i + 1), N) + k * (1 - theta) * build_F(k * i, N)
        u_sol = lin.solve(B_theta, C_theta * u_sol + F_theta)
    return u_sol


#### error analysis ####

l = [5,6,7]  # 8,9]
nb_samples = len(l)
N = np.power(2, l) - 1
# M in the case of 3 f)
M = np.power(4, l)
# M in the case of 3 e)
# M = np.power(2, np.arange(2, 2 + nb_samples))
theta = 1  # Change the theta to 0.5 or 1 when needed in 3 e)-f)

#### Do not change any code below! ####
l2error = np.zeros(nb_samples)
k = 1 / M

try:
    for i in range(nb_samples):
        l2error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(
            exact_solution_at_1((1 / (N[i] + 1)) * (np.arange(N[i]) + 1)).reshape(N[i], 1) - FEM_theta(N[i], M[i],
                                                                                                       theta), ord=2)
        if np.isnan(l2error[i]) == True:
            raise Exception("Error unbounded. Plots not shown.")
    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
    if conv_rate[0] < 0:
        raise Exception("Error unbounded. Plots not shown.")
    print("FEM method with theta=" + str(
        theta) + " converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(k, l2error, '-x', label='error')
    plt.loglog(k, k, '--', label='$O(k)$')
    plt.loglog(k, k ** 2, '--', label='$O(k^2)$')
    plt.title('$L^2$ convergence rate', fontsize=13)
    plt.xlabel('$k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
    plt.show()
except Exception as e:
    print(e)

#### error analysis ####
