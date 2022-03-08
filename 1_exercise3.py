import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
#You may add any package/module you need

np.seterr(all='raise')  # Set how floating-point errors are handled.


def initial_value(x):
    return np.cos(np.pi * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):

    #print("Call exact solution at t=1")
    # todo 3 a)
    res = np.exp(- np.pi**2) * np.cos(np.pi * x)
    #print("return exact solution at t=1")
    return res


#### exact solution at t=1 ####


def makeG(N : int) -> np.matrix :
    """
    Create the matrix G
    :param N:
    :return:
    """

    #print("Make G")
    # Create G
    G = -2 * np.identity(N+1)
    for n in range(1, N):
        G[n, n+1] = 1
        G[n, n-1] = 1

    G[0,1] = 1
    G[N, N-1] = 1

    #print("return G")
    return G


#### numerical scheme ####
def Eulerexplicit(N, M):

    print("M: ", M, " , N: ", N)

    h, k = 1/N, 1/M


    print("h:", h, " , k: ", k)

    #print("Calculate v")
    v = k / (h ** 2)


    # Create G
    G = makeG(N)

    #print("Calculate C")

    C = np.identity(N+1) + v * G
    print("v: ", v)

    #print("print C: ", C)

    x_initial = np.linspace(start=0, stop=1, num=N+1)

    u = initial_value(x_initial)

    #print("initial u: ", u)
    for m in range(M):

        # get u_{m+1}
        u = C.dot(u)
        if u[0] == np.nan:
            print("u is nan, iteration: ", m)

    # print("final u: ", u)

    return u


def Eulerimplicit(N, M):
    h, k = 1 / N, 1 / M

    print("Calculate v")
    v = k / (h ** 2)


    # Create G
    G = makeG(N)

    # Define C and invert it to go the iterative scheme forward, since we have to start at the initial condition
    C = np.identity(N + 1) - v * G
    C = np.invert(C)

    x_initial = np.linspace(start=0, stop=1, num=N + 1)

    u = initial_value(x_initial)

    for m in range(M):
        u = C.dot(u)

    return u

#### numerical scheme ####


#### error analysis ####
l = 5
N = np.array([10, 15, 20, 25, 30])  # todo for 3 c)
M = np.array([100, 100, 100, 100, 100])  # todo  for 3 c) and 3 d)
l2errorexplicit = np.zeros(5)  # error vector for explicit method
l2errorimplicit = np.zeros(5)  # error vector for implicit method
h2k = 1 / (N ** 2) + 1 / M

#### Do not change any code below! ####
try:
    for i in range(5):
        print("N[i]: ", N[i])
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1).reshape(N[i] + 1, 1)) - Eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    print("Explicit method converges: Convergence rate in discrete L2 norm with respect to h^2+k: " + str(conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorexplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='O(h^2+k)')
    plt.title('L2 convergence rate for explicit method', fontsize=13)
    plt.xlabel('h^2+k', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print("Error unbounded for explicit method", e)

try:
    for i in range(5):
        l2errorimplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1).reshape(N[i] + 1, 1)) - Eulerimplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorimplicit), deg=1)
    print("Implicit method converges: Convergence rate in discrete L2 norm with respect to h^2+k: " + str(conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorimplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='O(h^2+k)')
    plt.title('L2 convergence rate for implicit method', fontsize=13)
    plt.xlabel('h^2+k', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except:
    print("Error unbounded for implicit method")

plt.show()

#### error analysis ####
