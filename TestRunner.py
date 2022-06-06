import numpy as np
from SolverTypes import SolverTypes
from NewtonSolver import Solver
from tools.progress_bar import ProgressBar
from pathlib import Path
import matplotlib.pyplot as plt
import lhsmdu

# return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1,
#                  2 * x[0] ** 2 + x[1] ** 2 - 4 * x[2],
#                  3 * x[0] ** 2 - 4 * x[1] ** 2 + x[2] ** 2])

# This can be an array of functions that we then use one after the other
l = []

# old classic system
l.append(lambda x: np.array([np.exp(x[0]) - x[1],
                             x[0] * x[1] - np.exp(x[0])]))

# NEW systems

# system 1 exp in x1
# root (sqrt(2), sqrt(2)), (0,0)
l.append(lambda x: np.array([np.exp(x[0] ** 2) - np.exp(np.sqrt(2) * x[0]),
                             x[0] - x[1]]))

# system 2 exp in x2, trig in x2
# root (0,0)
l.append(lambda x: np.array([x[0] + np.exp(x[1]) - np.cos(x[1]),
                             3 * x[0] - x[1] - np.sin(x[1])]))

# system 3 quadratic in both
# root (1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2)
l.append(lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 1,
                             x[0] ** 2 - x[1] ** 2 + 0.5]))

# system 4 trig in x1
# root (0, 0)
l.append(lambda x: np.array([np.sin(x[0]) + x[1] * np.cos(x[0]),
                             x[0] - x[2]]))

# system 5, 3 dof
# root
# -2.09029464  2.14025812 -0.22352512
# [-0.25537234  2.5724839  -1.52220615]
# [ 2.14025812 -2.09029464 -0.22352512]

l.append(lambda x: np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 9,
                             x[0] * x[1] * x[2] - 1,
                             x[0] + x[1] - x[2] ** 2]))

# system 6, 4 dof
l.append(lambda x: np.array([x[1] * x[2] + x[3] * (x[1] + x[2]),
                             x[0] + x[2] + x[3] * (x[0] + x[2]),
                             x[0] * x[1] + x[3] * (x[0] + x[1]),
                             x[0] * x[1] + x[0] * x[2] + x[1] * x[3] - 1]))

FILE_PATH = Path('./Results/')


def do_test2D():
    # Solver to be used
    method = SolverTypes.ENM

    # Limit number of iterations
    ITERATION_LIMIT = 50

    # Valid roots
    valid_roots = np.array([[1.000, 2.71828183]])

    # Create solver object
    solver = Solver(l[0], method, valid_roots=valid_roots,iteration_limit=ITERATION_LIMIT)

    # Beginning and end of the test range
    start = -50
    stop = 50

    # average of roots
    center_x = 0
    center_y = 0

    # Number of points between start and stop
    n = 10

    # Choice of c
    c = np.array([1, 2])

    # Create progress bar
    progress_bar = ProgressBar(n ** 2)

    # Create data storage array
    result_array = np.zeros((n, n, 2))

    for i, ii in enumerate(np.linspace(start + center_x, stop + center_x, n)):
        for j, jj in enumerate(np.linspace(start + center_y, stop + center_y, n)):
            progress_bar.draw(i * n + j + 1, )
            print("Initial Guess:", ii,jj)
            x = np.array([ii, jj])
            res = solver.run(x,2*x,continually_change=True,change_first=False)
            if res[0] != -1:
                print("Result:", res)
                # order_conv(solver.error, valid_roots[res[0]])
                # res[0] shows root number
                # order_conv (values of x, root)
            result_array[i, j, :] = res

    name ="aaa"+ method.value + " Iter_Lim_{lim} range_x_{rangex} range_y_{rangey} stops_{stops}.npy".format(lim=ITERATION_LIMIT,
                                                                                      rangex=[start + center_x,
                                                                                              stop + center_x],
                                                                                      rangey=[start + center_y,
                                                                                              stop + center_y],
                                                                                      stops=n)
    with open(FILE_PATH / name, "w+") as file:
        np.save(FILE_PATH / name, result_array)
    print(name)


def do_testND():
    # Solver to be used
    method = SolverTypes.Broyden

    # Limit number of iterations
    ITERATION_LIMIT = 50

    # Valid roots
    valid_roots = np.array([[2.5, 0.5, 1.5]])

    # Create solver object
    solver = Solver(l[5], method, iteration_limit=ITERATION_LIMIT)
    dim = valid_roots.shape[1]

    # Beginning and end of the test range
    start = -100
    stop = 100

    # Choice of c
    c = np.array([5.1, 5.1])

    # Number of samples
    sample_size = 30

    # Sample data
    k = lhsmdu.sample(dim, sample_size)

    # Scale data
    k = start + (k * (start - stop))

    # Create data storage array
    result_array = np.zeros((sample_size, 2))

    # Create progress bar
    progress_bar = ProgressBar(sample_size)

    for i in range(sample_size):
        x = np.array(k[:, i]).flatten()
        progress_bar.draw(i+1)
        res = solver.run(x, 0.0001+ x)
        if res[0] != -1:
            print(res)
        result_array[i, :] = res
    name = "ENMc" + " Iter_Lim_{lim} range_{range} samples_{samples}.npy".format(lim=ITERATION_LIMIT,
                                                                                    range=[start, stop],
                                                                                    samples=sample_size
                                                                                    )
    with open(FILE_PATH / name, "w+") as file:
         np.save(FILE_PATH / name, result_array)

def order_conv(e, root):
    e = np.linalg.norm(e-root, axis=1)
    print("Error: ", e)
    p = np.log10(e[2:] / e[1:-1]) / np.log10(e[1:-1] / e[:-2])
    print("Order: ", p)
    mw = p.sum() / len(p)
    plt.plot(p)
    plt.show()
    print("Order of conv", mw)
    return mw  #



if __name__ == "__main__":
    do_test2D()
