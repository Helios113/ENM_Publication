import numpy as np
from SolverTypes import SolverTypes
from NewtonSolver import Solver
from tools.progress_bar import ProgressBar
from pathlib import Path

# return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1,
#                  2 * x[0] ** 2 + x[1] ** 2 - 4 * x[2],
#                  3 * x[0] ** 2 - 4 * x[1] ** 2 + x[2] ** 2])

# This can be an array of functions that we then use one after the other
l = lambda x: np.array([np.exp(x[0]) - x[1], x[0] * x[1] - np.exp(x[0])])
FILE_PATH = Path('./Results/')


def do_test():
    # Solver to be used
    method = SolverTypes.ENM

    # Limit number of iterations
    ITERATION_LIMIT = 50

    # Valid roots
    valid_roots = np.array([[1,2.7182818284590455],[0,-5]])

    # Create solver object
    solver = Solver(l, method,valid_roots, iteration_limit=ITERATION_LIMIT)

    # Beginning and end of the test range
    start = -1
    stop = 3

    # average of roots
    center_x = 1
    center_y = 2.7

    # Number of points between start and stop
    n = 100

    # Choice of c
    c = np.array([2, 3])

    # Create progress bar
    progress_bar = ProgressBar(n ** 2)

    # Create data storage array
    result_array = np.zeros((n, n, 2))

    for i, ii in enumerate(np.linspace(start+center_x, stop+center_x, n)):
        for j, jj in enumerate(np.linspace(start+center_y, stop+center_y, n)):
            progress_bar.draw(i * n + j + 1, )
            x = np.array([ii, jj])
            res = solver.run(x, c)
            result_array[i, j, :] = res

    name = "Test1" + " Iter_Lim_{lim} range_x_{rangex} range_y_{rangey}.npy".format(lim=ITERATION_LIMIT,rangex=[start+center_x, stop+center_x],rangey=[start+center_y, stop+center_y])
    with open(FILE_PATH / name, "w+") as file:
        np.save(FILE_PATH / name, result_array)


if __name__ == "__main__":
    do_test()
