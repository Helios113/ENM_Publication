import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from pathlib import Path
FILE_PATH = Path('./Results')

def run():

    # Target File and Iteration Limit
    target_file = "Test1 Iter_Lim_100 range_x_[-9, 11] range_y_[-9, 11].npy"
    ITERATION_LIMIT = 100
    range_x = [-9, 11]
    range_y = [-9, 11]

    data_array = []
    with open(FILE_PATH / target_file, "r") as file:
        data_array = np.load(FILE_PATH / target_file)
    print(data_array)
    fig, ax = plt.subplots()
    im = ax.imshow(data_array, interpolation='bilinear', cmap=cm.RdYlGn,
                   origin='lower', extent=range_x+range_y,
                   vmax=abs(data_array).max(), vmin=-abs(data_array).max())
    plt.show(block=True)
    plt.interactive(False)


if __name__ == "__main__":
    run()