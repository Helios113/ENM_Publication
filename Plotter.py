import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.ma import masked_array

from pathlib import Path
FILE_PATH = Path('./Results')

def run():

    # Target File and Iteration Limit
    target_file = "Test1 Iter_Lim_50 range_x_[0, 4] range_y_[1.7000000000000002, 5.7].npy"
    ITERATION_LIMIT = 100


    data_array = []
    with open(FILE_PATH / target_file, "r") as file:
        data_array = np.load(FILE_PATH / target_file)

    color_data(data_array)





def color_data(data_array):
    fig, ax = plt.subplots()
    range_x = [-9, 11]
    range_y = [-9, 11]

    colors = [cm.get_cmap('Blues'), cm.get_cmap('Greens'), cm.get_cmap('Oranges')]

    number_of_roots = int(np.max(data_array[:,:,0]))

    v1a = np.where(data_array[:, :, 0] == -1, -1, 0)

    v1a = masked_array(v1a, v1a != -1)
    im = ax.imshow(v1a, cmap=cm.gray,
                   origin='lower', extent=range_x + range_y)
    for i in range(number_of_roots+1):
        v1b = np.where(data_array[:,:,0]==i, data_array[:,:,1], 0)
        v1b = masked_array(v1b, v1b == 0)
        im = ax.imshow(v1b, cmap=colors[i],
                       origin='lower', extent=range_x + range_y)
    plt.show(block=True)
    plt.interactive(False)

if __name__ == "__main__":
    run()