import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.ma import masked_array
from matplotlib.ticker import PercentFormatter
from pathlib import Path
FILE_PATH = Path('./Results')

def run2D():

    # Target File and Iteration Limit
    target_file = "ENMv Iter_Lim_50 range_x_[-49.0, 51.0] range_y_[-47.28171817, 52.71828183].npy"
    ITERATION_LIMIT = 50


    data_array = []
    with open(FILE_PATH / target_file, "r") as file:
        data_array = np.load(FILE_PATH / target_file)

    # 2D data
    color_data(data_array)


def color_data(data_array):
    fig, ax = plt.subplots()
    range_x = [-50,50]
    range_y = [-50,50]

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
    # plt.title("ENM c = 0.001X \n[x[0]+np.exp(x[1])-np.cos(x[1]),\n 3*x[0]-x[1]-np.sin(x[1])]")
    plt.show(block=True)

def runNd():
    # Target File and Iteration Limit
    target_files = []
    target_files.append("ENM Iter_Lim_50 range_[-100, 100] samples_30.npy")
    target_files.append("LM Iter_Lim_50 range_[-100, 100] samples_30.npy")
    target_files.append("Powell Iter_Lim_50 range_[-100, 100] samples_30.npy")
    target_files.append("Broyden Iter_Lim_50 range_[-100, 100] samples_30.npy")
    ITERATION_LIMIT = 50

    data_array = []
    for i in target_files:
        with open(FILE_PATH / i, "r") as file:
            data_array.append(np.load(FILE_PATH / i))

    # 2D data
    nd_data(data_array)

def nd_data(data_array):
    fig, ax = plt.subplots()
    conditioned_data = []
    for data in data_array:
        v1a = np.where(data[:,0] != -1, data[:,1], 0)
        conditioned_data.append(v1a)
    ax.hist(conditioned_data)#, weights=[np.ones(len(v1a)) / len(v1a), np.ones(len(v1a)) / len(v1a),  np.ones(len(v1a)) / len(v1a),np.ones(len(v1a)) / len(v1a)])
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.show(block=True)
if __name__ == "__main__":
    run2D()