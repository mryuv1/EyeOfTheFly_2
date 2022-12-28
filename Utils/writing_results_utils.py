import os
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator
import csv


def results_path(clip_path, file_name=''):
    """
    Creates a results directory for a given clip.
    :param clip_path:
    :return: Path to the results directory
    """

    clip_path = os.path.splitext(clip_path)[0]
    base_path, name = os.path.split(clip_path)
    res = base_path + '\\' + name + '\\'
    if not os.path.exists(res):
        os.makedirs(res)
    return res + file_name


def save_surface_plot(output_array: np.array, clip_file_name: str, output_file_name: str):
    """
    Shows and saves a surface plot, to be used with emd_response_mid_row.
    :param output_array:
    :param clip_file_name:
    :param output_file_name:
    :return:
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    FRAMES, RESPONSES = output_array.shape
    x = np.array(range(FRAMES))
    y = np.array(range(RESPONSES))
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(np.transpose(X), np.transpose(Y), output_array, cmap=cm.coolwarm, linewidth=0,
                           antialiased=False)
    ax.set_ylabel('time [frames]')
    ax.set_xlabel('EMD')
    ax.set_zlabel('Amplitude')
    # Customize the z axis.
    # ax.set_zlim(-.015, .015)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title(clip_file_name + ' middle row EMD response')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(output_file_name)
    plt.show()
    print("Done")


def write_dict_to_csv(d, file_name):
    """
    Writing a dictionary to a csv format
    :param d: A dictionary, whose fields are lists *of the same length*
    :param file_name:
    :return:
    """
    with open(file_name, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        key_list = list(d.keys())
        writer.writerow(d.keys())
        for i in range(len(d[key_list[0]])):
            writer.writerow(d[key][i] for key in key_list)


def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    Print the value returned using ` end='' `
    Example: print(progress_bar(i, 100, prefix='Process: '), end='')
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    res = f'\r{prefix} |{bar}| {percent}% {suffix}'
    # Print New Line on Complete
    if iteration == total:
        res = res + '\n'
    return res

