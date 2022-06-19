import os
import cv2
import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import EOTF.Utils.VideoUtils as VideoUtils
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator
import imageio

import glob
from PIL import Image

BUFFER_SIZE = 120


def calc_emd_responses(frames, row=-1):
    """
    Calculates EMD responses of an input video.
    :param frames: Input video, as a list of numpy matrices
    :param row: Optional, if needed calculate the EMD responses of a single row. Relative to the frame's height - ie the
                middle row is 0.5.
                If not specified, the entire frame will be processed.
    :return: The EMD responses, as a list of frames.
    """
    # row is relative to the frame's height - for example, mid row is 0.5
    emd = EMD.EMD()
    photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
        PhotoreceptorImageConverter.make_gaussian_kernel(15),
        frames[0].shape, 6000)
    buffer = photoreceptor.receive(frames)  # A list of frames
    if row < 0:
        res = emd.forward_video(buffer)
    else:
        res = emd.forward_row(buffer, np.round(row * buffer[0].shape[0]).astype(np.uint8))
    res.pop()  # Remove last frames, because it's all zeros...
    return res


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


def results_path(clip_path):
    """
    Creates a results directory for a given clip.
    :param clip_path:
    :return: Path to the results directory
    """
    res = os.path.splitext(clip_path)[0] + '/'
    if not os.path.exists(res):
        os.makedirs(res)
    return res


clip_path = 'data/complex_stripes.gif'
clip_file_name = os.path.splitext(clip_path)[0]
frames = VideoUtils.read_frames(clip_path)

# Calculate EMD responses for middle row and save surface plot
emd_response_mid_row = calc_emd_responses(frames, 0.5)
save_surface_plot(np.array(emd_response_mid_row), clip_file_name, results_path(clip_path) + 'mid_row_graph.png')

# exit()

# Calculate EMD responses for entire frame and save as a gif
emd_result = calc_emd_responses(frames)
imageio.mimsave(results_path(clip_path) + 'emd_results_test.gif', VideoUtils.rescale(emd_result, 0, 255, np.uint8))
