import os
import cv2
import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import EOTF.Utils as Utils
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
        return emd.forward_video(buffer)
    return emd.forward_row(buffer, np.round(row*buffer[0].shape[0]).astype(np.uint8))


def read_frames(input_clip: str, grayscale: bool = True):
    """
    Reads a video or a gif.
    :param input_clip: Path of the input video or gif.
    :param grayscale: Whether or not convert to grayscale.
    :return: The input video, as a list of numpy matrices.
    """
    cap = cv2.VideoCapture(input_clip)
    frames = []
    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret is False:
            cap.release()
            cv2.destroyAllWindows()
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    return frames


def emd_response_mid_row(input_clip: str):
    """
    Calculates the angle response of an array of EMDs located at the the medial horizontal line of each frame.
    Inspired by the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    :param input_clip: The clip to process.
    """
    frames = read_frames(input_clip)
    angle_response_over_time_array = calc_emd_responses(frames, 0.5)
    averaged_responses = []
    # for i in range(angle_response_over_time_array.shape[1]):
    #     averaged_responses.append(moving_average(angle_response_over_time_array[:, i]))
    # averaged_responses = np.array(averaged_responses)
    # pickle_output_array(angle_response_over_time_array, clip_file_name, output_dir)
    save_surface_plot(np.array(angle_response_over_time_array), clip_file_name, results_path(clip_path)+'mid_row_graph.png')


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
    results_path = os.path.splitext(clip_path)[0] + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


clip_path = 'data/complex_stripes.gif'
clip_file_name = os.path.splitext(clip_path)[0]

emd_response_mid_row(clip_path)
exit()

emd_result = calc_emd_responses(read_frames(clip_path))
emd_result = Utils.rescale_frames(emd_result, 0, 255, np.uint8)
imageio.mimsave(results_path(clip_path)+'emd_results_test.gif', emd_result)