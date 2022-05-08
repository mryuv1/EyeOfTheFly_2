import os
import cv2
import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator


BUFFER_SIZE = 120


def basic_response_mid_row(output_dir: str, input_clip: str):
    """
    Calculates the angle response of an array of EMDs located at the the medial horizontal line of each frame.
    Inspired by the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    :param output_dir: The directory to save the output(s) to.
    :param input_clip: The clip to process.
    """
    clip_file_name = os.path.split(input_clip)[1]
    greyscale_frames = get_frames(input_clip)
    angle_response_over_time_array = get_emd_responses(greyscale_frames,
                                                       PhotoreceptorImageConverter.PhotoreceptorImageConverter(PhotoreceptorImageConverter.make_gaussian_kernel(15), greyscale_frames[0].shape, 6000))
    #pickle_output_array(angle_response_over_time_array, clip_file_name, output_dir)
    save_surface_plot(angle_response_over_time_array, clip_file_name, output_dir)


def get_emd_responses(frames, photoreceptor: PhotoreceptorImageConverter.PhotoreceptorImageConverter) -> np.array:
    """
    Calculate The angle responses through the photoreceptor's.
    :param frames: The frames to process.
    :param photoreceptor: The converter of frames to photoreceptor responses.
    :return: The angle response array for each frame.
    """
    angle_respone_over_time = []
    emd = EMD.EMD()
    for buffer in photoreceptor.stream(frames, buffer_size=BUFFER_SIZE):
        emd_mid_row_response = emd.forward_row(buffer, buffer[0].shape[0] // 2)
        frequency_response_emd = [np.abs(np.fft.rfft(tr)) for tr in emd_mid_row_response]
        angle_response_emd = angle_response_from_frequency_response_array(frequency_response_emd)
        angle_respone_over_time.append(angle_response_emd)
    return np.array(angle_respone_over_time)


def angle_response_from_frequency_response_array(frequency_response_array: np.array) -> np.array:
    """
    Calculates Equation (1) from the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    Namely, R_ø =∫R(ƒ)/ƒ^2dƒ
    :param frequency_response_array: R(ƒ)
    :return: R_ø
    """
    angle_response_emd = list()
    for fr in frequency_response_array:
        integrand = list()
        for idx, val in enumerate(fr):
            if idx:
                normalizer = idx ** -2  # 1/ƒ^2
            else:
                normalizer = 0  # Cancel DC response and prevent division by 0
            integrand.append(normalizer * val)
        angle_response_emd.append(sum(integrand))
    return angle_response_emd


def get_frames(input_clip: str):
    cap = cv2.VideoCapture(input_clip)
    g_frames = []
    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret is False:
            cap.release()
            cv2.destroyAllWindows()
            break
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_frames.append(g_frame)
    return g_frames


def save_surface_plot(output_array: np.array, clip_file_name: str, output_dir: str):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    FRAMES, RESPONSES = output_array.shape
    x = np.array(range(FRAMES))
    y = np.array(range(RESPONSES))
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(np.transpose(X), np.transpose(Y), output_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('EMD')
    ax.set_zlabel('Amplitude')
    # Customize the z axis.
    ax.set_zlim(-.1, .5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title(clip_file_name)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, os.path.splitext(clip_file_name)[0] + '.png'))
    print("Done")


basic_response_mid_row('', 'data/stripes.gif')
