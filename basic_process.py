import os
import cv2
import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
                                                       PhotoreceptorImageConverter.PhotoreceptorImageConverter(
                                                           PhotoreceptorImageConverter.make_gaussian_kernel(15),
                                                           greyscale_frames[0].shape, 6000))
    # pickle_output_array(angle_response_over_time_array, clip_file_name, output_dir)
    save_surface_plot(angle_response_over_time_array, clip_file_name, output_dir)


def get_emd_responses(frames, photoreceptor: PhotoreceptorImageConverter.PhotoreceptorImageConverter) -> np.array:
    """
    Calculate The angle responses through the photoreceptor's.
    :param frames: The frames to process.
    :param photoreceptor: The converter of frames to photoreceptor responses.
    :return: The angle response array for each frame.
    """
    angle_response_over_time = []
    emd = EMD.EMD()
    for buffer in photoreceptor.stream(frames, buffer_size=BUFFER_SIZE):
        emd_mid_row_response = emd.forward_row(buffer, buffer[0].shape[0] // 2)
        frequency_response_emd = [np.abs(np.fft.rfft(tr)) for tr in emd_mid_row_response]
        angle_response_emd = angle_response_from_frequency_response_array(frequency_response_emd)
        angle_response_over_time.append(angle_response_emd)
    #return np.array(angle_response_over_time)  # filtered version
    return np.array(emd_mid_row_response) # unfiltered version


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


def get_frames(input_clip: str, grayscale: bool = True):
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


def save_surface_plot(output_array: np.array, clip_file_name: str, output_dir: str):
    fig = go.Figure(data=[go.Surface(z=output_array)])

    fig.update_layout(title=clip_file_name, autosize=True, scene=dict(
                    xaxis_title='time [frames]',
                    yaxis_title='EMD',
                    zaxis_title='Amplitude'))
    fig.write_image(os.path.join(output_dir, os.path.splitext(clip_file_name)[0] + '.png'))
    fig.show()

    # If you want to plot also a EMD response and to see it's frames
    fig2 = px.line(pd.DataFrame({'amp':output_array[2]}), y="amp", title='Life expectancy in Canada')
    fig2.show()


#
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
#
# fig = go.Figure(data=[go.Surface(z=z_data.values)])
#
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
#
# fig.show()

basic_response_mid_row(os.getcwd(), 'data/stripes.gif')
