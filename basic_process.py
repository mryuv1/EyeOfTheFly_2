import os
from EOTF import EMD, PhotoreceptorImageConverter
from Utils import writing_results_utils, video_utils
import numpy as np

# Upload frames
clip_path = r'C:\Users\chent\PycharmProjects\EyeOfTheFly_2\data\complex_movement.gif'
clip_file_name = os.path.splitext(clip_path)[0]
frames = video_utils.read_frames(clip_path)

# Convert frames to a photoreceptor image (in practice, down sample it)
photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
    PhotoreceptorImageConverter.make_gaussian_kernel(15),
    frames[0].shape, 6000)
frames = photoreceptor.receive(frames)  # A list of frames

# Calculate EMD responses for entire frame and save as a gif
emd_result = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER, axis=1)
video_utils.save_gif(frames, 'emd_results_test.gif', writing_results_utils.results_path(clip_path), strech=True)

exit()

# Calculate EMD responses for middle row and save surface plot
emd_response_mid_row = EMD.forward_row(frames, np.round(0.5 * frames[0].shape[0]).astype(np.uint8), EMD.TEMPLATE_FOURIER)
writing_results_utils.save_surface_plot(np.array(emd_response_mid_row), clip_file_name, writing_results_utils.results_path(clip_path, 'mid_row_graph.png'))