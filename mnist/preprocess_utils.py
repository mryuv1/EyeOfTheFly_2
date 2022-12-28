from EOTF import EMD
from Utils.writing_results_utils import progress_bar
import numpy as np


def is_preprocessed(data_dict):
    return type(list(data_dict.values())[0][0][0]) is not np.ndarray


def emd_preprocess(dataset, print_progress=False):
    if print_progress:
        print(progress_bar(0, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    for i in range(len(dataset)):
        frames_orig = dataset[i][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            frames_orig[:-1],
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=1)
        ]
        dataset[i] = (frames_preprocessed, dataset[i][1][:-1])
        if print_progress:
            print(progress_bar(i, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    return dataset
