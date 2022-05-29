import numpy as np


def rescale_frames(frames, target_min, target_max):
    # frames = list of numpy matrices
    min_val = np.min(frames)
    max_val = np.max(frames)
    return [(target_min + (target_max - target_min) * (i - min_val) / (max_val - min_val)) for i in frames]


def rescale_frames(frames, target_min, target_max, target_type):
    # frames = list of numpy matrices
    min_val = np.min(frames)
    max_val = np.max(frames)
    return [(target_min + (target_max - target_min) * (i - min_val) / (max_val - min_val)).astype(target_type) for i in frames]