import cv2
import numpy as np


def read_frames(input_clip: str, grayscale: bool = True) -> list:
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


def rescale(frames, target_min=0, target_max=1):
    # frames = list of numpy matrices
    min_val = np.min(frames)
    max_val = np.max(frames)
    return [(target_min + (target_max - target_min) * (i - min_val) / (max_val - min_val)) for i in frames]


def rescale(frames, target_min, target_max, target_type):
    # frames = list of numpy matrices
    min_val = np.min(frames)
    max_val = np.max(frames)
    return [(target_min + (target_max - target_min) * (i - min_val) / (max_val - min_val)).astype(target_type) for i in frames]


def length(frames):
    return len(frames)


def width(frames):
    return frames[0].shape[1]


def height(frames):
    return frames[0].shape[0]


def mean_per_frame(frames):
    return [np.mean(frames[i]) for i in range(length(frames))]
