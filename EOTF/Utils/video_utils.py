import cv2
import imageio
import numpy as np
import EOTF.Utils.image_utils as ImageUtils


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
    if max_val == 0:
        return [np.zeros_like(f).astype(target_type) for f in frames]
    return [(target_min + (target_max - target_min) * (f - min_val) / (max_val - min_val)).astype(target_type) for f in frames]


def float_to_int(frames):
    # assuming frames is in range 0-1
    frames = [(f*256).astype('uint8') for f in frames]
    return frames


def length(frames):
    return len(frames)


def width(frames):
    return frames[0].shape[1]


def height(frames):
    return frames[0].shape[0]


def mean_per_frame(frames):
    return [np.mean(frames[i]) for i in range(length(frames))]


def total_variation_video(frames):
    s = 0
    for i in range(length(frames)):
        s = s + ImageUtils.total_variation(frames[i])
    return s


def shuffle(frames):
    frames1 = np.reshape(frames, -1)
    np.random.shuffle(frames1)
    frames1 = np.reshape(frames1, (len(frames),) + frames[0].shape)
    frames1 = list(frames1)
    return frames1


def save_gif(frames, name, path = '', strech = False):
    if strech:
        frames = rescale(frames, 0, 255, np.uint8)
    imageio.mimsave(path + name, frames)