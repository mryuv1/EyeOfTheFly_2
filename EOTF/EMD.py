import numpy as np
import EOTF.Utils.VideoUtils as VideoUtils
import cv2

TEMPLATE_FOURIER = np.array([[1, 0], [0, 1]])
TEMPLATE_SPATIAL = np.array([[1, 1, 0], [0, 1, 1]])
TEMPLATE_TEMPORAL = np.array([[1, 0], [1, 1], [0, 1]])
TEMPLATE_GLIDER = np.array([[1, 1], [0, 1]])


def forward(frames, r, c, t, template, axis=0):
    """
    Calculates EMD response for a single pixel in a single frame
    :param frames: A list of frames, each one is an np matrix
    :param r: Pixel row
    :param c: Pixel column
    :param t: Frame index
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :param center: Whether of not subtract the mean value of each frame
    :return: EMD response
    """

    def forward_raw(frames, c, r, t, template, axis=0):
        """
        Calculate RawCorr from (Nitzany 2014)
        """

        result = 1
        if axis == 0:
            if (t + template.shape[0] >= VideoUtils.length(frames)) \
                    or (c + template.shape[1] >= VideoUtils.width(frames)):
                return 0
            for xi in range(template.shape[1]):
                for ti in range(template.shape[0]):
                    if template[ti, xi]:
                        result *= frames[t + ti][r, c + xi]
        else:
            if (t + template.shape[0] >= VideoUtils.length(frames)) \
                    or (r + template.shape[1] >= VideoUtils.height(frames)):
                return 0
            for xi in range(template.shape[1]):
                for ti in range(template.shape[0]):
                    if template[ti, xi]:
                        result *= frames[t + ti][r + xi, c]
        return result

    template_x = np.flip(template, axis=1)
    template_t = np.flip(template, axis=0)
    template_xt = np.flip(np.flip(template, axis=1), axis=0)
    return (forward_raw(frames, c, r, t, template, axis) - forward_raw(frames, c, r, t, template_x, axis)) \
           - (forward_raw(frames, c, r, t, template_t, axis) - forward_raw(frames, c, r, t, template_xt, axis))


def forward_row(frames, r, template):
    """
    Calculates EMD response for a row in a video
    :param frames: A list of frames, each one is an np matrix
    :param r: Row index
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :param center: Whether of not subtract the mean value of each frame
    :return: A matrix of EMD responses. First axis is spacial, second is temporal.
    """

    result = np.zeros((frames[0].shape[1], len(frames)))
    for c in range(result.shape[0]):
        for t in range(result.shape[1]):
            result[c, t] = forward(frames, r, c, t, template, axis=0)
    return result


def forward_col(frames, c, template):
    """
    Calculates EMD response for a row in a video
    :param frames: A list of frames, each one is an np matrix
    :param c:
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :return: A matrix of EMD responses. First axis is spacial, second is temporal.
    """
    result = np.zeros((frames[0].shape[0], len(frames)))
    for r in range(result.shape[0]):
        for t in range(result.shape[1]):
            result[r, t] = forward(frames, r, c, t, template, axis=1)
    return result


def forward_video(frames, template, axis=0, center=False):
    """
    Calculates EMD response for an entire video.
    :param frames: A list of frames, each one is an np matrix
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :param axis: Spacial axis in which to detect movement. 0 = X, 1 = Y.
    :return: A new video (as a list of frames) of EMD responses.
    :param center: Whether of not subtract the mean value of each frame
    """

    if center:
        frames = [f - np.mean(f) for f in frames]

    result = []
    for t in range(len(frames)-1):
        result_t = np.zeros_like(frames[t])
        for r in range(frames[0].shape[0]):
            for c in range(frames[0].shape[1]):
                result_t[r, c] = forward(frames, r, c, t, template, axis)
        result.append(result_t)
    return result
