"""
Implements Elementary Motion Detector (Nitzany 2014)
Use function 'forward_video' (template is B in the article)
"""

import numpy as np

TEMPLATE_FOURIER = np.array([[1, 0], [0, 1]])
TEMPLATE_GLIDER = np.array([[1, 1], [1, 0]])
TEMPLATE_SPATIAL = np.array([[1, 1, 0], [0, 1, 1]])
TEMPLATE_TEMPORAL = np.array([[1, 0], [1, 1], [0, 1]])

def forward(frames, r, c, t, template, axis=0):
    if axis == 0:
        return _forward_x(frames, r, c, t, template)
    return _forward_y(frames, r, c, t, template)

def _forward_raw_x(frames, c, r, t, template):
    """
    Calculate RawCorr from (Nitzany 2014)
    """
    if (t + template.shape[0] - 1 >= len(frames)) or (c + template.shape[1] - 1 >= frames[0].shape[1]):
        return 0
    result = 1
    for xi in range(template.shape[1]):
        for ti in range(template.shape[0]):
            if template[ti, xi]:
                result *= frames[t + ti][r, c + xi]
    return result
def _forward_x(frames, r, c, t, template):
    """
    Calculates EMD response for a single pixel in a single frame
    :param frames: A list of frames, each one is an np matrix
    :param r: Pixel row
    :param c: Pixel column
    :param t: Frame index
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :return: EMD response
    """

    template_x = np.flip(template, axis=1)
    template_t = np.flip(template, axis=0)
    template_xt = np.flip(np.flip(template, axis=1), axis=0)
    local_motion = (_forward_raw_x(frames, c, r, t, template) - _forward_raw_x(frames, c, r, t, template_x)) \
           - (_forward_raw_x(frames, c, r, t, template_t) - _forward_raw_x(frames, c, r, t, template_xt))
    return local_motion

def _forward_raw_y(frames, c, r, t, template, axis=0):
    """
    Calculate RawCorr from (Nitzany 2014)
    """
    if (t + template.shape[0] - 1 >= len(frames)) or (r + template.shape[1] - 1 >= frames[0].shape[0]):
        return 0
    result = 1
    for xi in range(template.shape[1]):
        for ti in range(template.shape[0]):
            if template[ti, xi]:
                result *= frames[t + ti][r + xi, c]
    return result
def _forward_y(frames, r, c, t, template):
    """
    Calculates EMD response for a single pixel in a single frame
    :param frames: A list of frames, each one is an np matrix
    :param r: Pixel row
    :param c: Pixel column
    :param t: Frame index
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :return: EMD response
    """

    template_x = np.flip(template, axis=1)
    template_t = np.flip(template, axis=0)
    template_xt = np.flip(np.flip(template, axis=1), axis=0)
    return (_forward_raw_y(frames, c, r, t, template) - _forward_raw_y(frames, c, r, t, template_x)) \
           - (_forward_raw_y(frames, c, r, t, template_t) - _forward_raw_y(frames, c, r, t, template_xt))

def forward_row(frames, r, template):
    """
    Calculates EMD response for a row in a video
    :param frames: A list of frames, each one is an np matrix
    :param r: Row index
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :param center: Whether of not subtract the mean value of each frame
    :return: A matrix of EMD responses. First axis is spacial, second is temporal.
    """

    result = np.zeros((frames[0].shape[1], len(frames)-1))
    for c in range(result.shape[0]):
        for t in range(result.shape[1]):
            result[c, t] = _forward_x(frames, r, c, t, template)
    return result


def forward_col(frames, c, template):
    """
    Calculates EMD response for a row in a video
    :param frames: A list of frames, each one is an np matrix
    :param c:
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :return: A matrix of EMD responses. First axis is spacial, second is temporal.
    """
    result = np.zeros((frames[0].shape[0], len(frames)-1))
    for r in range(result.shape[0]):
        for t in range(result.shape[1]):
            result[r, t] = _forward_y(frames, r, c, t, template)
    return result


def forward_video(frames, template, axis=0):
    """
    Calculates EMD response for an entire video.
    :param frames: A list of frames, each one is an np matrix
    :param template: As explained in (Nitzany 2014). First axis temporal, second axis spacial.
    :param axis: Spacial axis in which to detect movement. 0 = X, 1 = Y.
    :return: A new video (as a list of frames) of EMD responses.
    :param center: Whether of not subtract the mean value of each frame
    """
    # TODO: Implement threading

    result = []
    if axis == 0:
        for r in range(frames[0].shape[0]):
            result.append(forward_row(frames, r, template))
        result = list(np.transpose(np.array(result), (2, 0, 1)))
    else:
        for c in range(frames[0].shape[1]):
            result.append(forward_col(frames, c, template))
        result = list(np.transpose(np.array(result), (2, 1, 0)))
    return result

