import numpy as np
import EOTF.Utils.VideoUtils as VideoUtils


TEMPLATE_FOURIER = np.array([[1, 0], [0, 1]])
TEMPLATE_SPATIAL = np.array([[1, 1, 0], [0, 1, 1]])
TEMPLATE_TEMPORAL = np.array([[1, 0], [1, 1], [0, 1]])
TEMPLATE_GLIDER = np.array([[1, 1], [0, 1]])


def forward(frames, r, c, t, template):
    """
    Calculates EMD response for a single pixel in a single frame
    :param frames: A list of frames, each one is an np matrix
    :param r: Pixel row
    :param c: Pixel column
    :param t: Frame index
    :param template: A template, as explained in (Nitzany 2014)
    :return: EMD response
    """
    def forward_raw(frames, c, r, t, template):
        """
        Calculate RawCorr from (Nitzany 2014)
        """
        if (t + template.shape[0] >= VideoUtils.length(frames)) or (c + template.shape[1] >= VideoUtils.width(frames)):
            return 0
        result = 1
        for xi in range(template.shape[1]):
            for ti in range(template.shape[0]):
                if template[ti, xi]:
                    result *= frames[t + ti][r, c + xi]
        return result

    template_x = np.flip(template, axis=1)
    template_t = np.flip(template, axis=0)
    template_xt = np.flip(np.flip(template, axis=1), axis=0)
    return (forward_raw(frames, c, r, t, template) - forward_raw(frames, c, r, t, template_x)) \
           - (forward_raw(frames, c, r, t, template_t) - forward_raw(frames, c, r, t, template_xt))


def forward_row(frames, row_index, template):
    """
    Calculates EMD response for a row in a video
    :param frames: A list of frames, each one is an np matrix
    :param row_index:
    :param template: A template, as explained in (Nitzany 2014)
    :return: A matrix of EMD responses. First axis is spacial, second is temporal.
    """
    result = np.zeros((frames[0].shape[1], len(frames)))
    for x in range(result.shape[0]):
        for t in range(result.shape[1]):
            result[x, t] = forward(frames, row_index, x, t, template)
    return result


def forward_video(frames, template, axis=0):
    """
    Calculates EMD response for an entire video.
    :param frames: A list of frames, each one is an np matrix
    :param template: A template, as explained in (Nitzany 2014)
    :param axis: Spaical axis in which to detect movement. 0 = X, 1 = Y.
    :return: A new video (as a list of frames) of EMD responses.
    """
    if axis:
        frames = [np.transpose(f) for f in frames]
    result = []
    for r in range(frames[0].shape[0]):
        result.append(forward_row(frames, r, template))
    if axis:
        result = list(np.transpose(np.array(result), (2, 1, 0)))
    else:
        result = list(np.transpose(np.array(result), (2, 0, 1)))
    return result

