import numpy as np
import EOTF.EMD as EMD

test_gif = [
    np.array([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ]),
    np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ]),
    np.array([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ]),
    np.array([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ]),
]
res = EMD.forward_video(test_gif, EMD.TEMPLATE_FOURIER)
print(res)