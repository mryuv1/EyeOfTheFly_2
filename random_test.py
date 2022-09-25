import numpy as np
import EOTF.EMD as EMD
import EOTF.Utils.video_utils as video_utils

# Create random image
image1 = np.random.randn(100,100)
image1 = (image1 > 0).astype('int8')

# Do Fourier movement (in x)
image2 = np.zeros_like(image1)
image2[:,1:] = image1[:,:99]
image2[:,0] = (np.random.randn(100) > 0).astype('int8')

# Calculate movements
frames = [image1, image2]
frames_rand = video_utils.shuffle(frames)
fourier_lm = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER)
fourier_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_FOURIER)
fourier_score = np.mean(np.square(fourier_lm)) / np.mean(np.square(fourier_lm_rand))
glider_lm = EMD.forward_video(frames, EMD.TEMPLATE_GLIDER)
glider_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_GLIDER)
glider_score = np.mean(np.square(glider_lm)) / np.mean(np.square(glider_lm_rand))

print('Scores for Fourier motion:')
print('Fourier: ' + repr(fourier_score) + ', Glider: ' + repr(glider_score))

# Do Glider movement (in x)
image2 = np.zeros_like(image1)
for r in range(100):
    for c in range(1,100):
        if image1[r, c] == 1 or image1[r, c - 1] == 1:
            image2[r,c] = 1
image2[:,0] = (np.random.randn(100) > 0).astype('int8')

# Calculate movements
frames = [image1, image2]
frames_rand = video_utils.shuffle(frames)
fourier_lm = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER)
fourier_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_FOURIER)
fourier_score = np.mean(np.square(fourier_lm)) / np.mean(np.square(fourier_lm_rand))
glider_lm = EMD.forward_video(frames, EMD.TEMPLATE_GLIDER)
glider_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_GLIDER)
glider_score = np.mean(np.square(glider_lm)) / np.mean(np.square(glider_lm_rand))

print('Scores for Glider motion:')
print('Fourier: ' + repr(fourier_score) + ', Glider: ' + repr(glider_score))
