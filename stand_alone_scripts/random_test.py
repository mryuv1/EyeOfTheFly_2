import numpy as np
from EOTF import EMD
from Utils import video_utils
import writing_results_utils

fourier_motion = {
    'fourier_score': [],
    'glider_score': []
}
glider_motion = {
    'fourier_score': [],
    'glider_score': []
}
random_motion = {
    'fourier_score': [],
    'glider_score': []
}

imsize = 10

for i in range(100):
    print(repr(i))
    # Create random image
    image1 = np.random.randn(imsize,imsize)
    image1 = (image1 > 0).astype(int)

    # Do Fourier movement (in x)
    image2 = np.zeros_like(image1)
    image2[:,1:] = image1[:,:imsize-1]
    image2[:,0] = (np.random.randn(imsize) > 0).astype('int8')

    # Calculate movements
    frames = [image1, image2]
    frames_rand = video_utils.shuffle(frames)
    fourier_lm = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER)
    fourier_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_FOURIER)
    fourier_score = np.reshape(np.abs(fourier_lm),-1)
    fourier_motion['fourier_score'].extend(fourier_score)
    glider_lm = EMD.forward_video(frames, EMD.TEMPLATE_GLIDER)
    glider_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_GLIDER)
    glider_score = np.reshape(np.abs(glider_lm),-1)
    fourier_motion['glider_score'].extend(glider_score)

    # Do Glider movement (in x)
    image2 = np.zeros_like(image1)
    for r in range(imsize):
        for c in range(1,imsize):
            if image1[r, c] == 1 or image1[r, c - 1] == 1:
                image2[r,c] = 1
    image2[:,0] = (np.random.randn(imsize) > 0).astype('int8')

    # Calculate movements
    frames = [image1, image2]
    frames_rand = video_utils.shuffle(frames)
    fourier_lm = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER)
    fourier_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_FOURIER)
    fourier_score = np.reshape(np.abs(fourier_lm),-1)
    glider_motion['fourier_score'].extend(fourier_score)
    glider_lm = EMD.forward_video(frames, EMD.TEMPLATE_GLIDER)
    glider_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_GLIDER)
    glider_score = np.reshape(np.abs(glider_lm),-1)
    glider_motion['glider_score'].extend(glider_score)

    # Do random movement
    image2 = np.random.randn(imsize,imsize)
    image2 = (image2 > 0).astype(int)

    # Calculate movements
    frames = [image1, image2]
    frames_rand = video_utils.shuffle(frames)
    fourier_lm = EMD.forward_video(frames, EMD.TEMPLATE_FOURIER)
    fourier_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_FOURIER)
    fourier_score = np.reshape(np.abs(fourier_lm),-1)
    random_motion['fourier_score'].extend(fourier_score)
    glider_lm = EMD.forward_video(frames, EMD.TEMPLATE_GLIDER)
    glider_lm_rand = EMD.forward_video(frames_rand, EMD.TEMPLATE_GLIDER)
    glider_score = np.reshape(np.abs(glider_lm),-1)
    random_motion['glider_score'].extend(glider_score)

print('Fourier Motion Scores:' + repr(fourier_motion))
print('Glider Motion Scores:' + repr(glider_motion))
print('Random Motion Scores:' + repr(random_motion))


writing_results_utils.write_dict_to_csv(fourier_motion, '../random_fourier_results.csv')
writing_results_utils.write_dict_to_csv(glider_motion, '../random_glider_results.csv')
writing_results_utils.write_dict_to_csv(random_motion, '../random_random_results.csv')
