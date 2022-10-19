"""
This code takes a list of video paths, calculates local motion using each of the templates on the videos,
and saves the results in a .csv file
To run this code, change the function "get_paths_and_names" so it will return a list of video paths and names.
Also note the function calc_results_score, this function takes a video of local motion and quantifies it in some way.
Change it however you want.
"""

import os
import pickle
import time

import numpy as np

from EOTF import EMD, PhotoreceptorImageConverter
from Utils import video_utils, old_videos_name_parser
import writing_results_utils
from stand_alone_scripts import create_animations

# FLAGS

save_gifs = True
override_local_motion = True
override_photoreceptor_converted = True

# Initializations
#templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER, EMD.TEMPLATE_SPATIAL, EMD.TEMPLATE_TEMPORAL]
#templates_names = ["Fourier", "Glider", "Spatial", "Temporal"]
templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER]
templates_names = ["Fourier", "Glider"]

def get_paths_and_names():
    """
    Change this function to select the specific gifs you want to test.
    :return: a list of gif paths, and a list of corresponding meaningful names for the gifs
    """

    """
    movement_types = [
#        'diagonal_down_left', 'diagonal_down_right', 'diagonal_up_left', 'diagonal_up_right',
        'down', 'left', 'grow',
#        'up', 'right', 'shrink'
    ]
    movement_speeds = [
        '1',
#        '2',
#        '3'
    ]
    gifs_names = []
    gifs_paths = []
    for type in movement_types:
        for speed in movement_speeds:
            gifs_paths.append('data/animations/' + type + '/' + type + '_speed_' + speed + '.gif')
            gifs_names.append(type + '_speed_' + speed)
    return gifs_paths, gifs_names
    """

    #base_path = r'C:\Users\chent\PycharmProjects\EyeOfTheFly_2\data\RealInputClips2\Pillar(A)\all_videos'
    base_path = os.path.join(os.getcwd(),'data','animations_2')
    base_path = os.path.join(os.getcwd(),'data','nature_images','gifs')
    files = os.listdir(base_path)
    files = [f for f in files if ('.mp4' in f) or ('.gif' in f)] # keep only gifs
    files_names = [os.path.splitext(f)[0] for f in files]
    files_paths = [base_path + '\\' + f for f in files]
    return files_paths, files_names

def calc_results_score(local_motion, local_motion_rand):
    """
    Change this function to determine the way you quantify the emd response (examples: Total Variation, Sum of Squares)
    :param local_motion_rand: Local motion results of a shuffled video (for normalization)
    :param local_motion:
    :return:
    """
    res = 0
    for f, randf in zip(local_motion, local_motion_rand):
        top = _frame_score(f)
        bottom = _frame_score(randf)
        if bottom == 0:
            if top == 0:
                res += 1
            else:
                raise Exception('Tried to divide by 0')
        else:
            res += np.divide(top, bottom)
    res = res / len(local_motion)
    return res

def preprocess_frames(frames):
    frames = [(f > np.mean(f)).astype(int) for f in frames]
    return frames

def _frame_score(lm_frame:np.array):
    # 1. Sum Of Squares
    return np.mean(np.square(lm_frame))
    # 2. Moving Variance
    #return np.mean(image_utils.moving_variance(local_motion, (3, 3)))
    # 3. Total Variation
    #return image_utils.total_variation(local_motion)
    # 4. Count instances
    #return np.count_nonzero(lm_frame)

gifs_paths, gifs_names = get_paths_and_names()
results = {}

for i, (name, path) in enumerate(zip(gifs_names, gifs_paths)):
    print('----- ' + repr(i+1) + '/' + repr(len(gifs_names)) + ': ' + name + ' -----')
    start_time = time.perf_counter()

    # Convert to photoreceptor's input - if already exist, read it; otherwise, convert and save it.
    frames_saved_path = writing_results_utils.results_path(path, 'photoreceptor_converted.pkl')
    if os.path.isfile(frames_saved_path) and not override_photoreceptor_converted:
        frames = pickle.load(open(frames_saved_path, 'rb'))
    else:
        frames = video_utils.read_frames(path)
        if frames[0].shape[0]*frames[0].shape[1] > 6000:
            photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
                PhotoreceptorImageConverter.make_gaussian_kernel(15),
                frames[0].shape, 6000)
            frames = photoreceptor.receive(frames)
        pickle.dump(frames, open(frames_saved_path, 'wb'))
        # Save photoreceptor gif
        if save_gifs:
            photoreceptor_gif_name = writing_results_utils.results_path(path, 'photoreceptor_converted.gif')
            if ~os.path.isfile(photoreceptor_gif_name):
                video_utils.save_gif(frames, photoreceptor_gif_name, strech=True)
    print('Converted to photoreceptor: ' + repr(time.perf_counter() - start_time))

    # Preprocess frame
    frames = preprocess_frames(frames)
    if save_gifs:
        preprocessed_gif_name = writing_results_utils.results_path(path, 'preprocessed.gif')
        video_utils.save_gif(frames, preprocessed_gif_name, strech=True)

    results[name] = {}
    # Parse video's name
    results[name] = old_videos_name_parser.parse_name_to_dict(name)
    if not results[name]:
        results[name] = create_animations.parse_name_to_dict(name)
    # Shuffle pixels for normalization
    frames_rand = video_utils.shuffle(frames)
    # Add the gif's info to the results
    results[name]["Length"] = len(frames)

    for j in range(len(templates_list)):
        template = templates_list[j]
        template_name = templates_names[j]

        # ---- X AXIS ----
        # Load or calculate local motion
        local_motion_saved_path = writing_results_utils.results_path(path, template_name + '_X.pkl')
        if os.path.isfile(local_motion_saved_path) and not override_local_motion:
            local_motion = pickle.load(open(local_motion_saved_path,'rb'))
        else:
            local_motion = EMD.forward_video(frames, template, axis=0)
            pickle.dump(local_motion, open(local_motion_saved_path,'wb'))
        # Load or calculate random local motion (for normalization)
        rand_local_motion_saved_path = writing_results_utils.results_path(path, template_name + '_X_rand.pkl')
        if os.path.isfile(rand_local_motion_saved_path) and not override_local_motion:
            local_motion_rand = pickle.load(open(rand_local_motion_saved_path, 'rb'))
        else:
            local_motion_rand = EMD.forward_video(frames_rand, template, axis=0)
            pickle.dump(local_motion_rand, open(rand_local_motion_saved_path, 'wb'))
        # Add to dictionary
        res = calc_results_score(local_motion, local_motion_rand)
        results[name][template_name + '_X'] = res
        # Save GIF
        if save_gifs:
            local_motion_gif_path = writing_results_utils.results_path(path, template_name + '_X.gif')
            if override_local_motion or ~os.path.isfile(local_motion_gif_path):
                local_motion_int = [((f/f.max()+1)*256/2).astype('uint8') for f in local_motion]
                video_utils.save_gif(local_motion_int, local_motion_gif_path)

        # ---- Y AXIS ----
        # Load or calculate local motion
        local_motion_saved_path = writing_results_utils.results_path(path, template_name + '_Y.pkl')
        if os.path.isfile(local_motion_saved_path) and not override_local_motion:
            local_motion = pickle.load(open(local_motion_saved_path,'rb'))
        else:
            local_motion = EMD.forward_video(frames, template, axis=1)
            pickle.dump(local_motion, open(local_motion_saved_path,'wb'))
        # Load or calculate random local motion (for normalization)
        rand_local_motion_saved_path = writing_results_utils.results_path(path, template_name + '_Y_rand.pkl')
        if os.path.isfile(rand_local_motion_saved_path) and not override_local_motion:
            local_motion_rand = pickle.load(open(rand_local_motion_saved_path, 'rb'))
        else:
            local_motion_rand = EMD.forward_video(frames_rand, template, axis=1)
            pickle.dump(local_motion_rand, open(rand_local_motion_saved_path, 'wb'))
        # Add to dictionary
        res = calc_results_score(local_motion, local_motion_rand)
        results[name][template_name + '_Y'] = res
        # Save GIF
        if save_gifs:
            local_motion_gif_path = writing_results_utils.results_path(path, template_name + '_Y.gif')
            if override_local_motion or ~os.path.isfile(local_motion_gif_path):
                local_motion_int = [((f/f.max()+1)*256/2).astype('uint8') for f in local_motion]
                video_utils.save_gif(local_motion_int, local_motion_gif_path)

        print('Calculated ' + template_name)

    print('Finished: ' + repr(time.perf_counter() - start_time) + ' seconds')

print()

# ------ WRITE TO FILE ------
# Change dictionary structure to fit 'write_dict_to_csv' function
results_new = {'Name': list(results.keys())}
for field_key in results[results_new['Name'][0]].keys():
    results_new[field_key] = [results[vid_key][field_key] for vid_key in results.keys()]
# Create file name: 'test_result' + date + running index
base_results_file_name = 'quantify_motion_results\\test_results_' + time.strftime('%y%m%d') + '_'
i = 1
while os.path.isfile(base_results_file_name + str(i) + '.csv'):
    i = i+1
results_file_name = base_results_file_name + str(i) + '.csv'
writing_results_utils.write_dict_to_csv(results_new, results_file_name)