"""
This code takes a list of video paths, calculates local motion using each of the templates on the videos,
and saves the results in a .csv file
To run this code, change the function "get_paths_and_names" so it will return a list of video paths and names.
Also note the function calc_results_score, this function takes a video of local motion and quantifies it in some way.
Change it however you want.
"""

import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import EOTF.Utils.video_utils as video_utils
import EOTF.Utils.image_utils as image_utils
import EOTF.Utils.file_utils as file_utils
import numpy as np
import time
import csv
import os
import old_videos_name_parser
import pickle


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

    base_path = r'C:\Users\chent\PycharmProjects\EyeOfTheFly_2\data\animations_2'
    files = os.listdir(base_path)
    files = [f for f in files if '.gif' in f] # keep only gifs
    files_names = [os.path.splitext(f)[0] for f in files]
    files_paths = [base_path + '\\' + f for f in files]
    return files_paths, files_names

def calc_results_score(local_motion, local_motion_rand, seperate_pos_neg=True):
    """
    Change this function to determine the way you quantify the emd response (examples: Total Variation, Sum of Squares)
    :param local_motion_rand: Local motion results of a shuffled video (for normalization)
    :param local_motion:
    :return:
    """

    def unnormalized_frame_score(local_motion):
        # 1. Sum Of Squares
        return np.mean(np.square(local_motion))
        # 2. Moving Variance
        #return np.mean(image_utils.moving_variance(local_motion, (3, 3)))
        # 3. Total Variation
        #return image_utils.total_variation(local_motion)

    def calc_results_score_internal(local_motion):
        res = 0
        for i in range(len(local_motion)):
            top = unnormalized_frame_score(local_motion[i])
            bottom = unnormalized_frame_score(local_motion_rand[i])
            if bottom == 0:
                if top == 0:
                    res += 1
                else:
                    print('warning: 0/0 encountered')
                    res += 0
            else:
                res += np.divide(top, bottom)
        res = res / len(local_motion)
        return res

    pos = [np.maximum(frame,0) for frame in local_motion]
    pos_rand = [np.maximum(frame,0) for frame in local_motion_rand]
    pos_res = calc_results_score_internal(pos, pos_rand)
    neg = [np.minimum(frame,0) for frame in local_motion]
    neg_rand = [np.maximum(frame,0) for frame in local_motion_rand]
    neg_res = calc_results_score_internal(neg, neg_rand)
    return pos_res - neg_res

# FLAGS
save_gifs = True
override_saved_date = True

# Some initializations
templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER, EMD.TEMPLATE_SPATIAL, EMD.TEMPLATE_TEMPORAL]
templates_names = ["Fourier", "Glider", "Spatial", "Temporal"]
#templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER]
#templates_names = ["Fourier", "Glider"]
gifs_paths, gifs_names = get_paths_and_names()
results = {}

for i in range(len(gifs_names)):
    print('----- ' + repr(i+1) + '/' + repr(len(gifs_names)) + ': ' + gifs_names[i] + ' -----')
    start_time = time.time()

    name = gifs_names[i]
    path = gifs_paths[i]
    results[name] = {}

    # Parse video's name (for previous project's videos only!!!)
    results[name] = old_videos_name_parser.parse_name_to_dict(name)

    # Convert to photoreceptor's input - if already exist, read it; otherwise, convert and save it.
    frames = video_utils.read_frames(path)
    frames_saved_path = file_utils.results_path(path, 'photoreceptor_converted.pkl')
    if os.path.isfile(frames_saved_path):
        frames = pickle.load(open(frames_saved_path, 'rb'))
    else:
        photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
            PhotoreceptorImageConverter.make_gaussian_kernel(15),
            frames[0].shape, 6000)
        frames = photoreceptor.receive(video_utils.read_frames(path))
        pickle.dump(frames, open(frames_saved_path, 'wb'))
    print('Converted to photoreceptor: ' + repr(time.time() - start_time))

    # Shuffle pixels for normalization
    frames_rand = video_utils.shuffle(frames)

    # Save photoreceptor gif
    if save_gifs:
        photoreceptor_gif_name = file_utils.results_path(path, 'photoreceptor_converted.gif')
        if ~os.path.isfile(photoreceptor_gif_name):
            video_utils.save_gif(frames, photoreceptor_gif_name, strech=True)

    # Add the gif's length to the results
    results[name]["Length"] = len(frames)

    for j in range(len(templates_list)):
        template = templates_list[j]
        template_name = templates_names[j]

        # ---- X AXIS ----
        # Load or calculate local motion
        local_motion_saved_path = file_utils.results_path(path, template_name + '_X.pkl')
        if os.path.isfile(local_motion_saved_path) and not override_saved_date:
            local_motion = pickle.load(open(local_motion_saved_path,'rb'))
        else:
            local_motion = EMD.forward_video(frames, template, axis=0, center=True)
            pickle.dump(local_motion, open(local_motion_saved_path,'wb'))
        # Load or calculate random local motion (for normalization)
        rand_local_motion_saved_path = file_utils.results_path(path, template_name + '_X_rand.pkl')
        if os.path.isfile(rand_local_motion_saved_path) and not override_saved_date:
            local_motion_rand = pickle.load(open(rand_local_motion_saved_path, 'rb'))
        else:
            local_motion_rand = EMD.forward_video(frames_rand, template, axis=0, center=True)
            pickle.dump(local_motion_rand, open(rand_local_motion_saved_path, 'wb'))
        # Add to dictionary
        res = calc_results_score(local_motion, local_motion_rand)
        results[name][template_name + '_X'] = res

        # ---- Y AXIS ----
        # Load or calculate local motion
        local_motion_saved_path = file_utils.results_path(path, template_name + '_Y.pkl')
        if os.path.isfile(local_motion_saved_path) and not override_saved_date:
            local_motion = pickle.load(open(local_motion_saved_path,'rb'))
        else:
            local_motion = EMD.forward_video(frames, template, axis=1, center=True)
            pickle.dump(local_motion, open(local_motion_saved_path,'wb'))
        # Load or calculate random local motion (for normalization)
        rand_local_motion_saved_path = file_utils.results_path(path, template_name + '_Y_rand.pkl')
        if os.path.isfile(rand_local_motion_saved_path) and not override_saved_date:
            local_motion_rand = pickle.load(open(rand_local_motion_saved_path, 'rb'))
        else:
            local_motion_rand = EMD.forward_video(frames_rand, template, axis=1, center=True)
            pickle.dump(local_motion_rand, open(rand_local_motion_saved_path, 'wb'))
        # Add to dictionary
        res = calc_results_score(local_motion, local_motion_rand)
        results[name][template_name + '_Y'] = res

        print('Calculated ' + template_name)

    end_time = time.time()
    print('Total time: ' + repr(end_time - start_time) + ' seconds')

print()

# WRITE TO FILE
# Create file name: 'test_result' + date + running index
i = 1
results_file_name = 'quantify_motion_results\\test_results_' + time.strftime('%y%m%d') + '_'
while os.path.isfile(results_file_name + str(i) + '.csv'):
    i = i+1
file_name = results_file_name + str(i) + '.csv'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
file = open(file_name, 'w+', newline='')

# Write the data (chen: I copied this code from stack overflow so don't ask me what's going on)
fields = list(results[name].keys())
fields.insert(0,'Name')
for key in fields:
    file.write(key + ',')
file.write('\n')
w = csv.DictWriter(file, fields)
for key,val in sorted(results.items()):
    row = {'Name': key}
    row.update(val)
    w.writerow(row)
