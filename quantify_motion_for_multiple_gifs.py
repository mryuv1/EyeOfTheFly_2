"""
This code takes a list of video paths, calculates local motion using each of the templates on the videos,
and saves the results in a .csv file
To run this code, change the function "get_paths_and_names" so it will return a list of video paths and names.
Also note the function calc_results_score, this function takes a video of local motion and quantifies it in some way.
Change it however you want.
"""

import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import EOTF.Utils.video_utils as VideoUtils
import numpy as np
import time
import csv
import prev_project_video_name_parser


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

    return prev_project_video_name_parser.get_paths_and_names(base_path=r'data\RealInputClips2\All Videos')


def calc_results_score(local_motion, local_motion_rand=-1):
    """
    Change this function to determine the way you quantify the emd response (examples: Total Variation, Sum of Squares)
    :param local_motions:
    :return:
    """

    if local_motion_rand==-1:
        local_motion_rand = np.ones_like(local_motion)

    #return VideoUtils.total_variation_video(local_motion)

    res = 0
    for i in range(len(local_motion)):
        top = np.mean(np.square(local_motion[i]))
        bottom = np.mean(np.square(local_motion_rand[i]))
        if bottom == 0:
            if top == 0:
                res += 1
            else:
                print('warning: 0/0 encountered')
                res += 0
        else:
            res += np.divide(top, bottom)
    return res


templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER, EMD.TEMPLATE_SPATIAL, EMD.TEMPLATE_TEMPORAL]
templates_names = ["Fourier", "Glider", "Spatial", "Temporal"]
gifs_paths, gifs_names = get_paths_and_names()
results = {}

for i in range(2): #range(len(gifs_names)):
    print('----- ' + repr(i+1) + '/' + repr(len(gifs_names)) + ': ' + gifs_names[i] + ' -----')
    start_time = time.time()

    name = gifs_names[i]
    path = gifs_paths[i]
    results[name] = {}

    # Parse video's name (for previous project's videos only!!!)
    results[name] = prev_project_video_name_parser.parse_name_to_dict(name)

    frames = VideoUtils.read_frames(path)
    photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
        PhotoreceptorImageConverter.make_gaussian_kernel(15),
        frames[0].shape, 6000)
    frames = photoreceptor.receive(frames)  # A list of frames
    print('Converted to photoreceptor: ' + repr(time.time() - start_time))

    frames_rand = VideoUtils.randomize(frames)
    for j in range(len(templates_list)):
        template = templates_list[j]
        template_name = templates_names[j]

        local_motion = EMD.forward_video(frames, template, axis=0, center=True)
        local_motion_rand = EMD.forward_video(frames_rand, template, axis=0, center=True)
        results[name][template_name + '_X'] = calc_results_score(local_motion, local_motion_rand)

        local_motion = EMD.forward_video(frames, template, axis=1, center=True)
        local_motion_rand = EMD.forward_video(frames_rand, template, axis=1, center=True)
        results[name][template_name + '_Y'] = calc_results_score(local_motion, local_motion_rand)

        print('Calculated ' + template_name)

    end_time = time.time()
    print('Total time: ' + repr(end_time - start_time) + ' seconds')

print()

# Write to file (chen: I copies this code from stack overflow so don't ask me what's going on)
file_name = 'test_results.csv'
file = open(file_name, 'w+', newline='')
fields = ["Name", "Object", "Movement", "Tripod", "ChickenFence", "Fourier_X", "Fourier_Y", "Glider_X", "Glider_Y", "Spatial_X", "Spatial_Y", "Temporal_X", "Temporal_Y"]
for key in fields:
    file.write(key + ',')
file.write('\n')
w = csv.DictWriter(file, fields)
for key,val in sorted(results.items()):
    row = {'Name': key}
    row.update(val)
    w.writerow(row)
