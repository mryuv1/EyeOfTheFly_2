import EOTF.PhotoreceptorImageConverter as PhotoreceptorImageConverter
import EOTF.EMD as EMD
import EOTF.Utils.VideoUtils as VideoUtils
import time
import csv
import sys


def get_paths_and_names():
    """
    Change this function to select the specific gifs you want to test.
    :return: a list of gif paths, and a list of corresponding meaningful names for the gifs
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


def calc_results_score(emd_results):
    """
    Change this function to determine the way you quantify the emd response (examples: Total Variation, Sum of Squares)
    :param emd_results:
    :return:
    """
    return VideoUtils.total_variation_video(emd_result)


templates_list = [EMD.TEMPLATE_FOURIER, EMD.TEMPLATE_GLIDER, EMD.TEMPLATE_SPATIAL, EMD.TEMPLATE_TEMPORAL]
templates_names = ["Fourier", "Glider", "Spatial", "Temporal"]
gifs_paths, gifs_names = get_paths_and_names()
results = {}

for i in range(len(gifs_names)):
    print(gifs_names[i], end=' ')
    start_time = time.time()

    name = gifs_names[i]
    path = gifs_paths[i]
    results[name] = {}
    frames = VideoUtils.read_frames(path)
    photoreceptor = PhotoreceptorImageConverter.PhotoreceptorImageConverter(
        PhotoreceptorImageConverter.make_gaussian_kernel(15),
        frames[0].shape, 6000)
    frames = photoreceptor.receive(frames)  # A list of frames

    for j in range(len(templates_list)):
        template = templates_list[j]
        template_name = templates_names[j]
        emd_result = EMD.forward_video(frames, template, axis=0, center=False)
        results[name][template_name + '_X'] = calc_results_score(emd_result)
        emd_result = EMD.forward_video(frames, template, axis=1, center=False)
        results[name][template_name + '_Y'] = calc_results_score(emd_result)

    end_time = time.time()
    print(', total time: ' + repr(end_time - start_time) + ' seconds')

print()
print('End results:')
fields = ["Animation", "Fourier_X", "Fourier_Y", "Glider_X", "Glider_Y", "Spatial_X", "Spatial_Y", "Temporal_X", "Temporal_Y"]
for key in fields:
    print(key, end=',')
print()
w = csv.DictWriter(sys.stdout, fields)
for key,val in sorted(results.items()):
    row = {'Animation': key}
    row.update(val)
    w.writerow(row)