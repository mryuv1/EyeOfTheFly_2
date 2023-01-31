from EOTF import EMD
from Utils.writing_results_utils import progress_bar
import numpy as np
from threading import Thread


def is_preprocessed(data_dict):
    if data_dict is dict:
        data_dict = list(data_dict.values())
    return type(data_dict[0][0][0]) is not np.ndarray


def preprocess(dataset, preprocess_type, print_progress=False):
    def _emd(dataset, k, print_progress):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization
        frames_preprocessed = [
            #frames_orig[:-1],
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_SPATIAL, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_SPATIAL, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_TEMPORAL, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_TEMPORAL, axis=1),
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])

    def _duplicate(dataset, k, print_progress):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization
        frames_preprocessed = [
            #frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1]
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])

    def _random_emd(dataset, k, print_progress):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            # frames_orig[:-1],
            EMD.forward_video(frames_orig, templates[0], axis=0),
            EMD.forward_video(frames_orig, templates[1], axis=1),
            EMD.forward_video(frames_orig, templates[2], axis=0),
            EMD.forward_video(frames_orig, templates[3], axis=0),
            EMD.forward_video(frames_orig, templates[4], axis=0),
            EMD.forward_video(frames_orig, templates[5], axis=0),
            EMD.forward_video(frames_orig, templates[6], axis=0),
            EMD.forward_video(frames_orig, templates[7], axis=1)
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])

    # for random emd preprocessing
    templates = [
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2)
    ]

    if preprocess_type == 'emd':
        preprocess_func = _emd
    elif preprocess_type == 'duplicate':
        preprocess_func = _duplicate
    elif preprocess_type == 'random_emd':
        preprocess_func = _random_emd

    if print_progress:
        print(progress_bar(0, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    for i, k in enumerate(list(dataset.keys())):
        x = Thread(target=preprocess_func, args=(dataset, k, print_progress))
        x.start()
        x.join()
        if print_progress:
            print(progress_bar(i, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    if print_progress:
        print()

    return dataset


def emd_preprocess(dataset, print_progress=False):
    if print_progress:
        print(progress_bar(0, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    for i, k in enumerate(list(dataset.keys())):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255.0 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            #frames_orig[:-1],
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_SPATIAL, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_SPATIAL, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_TEMPORAL, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_TEMPORAL, axis=1),
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])
        if print_progress:
            print(progress_bar(i, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    return dataset


def duplicate_preprocess(dataset, print_progress=False):
    if print_progress:
        print(progress_bar(0, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    for i,k in enumerate(list(dataset.keys())):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            #frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1],
            frames_orig[:-1]
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])
        if print_progress:
            print(progress_bar(i, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    return dataset


def random_emd_preprocess(dataset, print_progress=False):
    if print_progress:
        print(progress_bar(0, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')

    templates = [
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2),
        np.random.rand(2, 2)
    ]
    for i, k in enumerate(list(dataset.keys())):
        frames_orig = dataset[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            #frames_orig[:-1],
            EMD.forward_video(frames_orig, templates[0], axis=0),
            EMD.forward_video(frames_orig, templates[1], axis=1),
            EMD.forward_video(frames_orig, templates[2], axis=0),
            EMD.forward_video(frames_orig, templates[3], axis=0),
            EMD.forward_video(frames_orig, templates[4], axis=0),
            EMD.forward_video(frames_orig, templates[5], axis=0),
            EMD.forward_video(frames_orig, templates[6], axis=0),
            EMD.forward_video(frames_orig, templates[7], axis=1)
        ]
        dataset[k] = (frames_preprocessed, dataset[k][1][:-1])
        if print_progress:
            print(progress_bar(i, len(dataset), length=50, prefix='Preprocessed: ', decimals=2), end='')
    return dataset


def change_data_type(dataset, target_type):
    if type(dataset) is list:
        inds = range(len(dataset))
    if type(dataset) is dict:
        inds = dataset.keys()

    if is_preprocessed(dataset):
        for k in inds:
            dataset[k] = ([[b.astype(target_type) for b in a] for a in dataset[k][0]],
                          [[b.astype(target_type) for b in a] for a in dataset[k][1]])
    else:
        for k in inds:
            dataset[k] = ([a.astype(target_type) for a in dataset[k][0]],
                          [a.astype(target_type) for a in dataset[k][1]])
