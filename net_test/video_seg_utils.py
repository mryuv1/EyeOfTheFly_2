import cv2
import os
import pickle
from video_seg_main import CreateDatasetArgs
from preprocess_utils import *


def read_frames_from_file(filename, desired_dim=None, frames=None) -> list:
    final_list = list()
    pics_names = os.listdir(filename)
    if not frames:
        frames = range(len(pics_names))
    for pic_ind in frames:
        pic_name = pics_names[pic_ind]
        pic_tmp = cv2.imread(os.path.join(filename, pic_name))
        gray_tmp = cv2.cvtColor(pic_tmp, cv2.COLOR_BGR2GRAY)
        if desired_dim:
            gray_tmp = cv2.resize(gray_tmp, desired_dim)
        final_list.append(gray_tmp)
    return final_list


def load_dataset(filename, number_of_videos=np.inf, desired_dim=None, frames=None) -> list:
    annotations_path = os.path.join(filename, 'Annotations', '480p')
    JPEG_path = os.path.join(filename, 'JPEGImages', '480p')
    annotations_list = os.listdir(annotations_path)
    if len(annotations_list) < number_of_videos:
        number_of_videos = len(annotations_list)
    dataset = []
    for video_idx in range(number_of_videos):
        annototion_vid_path = os.path.join(annotations_path, annotations_list[video_idx])
        JPEG_vid_path = os.path.join(JPEG_path, annotations_list[video_idx])

        annotation_list = read_frames_from_file(annototion_vid_path, frames=frames, desired_dim=desired_dim)
        # for current use we want the segmentaion to be only 0 or 1
        annotation_list = [np.where(frame > 0, 1, 0) for frame in annotation_list]
        JPEG_list = read_frames_from_file(JPEG_vid_path, frames=frames, desired_dim=desired_dim)

        dataset.append((JPEG_list, annotation_list))
    return dataset


def train_test_split(dataset, train_part=0.8):
    number_of_videos = len(dataset)
    if type(dataset) is dict:
        indexes = list(dataset.keys())
    else:
        indexes = list(range(number_of_videos))
    np.random.shuffle(indexes)
    train_indexes = indexes[0:int(np.round(train_part * number_of_videos))]
    test_indexes = indexes[int(np.round(train_part * number_of_videos)) + 1::]

    train_dict = {}
    test_dict = {}
    results_dict = {}

    for idx in train_indexes:
        train_dict[idx] = dataset[idx]

    for idx in test_indexes:
        test_dict[idx] = dataset[idx]
        results_dict[idx] = list()

    return train_dict, test_dict, results_dict


def train_test_combine(dataset1, dataset2):
    return {**dataset1,**dataset2}


def create_dataset(dataset_dir,
                   args: CreateDatasetArgs,
                   override_saved_data=False,
                   save_new_data=True):
    data_pickle_path = DataPickleNamer.create(args.image_size, args.number_of_videos, args.preprocess_type)

    # Load pickle if exists
    if not override_saved_data:
        if not os.path.isdir('data_pickles'):
            os.makedirs('data_pickles')
        if os.path.isfile(data_pickle_path) and DataPickleNamer.check(data_pickle_path, args.image_size, args.number_of_videos):
            with open(data_pickle_path, 'rb') as f:
                train_dict, test_dict, results_dict = pickle.load(f)
            print('Loaded dataset from pickle.')
            return train_dict, test_dict, results_dict

    # Load data
    print('Pickle not loaded. Creating dataset...')
    dataset = load_dataset(dataset_dir, number_of_videos=args.number_of_videos, frames=range(args.video_length+1), desired_dim=args.image_size) \
              + load_dataset(dataset_dir, number_of_videos=args.number_of_videos, frames=range(-(args.video_length+1), 0), desired_dim=args.image_size)

    # Add augmentations
    print('Adding augmentations...')
    n = len(dataset)
    for i in range(n):
        element = dataset[i]
        dataset.append((
            [np.fliplr(f) for f in element[0]],
            [np.fliplr(f) for f in element[1]]
        ))
        dataset.append((
            [np.flipud(f) for f in element[0]],
            [np.flipud(f) for f in element[1]]
        ))
        for k in [1, 2, 3]:
            dataset.append((
                [np.rot90(f, k) for f in element[0]],
                [np.rot90(f, k) for f in element[1]]
            ))

    if args.preprocess_type:
        print('Starting Preprocess...')
        dataset = preprocess(dataset, preprocess_type=args.preprocess_type)
        print('Finished Preprocess.')

    if args.type:
        change_data_type(dataset, args.type)

    train_dict, test_dict, results_dict = train_test_split(dataset, args.train_part)

    # save pickle
    if save_new_data:
        # TODO: this part takes a lot of time, maybe it doesn't override previous data
        with open(data_pickle_path, 'wb+') as f:
            pickle.dump((train_dict, test_dict, results_dict), f)
        print('Saved data to file.')

    return train_dict, test_dict, results_dict


class DataPickleNamer:
    @staticmethod
    def create(desired_dim, number_of_videos, preprocess_type):
        name = 'x'.join(str(d) for d in desired_dim)
        if number_of_videos is not np.inf:
            name = name + '_num=' + str(number_of_videos)
        if preprocess_type is not None:
            name = name + '_pre=' + preprocess_type
        name = name + '.pickle'
        return os.path.join('data_pickles', name)

    @staticmethod
    def check(data_pickle_path, desired_dim, number_of_videos, preprocess_type):
        try:
            data_pickle_name = os.path.basename(data_pickle_path)
            data_pickle_name, path_extention = os.path.splitext(data_pickle_name)

            name_parts = data_pickle_name.split('_')
            data_pickle_name = name_parts[0]
            path_number_of_videos = np.inf
            path_preprocess_type = None
            if len(name_parts) > 1:
                for n in name_parts[1:]:
                    n = n.split('=')
                    if n[0] == 'num':
                        path_number_of_videos = int(n[1])
                    elif n[0] == 'pre':
                        path_preprocess_type = n[1]
            path_desired_dim = tuple([eval(i) for i in data_pickle_name.split('x')])
        except:
            return False
        return (path_extention == '.pickle') \
               and (path_desired_dim == desired_dim) \
               and (path_number_of_videos == number_of_videos) \
               and (preprocess_type == path_preprocess_type)


def save_results_to_date_file(data_to_save, file_name: str = None):
    from datetime import datetime, date

    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    if not file_name:
        now = datetime.now()
        file_name = now.strftime("%H:%M")

    path = os.path.join('saved_results', d1)
    if not os.path.isdir(path):
        os.mkdir(path)

    path = os.path.join(path, file_name)
    with open(path + '.pickle', 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    general_DS_folser = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')

    dataset_dict = load_dataset(general_DS_folser)
    print(dataset_dict.keys())

    plt.imshow(dataset_dict['bear'][0][2], cmap='gray')
    new_dim = (dataset_dict['bear'][0][2].shape[1] // 4, dataset_dict['bear'][0][2].shape[0] // 4)
    resized = cv2.resize(dataset_dict['bear'][0][2], new_dim)
    plt.imshow(resized, cmap='gray')
    plt.show()

    plt.imshow(dataset_dict['bear'][1][2], cmap='gray')
    new_dim = (dataset_dict['bear'][1][2].shape[1] // 4, dataset_dict['bear'][0][2].shape[0] // 4)
    resized = cv2.resize(dataset_dict['bear'][1][2], new_dim)
    plt.imshow(resized, cmap='gray')
    plt.show()

    plt.imshow(dataset_dict['bmx-trees'][0][2], cmap='gray')
    new_dim = (dataset_dict['bmx-trees'][0][2].shape[1] // 4, dataset_dict['bmx-trees'][0][2].shape[0] // 4)
    resized = cv2.resize(dataset_dict['bmx-trees'][0][2], new_dim)
    plt.imshow(resized, cmap='gray')
    plt.show()

    plt.imshow(dataset_dict['bmx-trees'][1][2], cmap='gray')
    new_dim = (dataset_dict['bmx-trees'][1][2].shape[1] // 4, dataset_dict['bmx-trees'][0][2].shape[0] // 4)
    resized = cv2.resize(dataset_dict['bmx-trees'][1][2], new_dim)
    plt.imshow(resized, cmap='gray')
    plt.show()
    print("check in the unils_for_DL file")
