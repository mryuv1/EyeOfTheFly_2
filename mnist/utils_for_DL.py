import cv2
import os
# import matplotlib.pyplot as plt
import numpy as np


def read_all_pictures_in_file(dir, desiered_dim=None, number_of_frames=np.inf) -> list:
    final_list = list()
    pics_names = os.listdir(dir)
    loop_idx = 0
    for pic in pics_names:
        loop_idx += 1
        pic_tmp = cv2.imread(os.path.join(dir, pic))
        gray_tmp = cv2.cvtColor(pic_tmp, cv2.COLOR_BGR2GRAY)

        if desiered_dim:
            gray_tmp = cv2.resize(gray_tmp, desiered_dim)
        final_list.append(gray_tmp)
        if loop_idx == number_of_frames:
            break

    return final_list


def create_data_tuple(dir, number_of_videos=5, desiered_dim=None, number_of_frames=np.inf) -> dict:
    annotations_path = os.path.join(dir, 'Annotations', '480p')
    JPEG_path = os.path.join(dir, 'JPEGImages', '480p')
    anotations_list = os.listdir(annotations_path)
    dataset_dict = {}
    for video_idx in range(number_of_videos):
        annototion_vid_path = os.path.join(annotations_path, anotations_list[video_idx])
        JPEG_path_vid_path = os.path.join(JPEG_path, anotations_list[video_idx])

        annotation_list = read_all_pictures_in_file(annototion_vid_path, desiered_dim=desiered_dim,
                                                    number_of_frames=number_of_frames)
        # for current use we want the segmentaion to be only 0 or 1
        annotation_list = [np.where(frame > 0, 1, 0) for frame in annotation_list]
        JPEG_list = read_all_pictures_in_file(JPEG_path_vid_path, desiered_dim=desiered_dim,
                                              number_of_frames=number_of_frames)

        dataset_dict[anotations_list[video_idx]] = (JPEG_list, annotation_list)
    return dataset_dict


if __name__ == '__main__':
    general_DS_folser = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')

    dataset_dict = create_data_tuple(general_DS_folser, desiered_dim=(200, 100))
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
