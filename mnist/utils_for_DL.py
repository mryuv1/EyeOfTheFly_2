import cv2
import os
import matplotlib.pyplot as plt



def read_all_pictures_in_file(dir, desiered_dim=None) -> list:
    final_list = list()
    pics_names = os.listdir(dir)
    for pic in pics_names:
        pic_tmp = cv2.imread(os.path.join(dir, pic))
        gray_tmp = cv2.cvtColor(pic_tmp, cv2.COLOR_BGR2GRAY)

        if desiered_dim:
            gray_tmp = cv2.resize(gray_tmp, desiered_dim)
        final_list.append(gray_tmp)

    return final_list


def create_data_tuple(dir, number_of_videos=5, desiered_dim=None) -> dict:
    annotations_path = os.path.join(dir, 'Annotations', '480p')
    JPEG_path = os.path.join(dir, 'JPEGImages', '480p')
    anotations_list = os.listdir(annotations_path)
    dataset_dict = {}
    for video_idx in range(number_of_videos):
        annototion_vid_path = os.path.join(annotations_path, anotations_list[video_idx])
        JPEG_path_vid_path = os.path.join(JPEG_path, anotations_list[video_idx])

        annotation_list = read_all_pictures_in_file(annototion_vid_path, desiered_dim=desiered_dim)
        JPEG_list = read_all_pictures_in_file(JPEG_path_vid_path, desiered_dim=desiered_dim)

        dataset_dict[anotations_list[video_idx]] = (JPEG_list, annotation_list)
    return dataset_dict



if __name__ == '__main__':
    general_DS_folser = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')

    dataset_dict = create_data_tuple(general_DS_folser, desiered_dim=(220, 120))
    print(dataset_dict.keys())

    plt.imshow(dataset_dict['bear'][0][2], cmap='gray')
    new_dim = (dataset_dict['bear'][0][2].shape[1]//4, dataset_dict['bear'][0][2].shape[0]//4)
    resized = cv2.resize(dataset_dict['bear'][0][2], new_dim)
    plt.imshow(resized, cmap='gray')