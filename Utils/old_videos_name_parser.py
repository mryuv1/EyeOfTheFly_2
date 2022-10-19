import os


def get_paths_and_names(base_path=r'data\RealInputClips2\All Videos'):
    files = os.listdir(base_path)
    files_names = [os.path.splitext(f)[0] for f in files]
    files_paths = [base_path + '\\' + f for f in files]

    return files_paths, files_names


def split_name(name):
    return name.split('_')


def get_object(name_parts):
    video_objects_list = ['Pillar', 'Corner', 'Wall Edge']
    return video_objects_list[ord(name_parts[0]) - ord('A')]


def get_movement(name_parts):
    video_movement_list = ['Camera Right', 'Camera Left', 'Camera Looming', 'Camera Receding',
                           'Camera Diagonal Looming', 'Camera Diagonal Receding', 'Camera Up', 'Camera Down',
                           'Object Right', 'Object Left', 'Object Receding', 'Object Looming']
    return video_movement_list[int(name_parts[3]) - 1]


def get_tripod(name_parts):
    return False if name_parts[4] == 'HAND' else True


def get_chicken_fence(name_parts):
    return True if name_parts[5] == 'Y' else False


def parse_name_to_dict(name):
    name_parts = split_name(name)
    if len(name_parts) != 9:
        return {}
    result = {'Object': get_object(name_parts),
              'Movement': get_movement(name_parts),
              'Tripod': get_tripod(name_parts),
              'ChickenFence': get_chicken_fence(name_parts)}
    return result
