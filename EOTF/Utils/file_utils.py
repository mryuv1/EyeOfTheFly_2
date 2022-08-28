import os

def load_or_create(path, load_func, create_func, save_func):
    if os.path.isfile(path):
        return load_func(path)
    res = create_func(path)
    save_func(path, res)
    return res

def results_path(clip_path, file_name = ''):
    """
    Creates a results directory for a given clip.
    :param clip_path:
    :return: Path to the results directory
    """

    clip_path = os.path.splitext(clip_path)[0]
    base_path, name = os.path.split(clip_path)
    res = base_path + '\\' + name + '\\'
    if not os.path.exists(res):
        os.makedirs(res)
    return res + file_name