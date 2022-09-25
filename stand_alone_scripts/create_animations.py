import cv2 as cv
import os
import video_utils
from EOTF import PhotoreceptorImageConverter

def create_left_animation(frame, length=15, speed=1, new_size = None):
    if new_size is None:
        new_size = [frame.shape[1], frame.shape[0]]
    small_size = [round(frame.shape[0]/2), round(frame.shape[1]/2)]
    col_start = round(frame.shape[1] / 2 - small_size[1] / 2)
    col_end = round(frame.shape[1] / 2 + small_size[1] / 2)
    row_start = round(frame.shape[0] / 2 - small_size[0] / 2)
    row_end = round(frame.shape[0] / 2 + small_size[0] / 2)
    new_frames = [
        cv.resize(
            frame[row_start:row_end, (col_start + i * speed):(col_end + i * speed)],
            new_size
        )
        for i in range(length)
    ]
    return new_frames

def create_down_animation(frame, length=15, speed=1, new_size = None):
    if new_size is None:
        new_size = [frame.shape[1], frame.shape[0]]
    small_size = [round(frame.shape[0]/2), round(frame.shape[1]/2)]
    col_start = round(frame.shape[1]/2 - small_size[1]/2)
    col_end = round(frame.shape[1]/2 + small_size[1]/2)
    row_start = round(frame.shape[0]/2 - small_size[0]/2)
    row_end = round(frame.shape[0]/2 + small_size[0]/2)
    new_frames = [
        cv.resize(
            frame[(row_start + i * speed):(row_end + i * speed), col_start:col_end],
            new_size
        )
        for i in range(length)
    ]
    return new_frames

def create_right_animation(frame, length=15, speed=1, new_size=None):
    r = create_left_animation(frame,length,speed,new_size)
    r.reverse()
    return r

def create_up_animation(frame, length=15, speed=1, new_size=None):
    r = create_down_animation(frame,length,speed, new_size)
    r.reverse()
    return r

def create_zoomin_animation(frame, length=15, speed=1, new_size=None):
    if new_size is None:
        new_size = [frame.shape[1], frame.shape[0]]
    small_size = [round(frame.shape[0]/2), round(frame.shape[1]/2)]
    col_start = round(frame.shape[1] / 2 - small_size[1] / 2)
    col_end = round(frame.shape[1] / 2 + small_size[1] / 2)
    row_start = round(frame.shape[0] / 2 - small_size[0] / 2)
    row_end = round(frame.shape[0] / 2 + small_size[0] / 2)
    new_frames = [
        cv.resize(
            frame[(row_start + speed * i):(row_end - speed * i), (col_start + speed * i):(col_end - speed * i)],
            new_size
        )
        for i in range(length)
    ]
    return new_frames

def parse_name_to_dict(name):
    name_parts = name.split('_')
    if len(name_parts) != 3:
        return {}
    return {'Object': name_parts[0],
              'Movement': name_parts[1],
              'Speed': name_parts[2]}


def create_zoomout_animation(frame, length=15, speed=1, new_size=None):
    r = create_zoomin_animation(frame,length,speed,new_size)
    r.reverse()
    return r

if __name__ == '__main__':
    base_path = r'../data/nature_images'
    files = os.listdir(base_path)
    files = [f for f in files if ('.jpg' in f)] # keep only gifs
    files_names = [os.path.splitext(f)[0] for f in files]
    files_paths = [base_path + '/' + f for f in files]
    gifs_path = os.path.join(base_path,'gifs/')
    if not os.path.exists(gifs_path):
        os.makedirs(gifs_path)
    for f,name,path in zip(files,files_names,files_paths):
        frame = cv.imread(path, cv.IMREAD_GRAYSCALE)
        for s in [1,3,5,10,15]:
            video_utils.save_gif(create_left_animation(frame, new_size=(400,400), speed=s), name+'_'+'left'+'_'+repr(s)+'.gif', path=gifs_path)
            video_utils.save_gif(create_down_animation(frame, new_size=(400,400), speed=s), name+'_'+'down'+'_'+repr(s)+'.gif', path=gifs_path)
            video_utils.save_gif(create_zoomin_animation(frame, new_size=(400,400), speed=s), name+'_'+'zoomin'+'_'+repr(s)+'.gif', path=gifs_path)