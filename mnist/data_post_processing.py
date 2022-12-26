import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


general_DS_folder = os.path.join('D:\Data_Sets', 'results', '21_11_2022', '16_33.pickle')
path = 'saved_results/21_11_2022/16:01.pickle'
obj = pd.read_pickle(general_DS_folder)

to_show = obj['stunt'][13][8]
mean_to_show = np.mean(to_show)

to_show = np.where(to_show < 0, 0, 1)

plt.imshow(to_show)
plt.show()
print('post_process')