import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

best_option_path = 'saved_results/15_01_2023/best_option.pickle'
best_option = pd.read_pickle(best_option_path)
general_DS_folder = os.path.join('D:\Data_Sets', 'results', '21_11_2022', '16_33.pickle')
path = 'saved_results/15_01_2023/gamma = (0.5), lr = (0.05), batch_size = (2).pickle'
obj = pd.read_pickle(path)

to_show_results = obj['results'][590][0][6]
show_origin = obj['test_dict'][590][1][6]
mean_to_show = np.mean(to_show_results)

to_show_tresh_zero = np.where(to_show_results < 0, 0, 1)
to_show_tresh_mean = np.where(to_show_results < mean_to_show, 0, 1)

# cv2.imwrite('color_img.jpg', to_show_tresh_zero)
image = to_show_tresh_mean.astype(np.uint8)
cv2.imshow("original", show_origin.astype(np.uint8)*255)
cv2.imshow("tresh_zero", to_show_tresh_zero.astype(np.uint8)*255)
cv2.imshow("tresh_mean", to_show_tresh_mean.astype(np.uint8)*255)
cv2.waitKey()
# plt.imshow(to_show_tresh_zero)
# plt.show()
print('post_process')